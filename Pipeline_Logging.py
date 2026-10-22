"""
Run logging for the ingest pipeline: a timestamped copy of everything printed to the terminal, plus a
real-time CPU / RAM / pagefile / GPU / VRAM trace, so a crash can be explained afterwards.

Files are written to <repo>/logs/ and flushed on every write, so they survive a hard kill:
  <name>_<YYYYmmdd_HHMMSS>.log        terminal output (stdout + stderr + logging), timestamped per line
  <name>_<YYYYmmdd_HHMMSS>_resources.csv   one row every few seconds (open in Excel to graph it)
  <name>_<YYYYmmdd_HHMMSS>_crash.txt  faulthandler dump if this Python process itself dies natively

Usage:
    import Pipeline_Logging
    monitor = Pipeline_Logging.start("pipeline")     # tees console, starts the resource monitor
    monitor.mark("doc start: foo.pdf")               # annotate the CSV
    peaks = monitor.mark("doc end")                  # peaks since the previous mark()

The monitor samples the whole machine, this process and all its children (the per-document pipeline
process, OCR helpers, ...) and every Ollama process, so it also shows a model server dying or growing.

Environment:
    PIPELINE_MONITOR_INTERVAL   seconds between samples (default 2)
    PIPELINE_LOG_DIR            override the log folder
"""
import atexit
import csv
import datetime
import faulthandler
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = Path(os.environ.get("PIPELINE_LOG_DIR") or REPO_ROOT / "logs")
GB = 1024 ** 3
MB = 1024 ** 2

# Warn (once per episode) when a resource gets this tight.
WARN_RAM_AVAILABLE_GB = 1.5
WARN_PAGEFILE_PCT = 90
WARN_VRAM_PCT = 95

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


_last_sync = [0.0]


class _Tee:
    """File-like that forwards to the real stream and also appends timestamped lines to the log file."""

    def __init__(self, stream, log_file, lock):
        self._stream = stream
        self._log = log_file
        self._lock = lock
        self._at_line_start = True
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, text):
        if not text:
            return 0
        try:
            self._stream.write(text)
        except Exception:
            pass  # a closed/odd console must never stop logging to the file
        try:
            normalized = text.replace("\r\n", "\n").replace("\r", "\n")
            with self._lock:
                for piece in normalized.splitlines(keepends=True):
                    if self._at_line_start:
                        self._log.write(datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3] + " ")
                    self._log.write(piece)
                    self._at_line_start = piece.endswith("\n")
                self._log.flush()
                # Force to disk (throttled) so the tail survives a BSOD / reset / power loss.
                now = time.time()
                if now - _last_sync[0] >= 1.0:
                    _last_sync[0] = now
                    os.fsync(self._log.fileno())
        except Exception:
            pass
        return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

    def __getattr__(self, name):  # isatty(), fileno(), ... fall through to the real stream
        return getattr(self._stream, name)


def _gb(n):
    return round(n / GB, 2)


class ResourceMonitor(threading.Thread):
    """Background sampler writing resource rows to a CSV (flushed per row) and warnings to the log."""

    COLUMNS = [
        "time", "elapsed_s", "event",
        "cpu_pct", "ram_used_gb", "ram_avail_gb", "ram_pct", "pagefile_used_gb", "pagefile_pct",
        "tree_procs", "tree_rss_gb", "tree_cpu_pct",
        "ollama_procs", "ollama_rss_gb", "ollama_cpu_pct",
        "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_temp_c", "gpu_power_w",
        "top_procs",
    ]
    # Columns whose maximum is tracked between mark() calls.
    PEAK_COLUMNS = ["tree_rss_gb", "ollama_rss_gb", "ram_used_gb", "pagefile_used_gb", "gpu_mem_used_mb", "cpu_pct"]

    def __init__(self, csv_path, interval=2.0):
        super().__init__(name="ResourceMonitor", daemon=True)
        self.interval = interval
        self._csv_path = csv_path
        self._stop_event = threading.Event()
        self._start_time = time.time()
        self._procs = {}  # pid -> psutil.Process (kept so cpu_percent has a previous reading)
        self._pending_event = ""
        self._peaks = {}
        self._peak_lock = threading.Lock()
        self._warned = set()
        self._have_gpu = True
        self._gpu_failures = 0
        self.last_row = {}
        psutil.cpu_percent(None)

    # ---- public ----
    def mark(self, event):
        """Annotate the next CSV row and return (and reset) the peak values seen since the last mark."""
        with self._peak_lock:
            peaks, self._peaks = self._peaks, {}
            self._pending_event = event
        logging.info(f"[MON] {event}")
        return peaks

    def stop(self):
        self._stop_event.set()

    # ---- sampling ----
    def _gpu(self):
        if not self._have_gpu:
            return {}
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, creationflags=_CREATE_NO_WINDOW,
            )
            if out.returncode != 0:
                raise RuntimeError(out.stderr.strip() or f"exit {out.returncode}")
            util, used, total, temp, power = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]

            def num(x):
                try:
                    return float(x)
                except ValueError:
                    return ""  # "[N/A]"
            self._gpu_failures = 0
            return {"gpu_util_pct": num(util), "gpu_mem_used_mb": num(used), "gpu_mem_total_mb": num(total),
                    "gpu_temp_c": num(temp), "gpu_power_w": num(power)}
        except FileNotFoundError:
            self._have_gpu = False
            logging.warning("[MON] nvidia-smi not found; GPU stats disabled.")
        except Exception as e:
            self._gpu_failures += 1
            # nvidia-smi failing/hanging mid-run usually means the GPU driver is in trouble: say so.
            if self._gpu_failures in (1, 5, 30):
                logging.warning(f"[MON] nvidia-smi failed ({self._gpu_failures}x): {e}")
        return {}

    def _tracked_processes(self):
        """(tree, ollama): this process's descendants plus itself, and all Ollama processes."""
        me = psutil.Process()
        tree, ollama = {}, {}
        try:
            candidates = [me] + me.children(recursive=True)
        except psutil.Error:
            candidates = [me]
        for p in candidates:
            tree[p.pid] = p
        for p in psutil.process_iter(["name"]):
            try:
                # llama-server.exe is Ollama's model runner: the process that actually holds the loaded models.
                if (p.info["name"] or "").lower().startswith(("ollama", "llama-server")):
                    ollama[p.pid] = p
            except psutil.Error:
                continue
        return tree, ollama

    def _measure(self, procs):
        """Sum RSS and CPU% over the given processes, returning (count, rss_bytes, cpu_pct, details)."""
        rss = cpu = 0.0
        details = []
        for pid, proc in procs.items():
            proc = self._procs.setdefault(pid, proc)
            try:
                with proc.oneshot():
                    mem = proc.memory_info().rss
                    c = proc.cpu_percent(None)
                    name = proc.name()
                rss += mem
                cpu += c
                details.append((mem, name, pid))
            except psutil.Error:
                self._procs.pop(pid, None)
        return len(details), rss, cpu, details

    def sample(self):
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()  # on Windows: the pagefile
        tree, ollama = self._tracked_processes()
        n_tree, tree_rss, tree_cpu, tree_details = self._measure(tree)
        n_oll, oll_rss, oll_cpu, oll_details = self._measure(ollama)
        for pid in set(self._procs) - set(tree) - set(ollama):
            self._procs.pop(pid, None)

        top = sorted(tree_details + oll_details, reverse=True)[:3]
        row = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(time.time() - self._start_time, 1),
            "cpu_pct": psutil.cpu_percent(None),
            "ram_used_gb": _gb(vm.total - vm.available), "ram_avail_gb": _gb(vm.available), "ram_pct": vm.percent,
            "pagefile_used_gb": _gb(sw.used), "pagefile_pct": sw.percent,
            "tree_procs": n_tree, "tree_rss_gb": _gb(tree_rss), "tree_cpu_pct": round(tree_cpu, 1),
            "ollama_procs": n_oll, "ollama_rss_gb": _gb(oll_rss), "ollama_cpu_pct": round(oll_cpu, 1),
            "top_procs": "; ".join(f"{name}[{pid}]={mem // MB}MB" for mem, name, pid in top),
        }
        row.update(self._gpu())
        return row

    def _check_warnings(self, row):
        gpu_pct = ""
        if row.get("gpu_mem_total_mb"):
            gpu_pct = 100 * row["gpu_mem_used_mb"] / row["gpu_mem_total_mb"]
        checks = {
            "ram": (row["ram_avail_gb"] < WARN_RAM_AVAILABLE_GB,
                    f"low RAM: {row['ram_avail_gb']} GB available ({row['ram_pct']}% used)"),
            "pagefile": (row["pagefile_pct"] > WARN_PAGEFILE_PCT,
                         f"pagefile {row['pagefile_pct']}% used ({row['pagefile_used_gb']} GB)"),
            "vram": (gpu_pct != "" and gpu_pct > WARN_VRAM_PCT,
                     f"GPU memory {row.get('gpu_mem_used_mb')}/{row.get('gpu_mem_total_mb')} MB "
                     f"({gpu_pct if gpu_pct == '' else round(gpu_pct)}%)"),
        }
        for key, (bad, message) in checks.items():
            if bad and key not in self._warned:
                self._warned.add(key)
                logging.warning(f"[MON] {message}; top: {row['top_procs']}")
            elif not bad:
                self._warned.discard(key)

    def run(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        last_summary = 0.0
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, restval="")
            writer.writeheader()
            f.flush()
            while not self._stop_event.is_set():
                started = time.time()
                try:
                    row = self.sample()
                    with self._peak_lock:
                        row["event"], self._pending_event = self._pending_event, ""
                        for col in self.PEAK_COLUMNS:
                            if isinstance(row.get(col), (int, float)):
                                self._peaks[col] = max(self._peaks.get(col, 0), row[col])
                    writer.writerow(row)
                    f.flush()
                    os.fsync(f.fileno())  # every row: this is the evidence if the whole machine dies
                    self.last_row = row
                    self._check_warnings(row)
                    if started - last_summary >= 60:
                        last_summary = started
                        logging.info(
                            "[MON] cpu %s%% | ram %s GB used, %s GB free | pagefile %s GB (%s%%) | "
                            "pipeline tree %s GB / %s%% cpu | ollama %s GB | gpu %s%% util, %s/%s MB, %s C",
                            row["cpu_pct"], row["ram_used_gb"], row["ram_avail_gb"], row["pagefile_used_gb"],
                            row["pagefile_pct"], row["tree_rss_gb"], row["tree_cpu_pct"], row["ollama_rss_gb"],
                            row.get("gpu_util_pct", "n/a"), row.get("gpu_mem_used_mb", "n/a"),
                            row.get("gpu_mem_total_mb", "n/a"), row.get("gpu_temp_c", "n/a"),
                        )
                except Exception:
                    logging.warning("[MON] sample failed:\n" + traceback.format_exc())
                self._stop_event.wait(max(0.1, self.interval - (time.time() - started)))


_state = {"monitor": None, "log_file": None}


def start(name="pipeline", monitor=True):
    """Tee the console to a timestamped log, install crash hooks and (optionally) start the monitor."""
    if _state["log_file"] is not None:
        return _state["monitor"]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = LOG_DIR / f"{name}_{stamp}"
    log_file = open(f"{base}.log", "a", encoding="utf-8", errors="replace")
    _state["log_file"] = log_file
    lock = threading.Lock()

    # A cp1252 console cannot print the pipeline's check marks; never let that raise mid-run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    sys.stdout = _Tee(sys.stdout, log_file, lock)
    sys.stderr = _Tee(sys.stderr, log_file, lock)

    # Native crashes (access violation, abort) and fatal signals: dump all thread stacks to a file.
    crash_file = open(f"{base}_crash.txt", "w", encoding="utf-8")
    _state["crash_file"] = crash_file
    faulthandler.enable(file=crash_file, all_threads=True)

    # (Re)point the root logger at the teed stderr so every logging call lands in the log too.
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    def _excepthook(exc_type, exc, tb):
        logging.critical("Uncaught exception:\n" + "".join(traceback.format_exception(exc_type, exc, tb)))
    sys.excepthook = _excepthook
    threading.excepthook = lambda a: logging.critical(
        f"Uncaught exception in thread {a.thread.name if a.thread else '?'}:\n"
        + "".join(traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback))
    )

    mon = None
    if monitor:
        interval = float(os.environ.get("PIPELINE_MONITOR_INTERVAL", "2"))
        mon = ResourceMonitor(f"{base}_resources.csv", interval)
        mon.start()
    _state["monitor"] = mon

    def _on_exit():
        logging.info(f"Process {os.getpid()} exiting normally.")
        if mon:
            mon.stop()
    atexit.register(_on_exit)

    logging.info(f"Logging to {base}.log" + (f" and {base}_resources.csv" if mon else ""))
    logging.info(f"Command: {' '.join(sys.argv)} (pid {os.getpid()}, python {sys.version.split()[0]})")
    return mon


def format_peaks(peaks):
    """One-line summary of the dict returned by ResourceMonitor.mark()."""
    if not peaks:
        return "no samples"
    return (f"peak: pipeline tree {peaks.get('tree_rss_gb', 0)} GB, ollama {peaks.get('ollama_rss_gb', 0)} GB, "
            f"system RAM used {peaks.get('ram_used_gb', 0)} GB, pagefile {peaks.get('pagefile_used_gb', 0)} GB, "
            f"VRAM {peaks.get('gpu_mem_used_mb', 0)} MB, cpu {peaks.get('cpu_pct', 0)}%")
