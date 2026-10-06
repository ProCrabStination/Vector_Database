"""
Preprocess a low-quality scanned PDF (skew, noise, faint/uneven text) before
feeding it into the Docling ingestion pipeline. Wraps ocrmypdf's image-cleanup
passes -- deskew, unpaper-based cleanup, upsampling -- without relying on
ocrmypdf's own OCR output (Docling still does the real OCR downstream).

Requires: pip install ocrmypdf (already installed in this repo's venv), a
Tesseract binary on PATH (ocrmypdf always needs *a* OCR engine present to run,
even though we discard its text layer and only want the image cleanup), and,
for --clean/--clean-final, unpaper on PATH (choco install unpaper, elevated
shell).

Note: --remove-background is NOT passed by default -- ocrmypdf 17.13.0 raises
NotImplementedError("--remove-background is temporarily not implemented"),
an upstream regression. Re-add it once a newer ocrmypdf release restores it.

Usage (from repo root, project venv):
    .venv\\Scripts\\python.exe Clean_PDF_For_OCR.py "path\\to\\bad_scan.pdf" "path\\to\\bad_scan_cleaned.pdf"
    .venv\\Scripts\\python.exe Clean_PDF_For_OCR.py "path\\to\\bad_scan.pdf" "path\\to\\bad_scan_cleaned.pdf" --clean

Then point Document_To_Database_Pipeline_Ollama.py at the cleaned output
instead of the original scan.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def clean_pdf(input_path, output_path, oversample=300, extra_clean=False):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "tesseract not found on PATH. ocrmypdf needs an OCR engine present "
            "to run its cleanup passes, even though we only use the image "
            "cleanup, not its text output."
        )

    args = [
        sys.executable, "-m", "ocrmypdf",
        "--deskew",
        # --remove-background is omitted: ocrmypdf 17.13.0 raises
        # NotImplementedError("--remove-background is temporarily not
        # implemented") -- an upstream regression, not a config issue here.
        # Re-add it once a newer ocrmypdf release restores support.
        "--oversample", str(oversample),
        "--force-ocr",  # rasterize + reprocess every page, even ones with an existing (bad) text layer
    ]
    if extra_clean:
        if shutil.which("unpaper") is None:
            raise RuntimeError(
                "--clean requires unpaper, which isn't installed. Run this in an "
                "elevated PowerShell first:\n    choco install unpaper"
            )
        args += ["--clean", "--clean-final"]

    args += [str(input_path), str(output_path)]

    print(f"Running: {' '.join(args)}")
    result = subprocess.run(args)
    if result.returncode != 0:
        raise RuntimeError(f"ocrmypdf exited with code {result.returncode}")

    print(f"Wrote cleaned PDF: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_pdf")
    parser.add_argument("output_pdf")
    parser.add_argument("--oversample", type=int, default=300, help="Target DPI for upsampling (default 300)")
    parser.add_argument(
        "--clean", action="store_true",
        help="Also run unpaper-based deep cleanup (requires: choco install unpaper, in an elevated shell)"
    )
    args = parser.parse_args()

    clean_pdf(args.input_pdf, args.output_pdf, oversample=args.oversample, extra_clean=args.clean)


if __name__ == "__main__":
    main()
