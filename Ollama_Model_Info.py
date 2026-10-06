import os
from functools import lru_cache

import requests


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


@lru_cache(maxsize=None)
def get_model_context_length(model: str) -> int:
    """Return a model's context length from Ollama metadata at startup."""
    response = requests.post(
        f"{OLLAMA_HOST}/api/show",
        json={"name": model},
        timeout=30,
    )
    response.raise_for_status()

    model_info = response.json().get("model_info", {})
    context_lengths = [
        value for key, value in model_info.items()
        if key.endswith(".context_length") and isinstance(value, int)
    ]
    if not context_lengths:
        raise RuntimeError(f"Ollama did not report a context length for model {model!r}.")

    return context_lengths[0]
