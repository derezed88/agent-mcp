"""
model_settings.py — Typed getters and setters for per-model token-selection parameters.

Getters return the live in-memory value from LLM_REGISTRY.
Setters update LLM_REGISTRY in-place AND persist to llm-models.json via
save_llm_model_field (same mechanism used by plugin-manager model commands).

These functions are the single authoritative path for reading and writing
temperature, top_p, top_k, and token_selection_setting.  Both routes.py
command handlers and plugin code should use these rather than touching
LLM_REGISTRY directly.

Supported parameter ranges:
  temperature : float  0.0–2.0   (both OPENAI and GEMINI)
  top_p       : float  0.0–1.0   (both OPENAI and GEMINI)
  top_k       : int    >= 1      (GEMINI native; also forwarded via model_kwargs for OPENAI-compatible
                                  backends such as llama.cpp that accept it as an extra field)
  token_selection_setting : str  "default" or "custom"

Plugin usage:
  from model_settings import (
      get_model_temperature, set_model_temperature,
      get_model_top_p, set_model_top_p,
      get_model_top_k, set_model_top_k,
      get_token_selection_setting, set_token_selection_setting,
  )
"""

from config import LLM_REGISTRY, save_llm_model_field, log

_VALID_TOKEN_SETTINGS = ("default", "custom")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_model(model_key: str) -> dict:
    """Return the registry entry or raise KeyError with a descriptive message."""
    cfg = LLM_REGISTRY.get(model_key)
    if cfg is None:
        available = ", ".join(sorted(LLM_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{model_key}'. Available: {available}")
    return cfg


def _require_gemini(model_key: str, cfg: dict, param: str):
    """No-op placeholder — top_k is now allowed for all model types."""
    pass


def _persist(model_key: str, field: str, value) -> None:
    """Write field to LLM_REGISTRY and to llm-models.json."""
    LLM_REGISTRY[model_key][field] = value
    if not save_llm_model_field(model_key, field, value):
        log.error(f"model_settings: save_llm_model_field failed for model='{model_key}' field='{field}'")


# ---------------------------------------------------------------------------
# temperature
# ---------------------------------------------------------------------------

def get_model_temperature(model_key: str) -> float:
    """Return the stored temperature for model_key."""
    cfg = _require_model(model_key)
    return float(cfg.get("temperature", 1.0))


def set_model_temperature(model_key: str, value: float) -> None:
    """Set temperature for model_key (range 0.0–2.0). Persists to llm-models.json."""
    _require_model(model_key)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"temperature must be a number, got {type(value).__name__}")
    if not (0.0 <= value <= 2.0):
        raise ValueError(f"temperature must be in [0.0, 2.0], got {value}")
    _persist(model_key, "temperature", value)


# ---------------------------------------------------------------------------
# top_p
# ---------------------------------------------------------------------------

def get_model_top_p(model_key: str) -> float:
    """Return the stored top_p for model_key."""
    cfg = _require_model(model_key)
    return float(cfg.get("top_p", 1.0))


def set_model_top_p(model_key: str, value: float) -> None:
    """Set top_p for model_key (range 0.0–1.0). Persists to llm-models.json."""
    _require_model(model_key)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"top_p must be a number, got {type(value).__name__}")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"top_p must be in [0.0, 1.0], got {value}")
    _persist(model_key, "top_p", value)


# ---------------------------------------------------------------------------
# top_k (GEMINI native; forwarded via model_kwargs for OPENAI-compatible backends)
# ---------------------------------------------------------------------------

def get_model_top_k(model_key: str) -> int | None:
    """Return the stored top_k for model_key, or None if unset."""
    cfg = _require_model(model_key)
    v = cfg.get("top_k")
    return int(v) if v is not None else None


def set_model_top_k(model_key: str, value: int) -> None:
    """Set top_k for model_key (integer >= 1, GEMINI only). Persists to llm-models.json."""
    cfg = _require_model(model_key)
    _require_gemini(model_key, cfg, "top_k")
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise TypeError(f"top_k must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"top_k must be >= 1, got {value}")
    _persist(model_key, "top_k", value)


# ---------------------------------------------------------------------------
# token_selection_setting
# ---------------------------------------------------------------------------

def get_token_selection_setting(model_key: str) -> str:
    """Return 'default' or 'custom' for model_key."""
    cfg = _require_model(model_key)
    return cfg.get("token_selection_setting", "default")


def set_token_selection_setting(model_key: str, value: str) -> None:
    """Set token_selection_setting for model_key ('default' or 'custom'). Persists."""
    _require_model(model_key)
    value = value.strip().lower()
    if value not in _VALID_TOKEN_SETTINGS:
        raise ValueError(
            f"token_selection_setting must be one of {_VALID_TOKEN_SETTINGS}, got '{value}'"
        )
    _persist(model_key, "token_selection_setting", value)
