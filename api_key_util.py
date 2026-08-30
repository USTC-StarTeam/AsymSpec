"""Read optional LLM-judge credentials from ``conf.yaml``.

Override the path with ``ASYMSPEC_CONF``. ``PPTAGENT_CONF`` remains a
backward-compatible alias for internal experiment snapshots.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONF = _REPO_ROOT / "conf.yaml"
CONF_YAML = Path(os.environ.get(
    "ASYMSPEC_CONF", os.environ.get("PPTAGENT_CONF", DEFAULT_CONF)
))


def _load(section: str = "BASIC_MODEL") -> dict:
    import yaml

    if not CONF_YAML.is_file():
        raise FileNotFoundError(
            f"LLM config not found at {CONF_YAML}. Set PPTAGENT_CONF "
            f"to point at a different conf.yaml."
        )
    with open(CONF_YAML) as f:
        cfg = yaml.safe_load(f) or {}
    if section not in cfg:
        raise KeyError(
            f"Section {section!r} not in {CONF_YAML}; "
            f"available: {sorted(cfg)}"
        )
    return cfg[section] or {}


def get_api_key(section: str = "BASIC_MODEL") -> str:
    val = _load(section).get("api_key")
    if not isinstance(val, str) or not val.strip() or val.startswith("sk-xxx"):
        raise ValueError(
            f"{section}.api_key in {CONF_YAML} is missing/placeholder."
        )
    return val


def get_base_url(section: str = "BASIC_MODEL") -> str:
    val = _load(section).get("base_url")
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"{section}.base_url in {CONF_YAML} is missing.")
    return val


# --- Optional API-call logging ---------------------------------------------
# Set ASYMSPEC_API_LOGGER_PATH to a directory providing local_api_logger.py.
# Without that optional hook, the client remains functional and logging is a
# no-op. Never commit judge requests or responses that contain private data.

API_LOGS_DIR = os.environ.get(
    "ASYMSPEC_API_LOGS_DIR", str(_REPO_ROOT / "runs" / "api_logs"))
_LOCAL_API_LOGGER_PATH = os.environ.get("ASYMSPEC_API_LOGGER_PATH", "")


def _get_api_logger():
    import sys
    if _LOCAL_API_LOGGER_PATH and _LOCAL_API_LOGGER_PATH not in sys.path:
        sys.path.insert(0, _LOCAL_API_LOGGER_PATH)
    try:
        from local_api_logger import APILogger
        return APILogger(log_dir=API_LOGS_DIR)
    except ImportError:
        # The logger is an optional integration, not a release dependency.
        class _NoOpLogger:
            def log_call(self, **_kw):
                pass
        return _NoOpLogger()


def get_logged_openai_client(section: str = "BASIC_MODEL",
                             user: str = "asymspec"):
    """Return an OpenAI client, optionally wrapped by ``local_api_logger``.

    The endpoint and key are read from ``conf.yaml``. When the optional logger
    is unavailable, calls proceed without request/response logging.
    """
    import time
    from openai import OpenAI

    client = OpenAI(base_url=get_base_url(section),
                    api_key=get_api_key(section))
    logger = _get_api_logger()
    _orig_create = client.chat.completions.create

    def _logged_create(*args, **kwargs):
        t0 = time.perf_counter()
        model = kwargs.get("model", "unknown")
        try:
            resp = _orig_create(*args, **kwargs)
        except Exception as e:  # log the failed call too, then re-raise
            logger.log_call(
                model=model, request_data=kwargs, response_data={},
                user=user, duration_ms=(time.perf_counter() - t0) * 1e3,
                success=False, error=repr(e),
                metadata={"base_url": get_base_url(section)})
            raise
        try:
            rd = resp.model_dump()
        except Exception:
            rd = {"raw": str(resp)}
        logger.log_call(
            model=model, request_data=kwargs, response_data=rd, user=user,
            duration_ms=(time.perf_counter() - t0) * 1e3, success=True,
            metadata={"base_url": get_base_url(section)})
        return resp

    client.chat.completions.create = _logged_create
    return client
