"""
SwanLab experiment tracking — optional, failure-tolerant.

Every entry point here degrades to a no-op when SwanLab is not installed or is
switched off, so the training scripts run unchanged either way:

    SWANLAB_MODE=disabled   -> tracking off (default when no API key is set)
    SWANLAB_MODE=local      -> writes ./swanlog, never uploads
    SWANLAB_MODE=online     -> uploads to the SwanLab cloud (needs SWANLAB_API_KEY)

Mode resolution, in order of precedence:
  1. SWANLAB_MODE, if set explicitly
  2. online, if SWANLAB_API_KEY is present
  3. local, otherwise  — so a keyless run still leaves a local trace
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

from app.utils.config import get_config


def _import_swanlab():
    """Import swanlab lazily; return None when it is not installed."""
    try:
        import swanlab  # noqa: PLC0415
        return swanlab
    except ImportError:
        return None


def resolve_mode(api_key: Optional[str], explicit: Optional[str] = None) -> str:
    """Pick the SwanLab mode from env/config, falling back to something safe."""
    if explicit:
        return explicit
    if api_key:
        return "online"
    return "local"


def tracking_enabled() -> bool:
    """True only when a run would actually record something."""
    cfg = get_config()
    mode = resolve_mode(cfg.swanlab_api_key, cfg.swanlab_mode)
    return mode != "disabled" and _import_swanlab() is not None


def init_tracking(
    name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    tags: Optional[list] = None,
) -> Optional[Any]:
    """Start a SwanLab run and return it, or None when tracking is off.

    Never raises: a broken tracking setup must not take down a multi-hour
    training run.
    """
    swanlab = _import_swanlab()
    if swanlab is None:
        print("[swanlab] not installed — skipping (uv sync --extra tracking)")
        return None

    cfg = get_config()
    mode = resolve_mode(cfg.swanlab_api_key, cfg.swanlab_mode)
    if mode == "disabled":
        print("[swanlab] SWANLAB_MODE=disabled — skipping")
        return None

    try:
        run = swanlab.init(
            project=cfg.swanlab_project,
            workspace=cfg.swanlab_workspace,
            mode=mode,
            config=dict(config or {}),
            name=name or cfg.swanlab_experiment,
            tags=tags,
            log_dir=cfg.swanlab_logdir,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[swanlab] init failed ({exc}) — continuing without tracking")
        return None

    # `id` is the run's short public id. Note: `run.url` only exists in online
    # mode and raises in local mode, so don't touch it here.
    print(f"[swanlab] mode={mode} project={cfg.swanlab_project} "
          f"run={getattr(run, 'id', '?')} logdir={cfg.swanlab_logdir}")
    return run


def log_metrics(run: Optional[Any], data: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log a dict of scalars. No-op when `run` is None."""
    if run is None:
        return
    try:
        run.log(data, step=step)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[swanlab] log failed at step {step}: {exc}")


def finish_tracking(run: Optional[Any]) -> None:
    """Close the run. No-op when `run` is None."""
    if run is None:
        return
    try:
        run.finish()
    except Exception:  # pragma: no cover - defensive
        pass
