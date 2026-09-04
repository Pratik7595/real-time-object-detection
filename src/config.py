"""Config loading: YAML file -> validated dataclasses -> CLI overrides on top.

Kept in its own module because main.py, benchmark.py, evaluate.py and the tests
all need it, and because validation errors are much easier to write here than
scattered across the call sites.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = parent of src/. Everything relative in the YAML resolves against
# this, so the app behaves the same whatever directory it is launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


class ConfigError(ValueError):
    """Raised for a malformed or out-of-range config value."""


@dataclass
class ModelConfig:
    path: Path = REPO_ROOT / "models" / "yolox_tiny.onnx"
    input_size: tuple[int, int] = (416, 416)  # (height, width)


@dataclass
class DetectionConfig:
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    max_detections: int = 100
    classes: list[str] | None = None


@dataclass
class CameraConfig:
    source: str | int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1


@dataclass
class RuntimeConfig:
    device: str = "cpu"
    intra_op_threads: int = 0
    inter_op_threads: int = 1
    opencv_threads: int = 2
    preprocess_mode: str = "prealloc"
    infer_every: int = 1


@dataclass
class DisplayConfig:
    show: bool = True
    window_name: str = "Real-Time Object Detection"
    show_hud: bool = True
    font_scale: float = 0.5


@dataclass
class OutputConfig:
    results_dir: Path = REPO_ROOT / "results"
    write_csv: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> Config:
        d, r, m = self.detection, self.runtime, self.model

        if not 0.0 < d.conf_threshold < 1.0:
            raise ConfigError(
                f"detection.conf_threshold must be in (0, 1), got {d.conf_threshold}"
            )
        if not 0.0 < d.iou_threshold < 1.0:
            raise ConfigError(
                f"detection.iou_threshold must be in (0, 1), got {d.iou_threshold}"
            )
        if d.max_detections < 1:
            raise ConfigError("detection.max_detections must be >= 1")

        h, w = m.input_size
        # YOLOX's FPN has strides 8/16/32, so a non-multiple-of-32 input silently
        # produces a grid that does not line up with the decode and every box is
        # wrong. Fail loudly instead.
        if h % 32 or w % 32:
            raise ConfigError(
                f"model.input_size must be a multiple of 32 (strides are 8/16/32), "
                f"got {h}x{w}"
            )
        if h < 64 or w < 64:
            raise ConfigError(f"model.input_size is implausibly small: {h}x{w}")

        if r.device not in ("cpu", "cuda", "auto"):
            raise ConfigError(f"runtime.device must be cpu|cuda|auto, got {r.device!r}")
        if r.preprocess_mode not in ("prealloc", "naive"):
            raise ConfigError(
                f"runtime.preprocess_mode must be prealloc|naive, "
                f"got {r.preprocess_mode!r}"
            )
        if r.infer_every < 1:
            raise ConfigError("runtime.infer_every must be >= 1")
        if r.intra_op_threads < 0 or r.inter_op_threads < 0:
            raise ConfigError("thread counts must be >= 0 (0 means auto)")

        return self


def _resolve_path(value: Any) -> Path:
    """Relative paths hang off the repo root; absolute paths are left alone."""
    p = Path(str(value))
    return p if p.is_absolute() else (REPO_ROOT / p)


def _coerce_source(value: Any) -> str | int:
    """A webcam index arrives as "0" from argparse and as 0 from YAML."""
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def load_config(path: Path | str | None = None) -> Config:
    """Load config.yaml. A missing file is fine -- the dataclass defaults stand."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    cfg = Config()

    if not cfg_path.exists():
        if path is not None:
            raise ConfigError(f"Config file not found: {cfg_path}")
        return cfg.validate()

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path} must contain a YAML mapping at the top level")

    for section_name in (f.name for f in dataclasses.fields(cfg)):
        section_raw = raw.get(section_name)
        if section_raw is None:
            continue
        if not isinstance(section_raw, dict):
            raise ConfigError(f"config section {section_name!r} must be a mapping")

        section = getattr(cfg, section_name)
        known = {f.name for f in dataclasses.fields(section)}
        for key, value in section_raw.items():
            if key not in known:
                raise ConfigError(
                    f"Unknown config key {section_name}.{key}. "
                    f"Valid keys: {', '.join(sorted(known))}"
                )
            setattr(section, key, value)

    # Type fix-ups for the handful of fields YAML cannot express directly.
    cfg.model.path = _resolve_path(cfg.model.path)
    cfg.model.input_size = tuple(int(v) for v in cfg.model.input_size)  # type: ignore[assignment]
    cfg.output.results_dir = _resolve_path(cfg.output.results_dir)
    cfg.camera.source = _coerce_source(cfg.camera.source)

    return cfg.validate()


def apply_cli_overrides(cfg: Config, args: Any) -> Config:
    """Overlay argparse results. Only flags the user actually passed apply.

    argparse defaults are None everywhere for exactly this reason: we cannot tell
    "user typed --conf 0.3" from "argparse filled in 0.3" otherwise, and the YAML
    would be silently ignored.
    """
    overrides: list[tuple[Any, str, Any]] = [
        (cfg.model, "path", getattr(args, "model", None)),
        (cfg.detection, "conf_threshold", getattr(args, "conf", None)),
        (cfg.detection, "iou_threshold", getattr(args, "iou", None)),
        (cfg.detection, "classes", getattr(args, "classes", None)),
        (cfg.camera, "source", getattr(args, "source", None)),
        (cfg.camera, "width", getattr(args, "width", None)),
        (cfg.camera, "height", getattr(args, "height", None)),
        (cfg.runtime, "device", getattr(args, "device", None)),
        (cfg.runtime, "intra_op_threads", getattr(args, "threads", None)),
        (cfg.runtime, "infer_every", getattr(args, "infer_every", None)),
    ]
    for target, attr, value in overrides:
        if value is not None:
            setattr(target, attr, value)

    imgsz = getattr(args, "imgsz", None)
    if imgsz is not None:
        cfg.model.input_size = (int(imgsz), int(imgsz))

    if getattr(args, "no_display", False):
        cfg.display.show = False

    cfg.model.path = _resolve_path(cfg.model.path)
    cfg.camera.source = _coerce_source(cfg.camera.source)
    return cfg.validate()
