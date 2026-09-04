"""Config loading, validation and CLI precedence."""

from __future__ import annotations

import argparse

import pytest
import yaml

from src.config import (
    REPO_ROOT,
    Config,
    ConfigError,
    apply_cli_overrides,
    load_config,
)


def write_config(tmp_path, data: dict):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_repo_config_is_valid():
    """The config the project actually ships with must load and validate."""
    cfg = load_config()
    assert cfg.model.input_size == (416, 416)
    assert 0 < cfg.detection.conf_threshold < 1


def test_shipped_default_is_the_int8_model():
    """Documents the decision: INT8 is the default because it measured 1.9x
    faster for 2.5 mAP points. If this is ever changed back, the performance
    and accuracy tables in the docs have to change with it."""
    assert load_config().model.path.name == "yolox_tiny_int8.onnx"


def test_missing_named_config_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_values_from_yaml_are_applied(tmp_path):
    path = write_config(
        tmp_path,
        {"detection": {"conf_threshold": 0.55}, "model": {"input_size": [320, 320]}},
    )
    cfg = load_config(path)
    assert cfg.detection.conf_threshold == 0.55
    assert cfg.model.input_size == (320, 320)


def test_unknown_key_is_rejected_rather_than_ignored(tmp_path):
    """A typo in config.yaml should fail loudly, not silently do nothing."""
    path = write_config(tmp_path, {"detection": {"confidence": 0.5}})
    with pytest.raises(ConfigError, match="Unknown config key"):
        load_config(path)


def test_relative_paths_resolve_against_the_repo_root(tmp_path):
    path = write_config(tmp_path, {"model": {"path": "models/custom.onnx"}})
    cfg = load_config(path)
    assert cfg.model.path.is_absolute()
    assert cfg.model.path == REPO_ROOT / "models" / "custom.onnx"


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("detection", "conf_threshold", 1.5, "conf_threshold"),
        ("detection", "conf_threshold", 0.0, "conf_threshold"),
        ("detection", "iou_threshold", -0.1, "iou_threshold"),
        ("detection", "max_detections", 0, "max_detections"),
        ("runtime", "device", "tpu", "device"),
        ("runtime", "preprocess_mode", "fancy", "preprocess_mode"),
        ("runtime", "infer_every", 0, "infer_every"),
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, section, key, value, match):
    path = write_config(tmp_path, {section: {key: value}})
    with pytest.raises(ConfigError, match=match):
        load_config(path)


@pytest.mark.parametrize("size", [[400, 400], [415, 416], [33, 33]])
def test_input_size_must_be_a_multiple_of_32(tmp_path, size):
    """YOLOX's strides are 8/16/32. A non-multiple silently misaligns the decode
    grid and every box comes out wrong, so it has to fail at load time."""
    path = write_config(tmp_path, {"model": {"input_size": size}})
    with pytest.raises(ConfigError, match="multiple of 32"):
        load_config(path)


def _args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        model=None, conf=None, iou=None, classes=None, source=None, width=None,
        height=None, device=None, threads=None, infer_every=None, imgsz=None,
        no_display=False,
    )
    return argparse.Namespace(**{**defaults, **kwargs})


def test_cli_overrides_beat_the_file():
    cfg = apply_cli_overrides(Config(), _args(conf=0.75, imgsz=320))
    assert cfg.detection.conf_threshold == 0.75
    assert cfg.model.input_size == (320, 320)


def test_unset_cli_flags_leave_config_values_alone():
    """argparse defaults are None precisely so an unset flag cannot silently
    overwrite what the YAML said."""
    cfg = Config()
    cfg.detection.conf_threshold = 0.42
    assert apply_cli_overrides(cfg, _args()).detection.conf_threshold == 0.42


def test_webcam_index_is_coerced_from_string():
    """argparse gives "0" but cv2.VideoCapture needs int 0 to mean a camera."""
    assert apply_cli_overrides(Config(), _args(source="0")).camera.source == 0


def test_file_source_stays_a_string():
    cfg = apply_cli_overrides(Config(), _args(source="assets/sample.jpg"))
    assert cfg.camera.source == "assets/sample.jpg"


def test_invalid_cli_override_is_rejected():
    with pytest.raises(ConfigError):
        apply_cli_overrides(Config(), _args(conf=2.0))
