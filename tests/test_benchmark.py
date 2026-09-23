"""Benchmark argument validation and table rendering.

Pure string and argparse work -- no model, no weights, no camera, so this runs
on a clean clone. The sweeps themselves are not covered here: they need weights
and take minutes, which is what `python -m src.benchmark` is for.
"""

from __future__ import annotations

import argparse

import pytest

from src.benchmark import (
    BenchResult,
    _fp32_base,
    _non_negative_int,
    _positive_int,
    format_stage_table,
    format_summary_table,
    format_tables,
    main,
)
from src.metrics import STAGES

from pathlib import Path


def _result(label: str = "imgsz 416", infer_every: int = 1) -> BenchResult:
    """One filled-in row. Stage means are distinct so a mix-up is visible."""
    return BenchResult(
        label=label,
        model="yolox_tiny_int8.onnx",
        imgsz=416,
        capture="640x480",
        preprocess="prealloc",
        ort_threads=0,
        cv_threads=2,
        infer_every=infer_every,
        frames=300,
        fps_mean=46.2,
        fps_median=45.7,
        fps_p95=53.5,
        fps_p5=40.9,
        cpu_percent=388.0,
        rss_peak_mb=108.0,
        detections_mean=11.25,
        stage_ms={
            "capture": 0.02,
            "preprocess": 0.89,
            "inference": 19.09,
            "postprocess": 1.31,
            "render": 0.31,
        },
    )


# --- argparse types ---------------------------------------------------------


@pytest.mark.parametrize("text", ["1", "30", "300"])
def test_positive_int_accepts_counts_above_zero(text):
    assert _positive_int(text) == int(text)


@pytest.mark.parametrize("text", ["0", "-1", "-300"])
def test_positive_int_rejects_zero_and_below(text):
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int(text)


def test_non_negative_int_accepts_zero():
    # --warmup 0 is a choice (measure from frame 0), not a mistake.
    assert _non_negative_int("0") == 0


def test_non_negative_int_rejects_negatives():
    with pytest.raises(argparse.ArgumentTypeError):
        _non_negative_int("-1")


# --- the parser rejects before any work happens -----------------------------
#
# The point of these: --frames and --warmup never reach config.validate(), and
# --frames 0 used to run the warm-up, summarise an empty history and write a
# table of 0.0 FPS into results/ as though it had been measured.


@pytest.mark.parametrize(
    "argv",
    [
        ["--frames", "0"],
        ["--frames", "-5"],
        ["--warmup", "-1"],
    ],
)
def test_parser_rejects_impossible_frame_counts(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "must be" in capsys.readouterr().err


# --- _fp32_base -------------------------------------------------------------
#
# This was defined twice, and the second copy silently won. A test pins the
# behaviour so a future duplicate cannot pass unnoticed.


def test_fp32_base_walks_back_from_the_int8_model():
    got = _fp32_base(Path("models/yolox_tiny_int8.onnx"))
    assert got.name == "yolox_tiny.onnx"


def test_fp32_base_leaves_an_unquantised_model_alone():
    model = Path("models/yolox_tiny.onnx")
    assert _fp32_base(model) == model


# --- compute_ms -------------------------------------------------------------


def test_compute_ms_is_the_sum_of_the_stage_means():
    r = _result()
    assert r.compute_ms == pytest.approx(sum(r.stage_ms.values()))


def test_compute_ms_stays_out_of_the_csv_row():
    # flat() feeds csv.DictWriter; a stray column would break every committed
    # table's schema. asdict() ignores properties -- this pins that.
    assert "compute_ms" not in _result().flat()


def test_compute_ms_tolerates_a_missing_stage():
    r = _result()
    del r.stage_ms["render"]
    assert r.compute_ms == pytest.approx(0.02 + 0.89 + 19.09 + 1.31)


# --- table rendering --------------------------------------------------------


def test_summary_table_has_a_header_a_rule_and_one_row_per_result():
    lines = format_summary_table([_result("a"), _result("b")]).splitlines()
    assert lines[0].startswith("| Config |")
    assert set(lines[1]) <= {"|", "-"}
    assert len(lines) == 4


def test_stage_table_names_every_stage():
    header = format_stage_table([_result()]).splitlines()[0]
    for stage in STAGES:
        assert f"{stage} ms" in header


def test_stage_table_reports_the_compute_sum_in_its_last_column():
    # Hardcoded, not compute_ms: comparing the table against the property it is
    # rendering would pass however wrong that property became.
    row = format_stage_table([_result()]).splitlines()[2]
    assert row.rstrip().endswith("| 21.62 |")


def test_frame_skipping_blanks_the_percentile_columns():
    # With --infer-every > 1 the per-frame cost is bimodal, so median/p95/p5 of
    # per-frame FPS describe nothing real. Mean is throughput and stays valid.
    table = format_summary_table([_result(infer_every=3)])
    row = table.splitlines()[2]
    assert row.count("n/a") == 3
    assert "46.2" in row
    assert "n/a`:" in table  # the explanatory footnote


def test_no_footnote_when_every_frame_is_inferred():
    assert "n/a" not in format_summary_table([_result()])


def test_format_tables_joins_both_tables_in_print_order():
    combined = format_tables([_result()])
    summary = format_summary_table([_result()])
    stages = format_stage_table([_result()])
    assert combined == summary + "\n\n" + stages
