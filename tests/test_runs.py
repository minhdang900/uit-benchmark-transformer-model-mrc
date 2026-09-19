"""Quy tắc chọn lần chạy PhoBERT chính (spec: 'stable AND better' trên DEV)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _runs import primary_phobert  # noqa: E402


def _run(res: Path, name: str, dev_f1: float, stable=None, curve_f1=None):
    (res / f"predictions_{name}_validation.json").write_text("{}")
    (res / f"windows_{name}_dev.json").write_text(json.dumps({"at_tau0": {"F1": dev_f1}}))
    curve = {"curve": [{"val_f1": curve_f1 if curve_f1 is not None else dev_f1}]}
    if stable is not None:
        curve["stability"] = {"stable": stable}
    (res / f"training_curve_{name}.json").write_text(json.dumps(curve))


def test_current_primary_is_best_full_dev_f1_among_original_runs(tmp_path):
    _run(tmp_path, "phobert", 61.16)
    _run(tmp_path, "phobert_lr2e5", 57.74)
    assert primary_phobert(tmp_path) == "phobert"


def test_stable_run_replaces_only_if_better_on_dev(tmp_path):
    _run(tmp_path, "phobert", 61.16)
    _run(tmp_path, "phobert_stable", 62.0, stable=True)
    assert primary_phobert(tmp_path) == "phobert_stable"


def test_stable_but_worse_run_does_not_replace(tmp_path):
    _run(tmp_path, "phobert", 61.16)
    _run(tmp_path, "phobert_stable", 60.0, stable=True)
    assert primary_phobert(tmp_path) == "phobert"


def test_better_but_unstable_run_does_not_replace(tmp_path):
    _run(tmp_path, "phobert", 61.16)
    _run(tmp_path, "phobert_stable", 65.0, stable=False)
    assert primary_phobert(tmp_path) == "phobert"


def test_candidate_without_validation_predictions_is_ignored(tmp_path):
    _run(tmp_path, "phobert", 61.16)
    _run(tmp_path, "phobert_stable", 65.0, stable=True)
    (tmp_path / "predictions_phobert_stable_validation.json").unlink()
    assert primary_phobert(tmp_path) == "phobert"


def test_falls_back_to_training_curve_when_full_dev_missing(tmp_path):
    _run(tmp_path, "phobert", 61.16, curve_f1=60.61)
    _run(tmp_path, "phobert_lr2e5", 57.74, curve_f1=58.36)
    (tmp_path / "windows_phobert_dev.json").unlink()
    (tmp_path / "windows_phobert_lr2e5_dev.json").unlink()
    assert primary_phobert(tmp_path) == "phobert"
