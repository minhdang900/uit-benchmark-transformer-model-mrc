"""Chấm stress-test v2 ở τ dev (scripts/score_stress2_tuned.py) và split stress2 của score_windows."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import score_stress2_tuned  # noqa: E402
import score_windows  # noqa: E402
from mrc.stress_v2 import StressItem, to_squad  # noqa: E402

# Một cặp E3b: câu gốc có đáp án "B"; câu biến đổi đã xoá câu chứa đáp án nên không
# có đáp án. Mô hình (giả) trả lời "B" ở cả hai, với delta = −2 ở mọi cửa sổ.
ORIG = StressItem(qid="p1-orig", question="A là gì?", context="A là B.", title="t",
                  answers=["B"], answer_starts=[5], category="E3", subset="E3b",
                  role="original", source_qid="s1", pair_id="p1")
PERT = StressItem(qid="p1", question="A là gì?", context="A là cái khác.", title="t",
                  answers=[], answer_starts=[], category="E3", subset="E3b",
                  role="perturbed", source_qid="s1", pair_id="p1", note="câu đã xoá: A là B.")
WINDOWS = {"p1-orig": [{"delta": -2.0, "score": 5.0, "answer": "B"}],
           "p1": [{"delta": -2.0, "score": 4.0, "answer": "B"}]}


def _setup(tmp_path: Path, tau: float, run: str = "m") -> Path:
    (tmp_path / "stress.json").write_text(json.dumps(to_squad([ORIG, PERT])), encoding="utf-8")
    (tmp_path / f"windows_{run}_stress2.json").write_text(
        json.dumps({"checkpoint": "models/m", "commit": "abc", "records": WINDOWS}))
    (tmp_path / "thresholds.json").write_text(json.dumps({"systems": {run: {"tau": tau}}}))
    return tmp_path


def _run(tmp_path: Path, *runs: str) -> None:
    score_stress2_tuned.main(["--runs", *runs, "--results", str(tmp_path),
                              "--stress", str(tmp_path / "stress.json")])


def _out(tmp_path: Path, run: str = "m") -> dict:
    return json.loads((tmp_path / f"eval_{run}_stress2_tuned.json").read_text(encoding="utf-8"))


def test_at_tau_zero_answering_the_deleted_sentence_counts_as_broken(tmp_path):
    # delta + τ = −2 < 0 → trả lời ở cả hai: đúng câu gốc, sai câu đã xoá đáp án.
    _run(_setup(tmp_path, tau=0.0), "m")
    pairs = _out(tmp_path)["by_subset"]["E3b"]["pairs"]
    assert (pairs["both_correct"], pairs["broken"], pairs["fixed"]) == (0, 1, 0)
    assert pairs["broken_pct_of_original_correct"] == 100.0


def test_tau_is_read_from_thresholds_and_changes_the_decision(tmp_path):
    # τ = 5 → delta + τ = 3 ≥ 0 → từ chối cả hai: sai câu gốc, đúng câu biến đổi.
    _run(_setup(tmp_path, tau=5.0), "m")
    out = _out(tmp_path)
    assert out["tau"] == 5.0
    assert out["abstain_rate"] == 100.0
    pairs = out["by_subset"]["E3b"]["pairs"]
    assert (pairs["both_correct"], pairs["broken"], pairs["fixed"]) == (0, 0, 1)
    # Không cặp nào đúng câu gốc → tỉ lệ hỏng không xác định, không phải 0.
    assert pairs["broken_pct_of_original_correct"] is None


def test_output_has_the_same_blocks_as_direct_eval(tmp_path):
    _run(_setup(tmp_path, tau=0.0), "m")
    out = _out(tmp_path)
    for key in ("overall", "answerable_only", "impossible_only", "by_category", "by_subset"):
        assert key in out
    # "overall" chấm MỌI mục, kể cả câu gốc đi cặp — như run_eval.py.
    assert out["overall"]["count"] == 2
    assert out["answerable_only"]["count"] == 1 and out["impossible_only"]["count"] == 1
    assert out["tau_source"].startswith("results/thresholds.json")


def test_missing_windows_or_tau_is_skipped_without_writing(tmp_path):
    _setup(tmp_path, tau=0.0, run="m")
    _run(tmp_path, "no_windows")          # không có windows_no_windows_stress2.json
    assert not (tmp_path / "eval_no_windows_stress2_tuned.json").exists()
    (tmp_path / "windows_no_tau_stress2.json").write_text(json.dumps({"records": WINDOWS}))
    _run(tmp_path, "no_tau")              # có windows nhưng không có τ
    assert not (tmp_path / "eval_no_tau_stress2_tuned.json").exists()


def test_score_windows_stress2_split_loads_the_committed_set():
    audit = json.loads((Path(__file__).resolve().parents[1] / "results/stress_v2_audit.json")
                       .read_text(encoding="utf-8"))
    examples = score_windows.load_split("stress2", "data/raw", 42, 0.05)
    # stress2 phải chấm mọi mục, kể cả câu gốc đi cặp (để đếm "broken").
    assert len(examples) == audit["n_items"]
    assert len({e.qid for e in examples}) == len(examples)


def test_score_windows_rejects_unknown_split():
    with pytest.raises(SystemExit):
        score_windows.main(["--checkpoint", "x", "--name", "x", "--split", "stress"])
