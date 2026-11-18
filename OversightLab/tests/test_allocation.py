"""Exact-oracle, experimental-control, and offline-evaluation checks."""
from oversight_lab.allocation import (
    ARMS, LOW, HIGH, inspected_customers, make_state, oracle,
    proportional, score, sensitivity, specialist, valid,
)


def test_oracle_feasible_and_optimal_to_single_unit_moves():
    for seed in range(20):
        state = make_state(seed, 1)
        allocation = oracle(state)
        assert valid(state, allocation)[0]
        assert sum(allocation.values()) == state["available_stock"]
        reference = score(state, allocation)
        assert reference["quality"] == 1
        for old in allocation:
            for new in allocation:
                if old == new or not allocation[old] or allocation[new] >= state["demand"][new]:
                    continue
                alternative = allocation.copy()
                alternative[old] -= 1
                alternative[new] += 1
                assert score(state, alternative)["quality"] <= reference["quality"]


def test_specialist_has_real_conflict():
    rows = sensitivity(16)
    specialists = [r for r in rows if r["baseline"] == "inspector_specialist"]
    assert all(r["valid"] for r in rows)
    assert sum(r["common"] - r["quality"] for r in specialists) > 0
    assert sum(r["quality"] for r in specialists) < len(specialists)


def test_stratified_inspections_and_matched_instances():
    assert make_state(11, 3) == make_state(11, 3)
    for arm in ARMS[1:]:
        for rnd in range(1, 8):
            checked = inspected_customers(arm, 1, rnd)
            assert len(checked) == 2
            assert checked[0] in LOW and checked[1] in HIGH
            if arm.startswith("predictable"):
                assert checked == ("A", "D")


def test_invalid_allocation_and_proportional_baseline():
    state = make_state(0, 1)
    assert not valid(state, {"A": 5})[0]
    assert not valid(state, {c: 10000 for c in state["customers"]})[0]
    assert valid(state, proportional(state))[0]


def test_free_mock_run_and_resume(tmp_path):
    from oversight_lab.allocation import run, analyze
    out = tmp_path / "mock"
    rows = run("mock", out, rounds=2, seeds=1,
               arms=("control", "predictable_detailed"))
    assert len(rows) == 4
    assert all(r["valid"] and r["quality"] == 1 for r in rows)
    assert all(r["finalized"] for r in rows)
    assert len(run("mock", out, rounds=2, seeds=1,
                   arms=("control", "predictable_detailed"))) == 4
    summary = analyze(out)
    assert summary["api_calls"] == 0


def test_partial_observability_and_common_operational_signal():
    from oversight_lab.allocation import public_state, operational_feedback
    a = make_state(7, 1, "partial")
    b = make_state(7, 2, "partial")
    assert a["true_priority"] == b["true_priority"]
    assert a["priority"] != a["true_priority"]
    assert "true_priority" not in public_state(a)
    observed = operational_feedback(a, 7, 1)
    assert observed["status"] == "operational_outcome"
    assert "quality" not in observed and "true_priority" not in observed
    # The same signal is available to all oversight conditions.
    assert operational_feedback(a, 7, 1) == operational_feedback(a, 7, 1)


def test_automatic_feedback_and_blind_heldout(tmp_path):
    import json
    from oversight_lab.allocation import run
    rows = run("mock", tmp_path, rounds=3, seeds=1,
               arms=("control", "predictable_detailed"), observability="partial")
    assert len(rows) == 6
    assert sum(r["automatic_inspections"] for r in rows) == 3
    assert sum(bool(r["feedback_delivered"]) for r in rows) == 6
    assert all(r["operational_feedback"] is not None for r in rows)
    assert all(r["observed_inspection"] is None for r in rows if r["condition"] == "control")
    assert all(r["observed_inspection"] is not None for r in rows if r["condition"] != "control")
    assert sum("heldout_pre" in r for r in rows) == 2
    assert sum("heldout_post" in r for r in rows) == 2
    assert all(r["heldout_post"]["quality"] == 1 for r in rows if "heldout_post" in r)
    # Stored offline ground truth is separate from anything presented to the LLM.
    assert all("true_priority" not in json.dumps(r.get("observed_inspection")) for r in rows)


def test_heldout_trajectory_pairing_and_invalid_probe_reporting(tmp_path):
    import json
    import pandas as pd
    from oversight_lab.allocation import run, analyze

    run("mock", tmp_path, rounds=2, seeds=1,
        arms=("control", "predictable_coarse"), observability="partial")
    path = tmp_path / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # One failed *pre* probe must count as zero end-to-end, but not as a
    # conditionally valid quality gain.
    target = next(row for row in rows if row["condition"] == "control" and row["round"] == 1)
    target["heldout_pre"] = {"quality": 0.0, "valid": False, "allocation": None}
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = analyze(tmp_path)
    pairs = pd.read_csv(tmp_path / "evaluation" / "heldout_pre_post.csv")
    assert len(pairs) == 2
    assert pairs.pair_complete.all()
    control = pairs.loc[pairs.condition == "control"].iloc[0]
    assert control.heldout_gain_end_to_end == 1.0
    assert pd.isna(control.heldout_gain_both_valid)
    assert report["heldout_complete_pairs"] == 2
    assert report["heldout_pre_valid_rate"] == 0.5
    assert report["heldout_post_valid_rate"] == 1.0
