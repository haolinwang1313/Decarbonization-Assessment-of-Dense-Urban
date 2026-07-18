from __future__ import annotations

from pathlib import Path

from paper05.experiments.run_one import run_one
from paper05.model.checks import valid_result


def test_no_fake_valid_result() -> None:
    passed, reason = valid_result(
        solver_success=True,
        cap_violation_ton=0.1,
        load_shedding_mwh=0.0,
        tolerance=1e-6,
    )

    assert passed is False
    assert reason == "co2_cap_violation"


def test_failed_full8760_guard_is_not_marked_valid(tmp_path: Path) -> None:
    status = run_one(
        {
            "experiment_id": "E2_8760_validation",
            "run_id": "guard_failure",
            "temporal_mode": "full8760_opt",
        },
        output_root=tmp_path,
    )

    assert status["solver_status"] == "failed"
    assert status["is_valid_result"] is False
    assert "failed_full8760_opt" in status["invalid_reason"]
