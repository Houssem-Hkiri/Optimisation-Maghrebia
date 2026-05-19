from pathlib import Path

import numpy as np

from maghrebia_quant.allocation_10md import AdditionalAllocationConfig, run_allocation_analysis


def test_allocation_10md_budget_target_and_regulatory_controls():
    project_dir = Path(__file__).resolve().parents[1]
    result = run_allocation_analysis(project_dir, AdditionalAllocationConfig())

    weights = result["recommended_weights"]
    comparison = result["05_Resultats_Modeles"]
    recommended = result["recommended_model"]
    recommended_row = comparison.loc[comparison["Model"].eq(recommended)].iloc[0]
    regulatory = result["08_Contraintes_Reglementaires"]
    testable = regulatory.loc[regulatory["model"].eq(recommended) & regulatory["testable"].astype(bool)]

    assert np.isclose(weights.sum(), 1.0, atol=1e-8)
    assert (weights >= -1e-10).all()
    assert np.isclose(weights.sum() * result["config"].additional_budget, 10_000_000.0, atol=1e-4)
    assert np.isclose(result["V_TOTAL_CURRENT"], result["V_OPT_CURRENT"] + result["V_FIXED_CURRENT"], atol=1e-2)
    assert np.isclose(result["V_OPT_FINAL"], result["V_OPT_CURRENT"] + result["config"].additional_budget)
    assert np.isclose(result["V_TOTAL_FINAL"], result["V_TOTAL_CURRENT"] + result["config"].additional_budget)
    assert np.isfinite(result["R_REQUIRED_ADDITIONAL_FOR_OPT_TARGET"])
    assert np.isfinite(result["R_REQUIRED_ADDITIONAL_FOR_TOTAL_TARGET"])
    assert np.isclose(
        result["R_REQUIRED_ADDITIONAL_FOR_OPT_TARGET"],
        (result["TARGET_RETURN"] * result["V_OPT_FINAL"] - result["R_OPT_CURRENT"] * result["V_OPT_CURRENT"])
        / result["config"].additional_budget,
    )
    if result["target_total_feasible"]:
        assert recommended_row["R_total_final"] >= result["TARGET_RETURN"] - 1e-10
    else:
        assert result["max_return_additional"] < result["R_REQUIRED_ADDITIONAL_FOR_TOTAL_TARGET"]
        assert recommended_row["Target_Status"] == "TARGET_NOT_REACHED"
    assert recommended != "Max_Return_Constraints"
    assert not testable["Status"].eq("FAILED").any()
    assert len(result["09_Monte_Carlo"]) >= 15_000
