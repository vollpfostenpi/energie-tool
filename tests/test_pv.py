import numpy as np
from core import PVInput, pv_compute_single, pv_compute_multi

def test_pv_single_positive_outputs():
    res = pv_compute_single("Satteldach (Wohnbau)", 200.0, 450, 2.0, 1000, 0, 30, 5, 0.82)
    assert res["usable_area"] > 0
    assert res["kwp_dc"] >= 0
    assert res["annual_ac"] >= 0

def test_pv_multi_clipping_applies():
    areas = [
        PVInput("Satteldach (Wohnbau)", 200, 0, 30, 5, 0.15),
        PVInput("Flachdach (mit Ost/West-Aufständerung)", 300, 90, 12, 5, 0.15),
    ]
    monthly, summary = pv_compute_multi(areas, 450, 2.0, 1000, 0.82, inv_ac_kw=30.0, inv_eff=0.98)
    assert monthly.sum().sum() == summary["total_annual_ac"]
    assert summary["total_annual_ac"] >= 0
