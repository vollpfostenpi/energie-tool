import numpy as np
import pandas as pd

from core import (
    try_parse_datetime, clean_numeric, orient_tilt_factors, _ROOF_FACTOR,
    distribute_monthly_to_index
)

def test_try_parse_datetime_formats():
    assert pd.notna(try_parse_datetime("2025-01-02 13:45"))
    assert pd.notna(try_parse_datetime("02.01.2025 13:45"))
    assert pd.notna(try_parse_datetime("2025-01-02"))
    assert pd.notna(try_parse_datetime("02.01.2025"))

def test_clean_numeric_de_locale():
    assert clean_numeric("1.234,56") == 1234.56
    assert clean_numeric(" 12,5 ") == 12.5
    assert np.isnan(clean_numeric("NaN"))

def test_orient_tilt_bounds():
    of, tf = orient_tilt_factors(0, 30)
    assert 0.70 <= of <= 1.0
    assert 0.85 <= tf <= 1.0
    of2, tf2 = orient_tilt_factors(180, 10)
    assert 0.70 <= of2 <= 1.0
    assert 0.85 <= tf2 <= 1.0

def test_roof_factor_keys():
    for k in ["Satteldach (Wohnbau)", "Walmdach (Wohnbau)", "Flachdach (mit Ost/West-Aufständerung)", "Sheddach/Halle (Industrie)"]:
        assert k in _ROOF_FACTOR

def test_distribute_monthly_to_index_sum():
    monthly = pd.Series([1000, 1000], index=["Jan", "Feb"])
    rng = pd.date_range("2025-01-01", "2025-03-01", freq="15min", inclusive="left")
    pv = distribute_monthly_to_index(monthly, rng)
    assert abs(pv.sum() - 2000) < 1e-6
