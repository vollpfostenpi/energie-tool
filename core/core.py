# -*- coding: utf-8 -*-
"""
core/core.py – Vollständige Version
Fix: Enthält pv_compute_multi und alle notwendigen Hilfsfunktionen.
"""

from __future__ import annotations
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# --- Hilfsfunktionen ---
def try_parse_datetime(s):
    if isinstance(s, pd.Timestamp): return s
    s_str = str(s).strip()
    if ";" in s_str and len(s_str) >= 15:
        try:
            parts = s_str.split(";")
            return pd.to_datetime(f"{parts[0]} {parts[1]}", format="%Y%m%d %H%M%S")
        except: pass
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y%m%d;%H%M%S"):
        try: return pd.to_datetime(s_str, format=fmt)
        except: pass
    return pd.to_datetime(s_str, errors="coerce", dayfirst=True)

def auto_read(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None: return pd.DataFrame()
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")): return pd.read_excel(uploaded_file)
    content = uploaded_file.read()
    try: text = content.decode("utf-8")
    except: text = content.decode("latin-1")
    lines = [l for l in text.splitlines() if not l.startswith("#") and l.strip()]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep=None, engine="python")

def clean_numeric(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x).replace(".", "").replace(",", ".").strip()
    try: return float(s)
    except: return np.nan

# --- PV & Sektorenkopplung ---
_ROOF_FACTOR = {"Satteldach (Wohnbau)": 0.85, "Flachdach (aufgeständert)": 0.70, "Flachdach (ost/west)": 0.90, "Trapezblech (Industrie)": 0.95, "Freifläche": 1.0}

@dataclass
class PVInput:
    roof_type: str
    area_m2: float
    azimuth: float
    tilt: float
    spec_yield: float
    efficiency: float = 0.18

def orient_tilt_factors(azimuth: float, tilt: float) -> Tuple[float, float]:
    rad_az = math.radians(azimuth)
    rad_tilt = math.radians(tilt)
    of = 0.7 + 0.3 * math.cos(rad_az)
    tf = 0.85 + 0.15 * math.sin(rad_tilt * 2) 
    return (float(of), float(tf))

def pv_compute_single(roof_type, area, module_w, mod_eff, spec_yield, az, tilt, margin, inv_eff):
    usable = area * (1 - margin/100.0)
    kwp = usable * (mod_eff/100.0 if mod_eff > 1 else mod_eff) * _ROOF_FACTOR.get(roof_type, 1.0)
    of, tf = orient_tilt_factors(az, tilt)
    return {"usable_area": usable, "kwp_dc": kwp, "annual_ac": kwp * spec_yield * of * tf * inv_eff}

def pv_compute_multi(areas: List[PVInput], module_w, margin, spec_yield, inv_eff, inv_ac_kw=1000.0):
    """Berechnet Ertrag für mehrere Dachflächen."""
    total_kwp = 0.0
    total_ac = 0.0
    for a in areas:
        res = pv_compute_single(a.roof_type, a.area_m2, module_w, a.efficiency, spec_yield, a.azimuth, a.tilt, margin, inv_eff)
        total_kwp += res["kwp_dc"]
        total_ac += res["annual_ac"]
    return None, {"total_kwp": total_kwp, "total_annual_ac": total_ac}

def distribute_monthly_to_index(total_val, index):
    """Hilfsfunktion für Zeitreihen-Verteilung."""
    return pd.Series(total_val / len(index), index=index)

def simulate_self_consumption(load_e, pv_e, battery_kwh=0, battery_kw=0):
    soc, e_self, e_export, e_import = 0.0, 0.0, 0.0, 0.0
    hours = (load_e.index[1] - load_e.index[0]).total_seconds()/3600.0 if len(load_e)>1 else 0.25
    for d, g in zip(load_e, pv_e):
        s = g - d
        if s > 0:
            ch = min(s, battery_kw*hours if battery_kw>0 else s, battery_kwh - soc)
            soc += ch * 0.95
            e_export += (s - ch)
            e_self += d
        else:
            dis = min(abs(s), battery_kw*hours if battery_kw>0 else abs(s), soc)
            soc -= dis
            e_import += (abs(s) - dis)
            e_self += (g + dis)
    return {"self_cons_kwh": e_self, "autarky_rate": (e_self/load_e.sum()*100) if load_e.sum()>0 else 0}

# --- Branchendaten ---
def get_thg_quote_params(): return {"PKW": 250.0, "LKW_leicht": 450.0, "LKW_schwer": 11000.0}
def get_co2_price_prognosis(): return {2024: 45.0, 2025: 55.0, 2026: 65.0, 2027: 85.0, 2028: 105.0, 2029: 120.0, 2030: 135.0}
def get_official_sources(): return {"THG-Quote": "https://www.now-gmbh.de/schwerpunkte/thg-quote/", "CO2-Abgabe": "https://www.umweltbundesamt.de/daten/klima/treibhausgas-emissionshandel", "Marktdaten": "https://www.smard.de/home"}
