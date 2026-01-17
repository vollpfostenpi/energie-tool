import streamlit as st
import json
from pathlib import Path
import pandas as pd
import numpy as np
import zlib
import hashlib

# ================================================================
# 0) PAGE CONFIG (MUSS ALS ERSTES STREAMLIT-KOMMANDO KOMMEN)
# ================================================================
st.set_page_config(page_title="Profi-Planung OS", layout="wide", page_icon="🏗️")

# ================================================================
# 1) KONSTANTEN / OPTIONS
# ================================================================
SCOPE_OPTIONS = ["PV-System", "Speicher", "Ladeinfrastruktur"]
HAK_OPTIONS = [35, 50, 63, 80, 100, 125, 160, 250]
DACH_FORMS = ["Satteldach", "Flachdach", "Pultdach", "Walmdach"]
BAT_TYPES = ["Hochvolt", "Niedervolt"]
PV_KONZEPTE = ["Überschusseinspeisung", "Volleinspeisung"]
WB_POWER_OPTIONS = [11, 22, 50, 150]
FZ_KLASSEN = ["PKW", "Transporter", "LKW"]

REGIONS = {"Nord": 940, "Mitte": 1060, "Süd": 1180}  # kWh/kWp/a (heuristisch)

OBJECTIVES = [
    "Eigenverbrauch/Autarkie",
    "Depotcharging/Lastmanagement",
    "Arbitrage (Stromhandel)",
    "Kombi (Eigenverbrauch + Arbitrage)",
]

DEFAULTS = {
    "kunde": {"name": "—"},
    "scope": ["PV-System", "Speicher", "Ladeinfrastruktur"],
    "tech": {"hak_ampere": 63, "ems_required": False, "berechnungs_modus": "Detail"},
    # Altbestand (für Rückwärtskompatibilität)
    "dach": {"form": "Satteldach", "breite": 10.0, "tiefe": 6.0, "flaeche": 60.0, "neigung": 35, "azimut": 0},
    # Neu: mehrere Dachflächen
    "daecher": [
        {
            "name": "Dachfläche 1",
            "form": "Satteldach",
            "breite": 10.0,
            "tiefe": 6.0,
            "flaeche": 60.0,
            "flaeche_auto": True,
            "neigung": 35,
            "azimut": 0,
            "nutzfaktor": 0.80,
            "hinweis": "",
        }
    ],
    "pv": {
        "felder": [],  # Liste von Modulfeldern
        "wr": [],      # Liste WR
        "konzept": "Überschusseinspeisung",
        "total_kwp": 0.0,
        "total_ac_kw": 0.0,
    },
    "speicher": {
        "kap": 0.0,          # kWh
        "p": 0.0,            # kW
        "hersteller": "",
        "db_file": None,
        "spannung": "Hochvolt",
        "objective": "Eigenverbrauch/Autarkie",
    },
    "mobilität": {"ladepunkte": [], "fuhrpark": [], "total_ev_kwh": 0.0},
    "wirtschaft": {
        "strompreis": 0.35,
        "dieselpreis": 1.70,
        "einspeise_v": 0.08,
        "lastgang_jahr": 5000,
        # ROI/Annahmen
        "capex_pv_eur_per_kwp": 1100.0,
        "capex_bat_eur_per_kwh": 450.0,
        "capex_bat_eur_per_kw": 200.0,
        "opex_pv_pct": 0.015,  # 1.5%/a
        "opex_bat_pct": 0.01,  # 1%/a
        "discount_rate": 0.06,
        # Arbitrage Annahmen
        "arb_enabled": False,
        "arb_low_price": 0.18,
        "arb_high_price": 0.32,
        "arb_cycles_per_year": 180,
        "arb_roundtrip_eff": 0.90,
        "arb_dod": 0.90,
        "arb_degradation_eur_per_kwh": 0.03,
    },
}

# ================================================================
# 2) HILFSFUNKTIONEN (STATE / SAVE / SANITIZE)
# ================================================================
def deep_merge(target: dict, defaults: dict) -> None:
    """Fügt fehlende Keys rekursiv hinzu, überschreibt nichts."""
    for k, v in defaults.items():
        if k not in target:
            target[k] = v
        else:
            if isinstance(v, dict) and isinstance(target.get(k), dict):
                deep_merge(target[k], v)

def sanitize(obj):
    """Sichert JSON-Kompatibilität (keine NaN/Inf, keine numpy types)."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return 0.0
        return x
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return obj
    return obj

def dict_hash(obj: dict) -> str:
    s = json.dumps(sanitize(obj), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(s).hexdigest()

def stable_seed(*parts) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF

def safe_index(options, value, fallback=0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return fallback

def hak_p_max_kw(hak_ampere: int) -> float:
    return (hak_ampere * 400 * 1.73) / 1000  # 3~ 400V

def sum_wr_ac_kw(wr_list: list[dict]) -> float:
    total = 0.0
    for w in wr_list:
        try:
            total += float(w.get("p", 0.0))
        except Exception:
            pass
    return total

def roof_area(roof: dict) -> float:
    b = float(roof.get("breite", 0.0))
    t = float(roof.get("tiefe", 0.0))
    return max(b * t, 0.0)

def pv_potential_from_roofs(daecher: list, module_area_m2: float = 2.0):
    total_area = 0.0
    total_effective = 0.0
    for r in daecher:
        a = float(r.get("flaeche", 0.0))
        nf = float(r.get("nutzfaktor", 0.8))
        total_area += a
        total_effective += a * nf
    modules_max = int(total_effective / max(module_area_m2, 0.1))
    return total_area, total_effective, modules_max

def list_of_dicts_to_df(lst: list[dict], cols: list[str]) -> pd.DataFrame:
    if not lst:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(lst)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

def df_to_list_of_dicts(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    # NaNs entfernen/ersetzen
    clean = df.copy()
    clean = clean.replace({np.nan: ""})
    return clean.to_dict(orient="records")

def load_project(slug: str) -> tuple[dict, Path, Path]:
    p_path = Path("projects") / slug
    state_file = p_path / "state.json"
    doc_path = p_path / "documents"
    doc_path.mkdir(parents=True, exist_ok=True)

    if not state_file.exists():
        st.error("⚠️ Projektdatei nicht gefunden (state.json). Bitte Projekt über Dashboard anlegen/öffnen.")
        st.stop()

    with state_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    deep_merge(data, DEFAULTS)

    # Migration: wenn keine daecher-Liste existiert, aus altem d['dach'] erstellen
    if "daecher" not in data or not isinstance(data["daecher"], list) or len(data["daecher"]) == 0:
        old = data.get("dach", {})
        data["daecher"] = [{
            "name": "Dachfläche 1",
            "form": old.get("form", "Satteldach"),
            "breite": float(old.get("breite", 10.0)),
            "tiefe": float(old.get("tiefe", 6.0)),
            "flaeche": float(old.get("flaeche", 60.0)),
            "flaeche_auto": True,
            "neigung": int(old.get("neigung", 35)),
            "azimut": int(old.get("azimut", 0)),
            "nutzfaktor": 0.80,
            "hinweis": "",
        }]

    return data, state_file, doc_path

def save_project(state_file: Path, data: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(sanitize(data), f, indent=4, ensure_ascii=False)

def save_if_changed(state_file: Path, data: dict) -> None:
    h = dict_hash(data)
    if st.session_state.get("_last_saved_hash") != h:
        save_project(state_file, data)
        st.session_state["_last_saved_hash"] = h

# ================================================================
# 3) "KI" SPEICHER-EMPFEHLUNG
# ================================================================
def recommend_storage(d: dict, hak_kw: float, objective: str, region: str) -> dict:
    pv_kwp = float(d["pv"].get("total_kwp", 0.0))
    pv_ac  = float(d["pv"].get("total_ac_kw", 0.0))
    load_building = float(d["wirtschaft"].get("lastgang_jahr", 0.0))
    ev_kwh = float(d["mobilität"].get("total_ev_kwh", 0.0))
    load_total = max(load_building + ev_kwh, 0.0)
    day_load = load_total / 365.0 if load_total > 0 else 0.0

    wb_sum = sum(float(lp.get("p", 0.0)) for lp in d["mobilität"].get("ladepunkte", []))

    # Heuristiken nach Ziel
    # Energie (kWh)
    ev_factor = 1.0 + min(ev_kwh / max(load_building, 1.0), 1.0) * 0.25  # +0–25%
    base_kwh = pv_kwp * 1.0 * ev_factor if pv_kwp > 0 else max(day_load * 0.6, 10.0)

    if objective == "Depotcharging/Lastmanagement":
        # mehr Energie für Ladefenster + Lastglättung
        base_kwh *= 1.15
    elif objective == "Arbitrage (Stromhandel)":
        # arbitrage eher "Energiepakete" + höherer Durchsatz
        base_kwh *= 1.30
    elif objective == "Kombi (Eigenverbrauch + Arbitrage)":
        base_kwh *= 1.20

    # Grenzen über Tagesverbrauch
    min_kwh = day_load * 0.4
    max_kwh = day_load * 2.0 if day_load > 0 else base_kwh * 2.0
    rec_kwh = max(base_kwh, min_kwh)
    rec_kwh = min(rec_kwh, max_kwh) if max_kwh > 0 else rec_kwh

    # Leistung (kW) / C-Rate
    if objective == "Eigenverbrauch/Autarkie":
        target_c = 0.35 if wb_sum <= 11 else 0.45
    elif objective == "Depotcharging/Lastmanagement":
        target_c = 0.55
    elif objective == "Arbitrage (Stromhandel)":
        target_c = 0.75
    else:
        target_c = 0.55

    rec_kw = rec_kwh * target_c

    # Anschluss-Headroom (konservativ)
    reserve = 0.85
    headroom = max(hak_kw * reserve - (pv_ac + wb_sum), 0.0)
    reasons = []
    if headroom > 0:
        if rec_kw > headroom:
            reasons.append(f"Leistung an HAK-Headroom angepasst (Headroom ~ {headroom:.1f} kW).")
        rec_kw = min(rec_kw, headroom)
    else:
        reasons.append("Kein Headroom (PV-AC + WB >= HAK) → Leistung konservativ (EMS/LM Pflicht).")
        rec_kw = min(rec_kw, max(hak_kw * 0.10, 5.0))

    c_rate = (rec_kw / rec_kwh) if rec_kwh > 0 else 0.0
    hv_lv = "Hochvolt" if rec_kwh >= 15 else "Niedervolt"

    ems_needed = False
    if (pv_ac + wb_sum) > hak_kw:
        ems_needed = True
        reasons.append("PV-AC + Ladeleistung über HAK → Lastmanagement/EMS erforderlich.")
    if wb_sum >= 22:
        ems_needed = True
        reasons.append("Mehrere/leistungsstarke Ladepunkte → dynamisches Lastmanagement sinnvoll.")
    if pv_kwp >= 30:
        reasons.append("Ab ~30 kWp lohnt Monitoring/Regelung (Eigenverbrauch, Einspeisebegrenzung, Netzvorgaben).")

    hints = []
    if pv_kwp > 0 and day_load > 0:
        hints.append("Ziel: Eigenverbrauch erhöhen, Abend-/Nachtverbrauch abdecken, PV-Spitzen glätten.")
    if ev_kwh > 0:
        hints.append("Mit EV-Anteil: Ladefenster + PV-Mittagsspitzen durch Speicher/EMS besser nutzbar.")
    if "Arbitrage" in objective:
        hints.append("Arbitrage: Wirtschaftlichkeit hängt stark von Spread, Zyklen, Effizienz und Degradation ab.")

    return {
        "rec_kwh": round(float(rec_kwh), 1),
        "rec_kw": round(float(rec_kw), 1),
        "c_rate": round(float(c_rate), 2),
        "empf_typ": hv_lv,
        "ems_empfohlen": ems_needed,
        "gruende": reasons,
        "hinweise": hints,
        "inputs": {
            "objective": objective,
            "pv_kwp": round(pv_kwp, 1),
            "pv_ac_kw": round(pv_ac, 1),
            "wb_kw_sum": round(wb_sum, 1),
            "load_building_kwh_a": round(load_building, 0),
            "ev_kwh_a": round(ev_kwh, 0),
            "hak_kw": round(hak_kw, 1),
            "region": region,
        },
    }

# ================================================================
# 4) HOURLY SIMULATION (PV/LOAD/EV + BAT MIT LEISTUNGSLIMIT)
# ================================================================
@st.cache_data(show_spinner=False)
def simulate_hourly(
    pv_kwp: float,
    load_building_kwh_year: float,
    ev_kwh_year: float,
    bat_kwh: float,
    bat_kw: float,
    region: str,
    seed: int,
    roundtrip_eff: float = 0.92,
) -> pd.DataFrame:
    """
    8760h Simulation (vereinfachtes Profil, deterministisch).
    - PV: saisonal + tagesgang + wolkenrauschen, skaliert auf pv_kwp*spec_yield.
    - Gebäude-Last: saisonal + tagesprofil + rauschen, skaliert auf load_building_kwh_year.
    - EV-Last: abends stärker, skaliert auf ev_kwh_year.
    - Batterie: sequentiell mit kW-Limit (charge/discharge), SOC in kWh.
    Return: DF mit PV, Load, Direct, Charge, Discharge, Import, Export, SoC, Served
    """
    rng = np.random.default_rng(seed)
    hours = 8760
    t = np.arange(hours)
    day = t / 24.0
    year_frac = day / 365.0

    spec_yield = REGIONS.get(region, 1000)  # kWh/kWp/a
    pv_year = max(pv_kwp, 0.0) * spec_yield

    # PV shape: seasonal * daily bell
    seasonal = (np.sin(np.pi * (year_frac - 0.08)) ** 2 + 0.08)  # >0
    hour_in_day = t % 24
    daily = np.maximum(np.sin(np.pi * (hour_in_day - 6) / 12), 0) ** 1.6  # peaks midday
    clouds = np.clip(rng.normal(1.0, 0.18, hours), 0.3, 1.7)
    pv_raw = seasonal * daily * clouds
    pv_raw_sum = pv_raw.sum()
    pv = (pv_raw / pv_raw_sum) * pv_year if pv_raw_sum > 0 else np.zeros(hours)

    # Building load: seasonal + daily shape
    seasonal_l = (np.cos(2 * np.pi * year_frac) * 0.12 + 1.0)  # winter slightly higher
    daily_l = (0.65 + 0.25 * np.sin(2 * np.pi * (hour_in_day - 7) / 24) + 0.12 * np.sin(4 * np.pi * (hour_in_day - 7) / 24))
    daily_l = np.clip(daily_l, 0.35, 1.35)
    noise_l = np.clip(rng.normal(1.0, 0.07, hours), 0.75, 1.25)
    load_raw = seasonal_l * daily_l * noise_l
    load_raw_sum = load_raw.sum()
    load_building = (load_raw / load_raw_sum) * max(load_building_kwh_year, 0.0) if load_raw_sum > 0 else np.zeros(hours)

    # EV load: evening peak (17-23), some midday charging
    ev = np.zeros(hours)
    if ev_kwh_year > 0:
        evening = ((hour_in_day >= 17) & (hour_in_day <= 23)).astype(float)
        midday = ((hour_in_day >= 11) & (hour_in_day <= 15)).astype(float) * 0.4
        ev_shape = (evening + midday)
        ev_noise = np.clip(rng.normal(1.0, 0.35, hours), 0.2, 2.0)
        ev_raw = ev_shape * ev_noise
        s = ev_raw.sum()
        ev = (ev_raw / s) * ev_kwh_year if s > 0 else np.zeros(hours)

    load = load_building + ev

    # Battery simulation
    cap = max(bat_kwh, 0.0)
    pmax = max(bat_kw, 0.0)
    eff = np.clip(roundtrip_eff, 0.5, 0.99)
    eff_c = np.sqrt(eff)
    eff_d = np.sqrt(eff)

    soc = 0.0
    soc_series = np.zeros(hours)
    direct = np.zeros(hours)
    charge = np.zeros(hours)
    discharge = np.zeros(hours)
    imp = np.zeros(hours)
    exp = np.zeros(hours)
    served = np.zeros(hours)

    for i in range(hours):
        p = pv[i]
        l = load[i]

        # direct PV -> load
        d = min(p, l)
        direct[i] = d

        surplus = p - d
        deficit = l - d

        # charge from surplus, limited by pmax (kWh per hour) and cap
        if cap > 0 and pmax > 0 and surplus > 0:
            ch_possible = min(surplus, pmax)              # kWh this hour
            ch_cap = max(cap - soc, 0.0)                  # remaining space
            ch = min(ch_possible, ch_cap)
            # account charging efficiency: energy into battery = ch*eff_c, but from PV we consume ch
            soc += ch * eff_c
            charge[i] = ch
            surplus -= ch

        # discharge to cover deficit, limited by pmax and soc
        if cap > 0 and pmax > 0 and deficit > 0 and soc > 0:
            dis_possible = min(deficit, pmax)
            # energy available from SOC considering discharge eff: deliver = soc*eff_d?
            # Here: removing x from SOC yields x*eff_d delivered. So to deliver dis, we need dis/eff_d from SOC.
            need_from_soc = dis_possible / eff_d
            take = min(need_from_soc, soc)
            delivered = take * eff_d
            soc -= take
            discharge[i] = delivered
            deficit -= delivered

        # grid interactions
        exp[i] = max(surplus, 0.0)
        imp[i] = max(deficit, 0.0)

        served[i] = d + discharge[i]
        soc_series[i] = soc

    return pd.DataFrame({
        "PV": pv,
        "Last": load,
        "Direct": direct,
        "Charge": charge,
        "Discharge": discharge,
        "Import": imp,
        "Export": exp,
        "SoC": soc_series,
        "Served": served,
        "EV": ev,
        "Building": load_building,
    })

# ================================================================
# 5) PROJEKT-STATE LADEN
# ================================================================
if "active_slug" not in st.session_state:
    st.error("⚠️ Kein aktives Projekt. Bitte über das Dashboard starten.")
    st.stop()

P_SLUG = st.session_state["active_slug"]

if "project_data" not in st.session_state or st.session_state.get("_loaded_slug") != P_SLUG:
    d, STATE_FILE, DOC_PATH = load_project(P_SLUG)
    st.session_state["project_data"] = d
    st.session_state["_loaded_slug"] = P_SLUG
    st.session_state["_last_saved_hash"] = dict_hash(d)
else:
    d = st.session_state["project_data"]
    STATE_FILE = Path("projects") / P_SLUG / "state.json"
    DOC_PATH = Path("projects") / P_SLUG / "documents"
    DOC_PATH.mkdir(parents=True, exist_ok=True)

# ================================================================
# 6) SIDEBAR: SCOPE + HAK + SAVE
# ================================================================
with st.sidebar:
    st.header("🎯 Projekt-Sektoren")
    current_scope = [x for x in d.get("scope", []) if x in SCOPE_OPTIONS] or DEFAULTS["scope"]
    d["scope"] = st.multiselect("Themen zur Planung:", SCOPE_OPTIONS, default=current_scope)

    st.divider()
    st.header("⚡ Gebäudeanschluss")
    hak_val = d["tech"].get("hak_ampere", 63)
    d["tech"]["hak_ampere"] = st.selectbox("HAK Größe (Ampere)", HAK_OPTIONS, index=safe_index(HAK_OPTIONS, hak_val, 2))
    hak_kw = hak_p_max_kw(int(d["tech"]["hak_ampere"]))
    st.metric("Max. Belastbarkeit (theoret.)", f"{hak_kw:.1f} kW")

    st.divider()
    if st.button("💾 Projektstand speichern", use_container_width=True):
        save_project(STATE_FILE, d)
        st.session_state["_last_saved_hash"] = dict_hash(d)
        st.toast("Daten lokal gesichert!")

# ================================================================
# 7) TITEL + TAB-SETUP
# ================================================================
st.title(f"Fachplanung: {d['kunde'].get('name','—')}")

tabs = []
if "PV-System" in d["scope"]:
    tabs.append("☀️ PV & Dächer")
if "Speicher" in d["scope"]:
    tabs.append("🔋 Speicher")
if "Ladeinfrastruktur" in d["scope"]:
    tabs.append("🔌 E-Mobilität")
tabs.append("📊 KI-Analyse, ROI & Arbitrage")

tabs = st.tabs(tabs)
t_idx = 0

# ================================================================
# TAB: PV & DÄCHER
# ================================================================
if "PV-System" in d["scope"]:
    with tabs[t_idx]:
        st.header("☀️ PV-Planung & Dachflächen")

        # ------------------ Dachflächen ------------------
        with st.container(border=True):
            st.subheader("Dachflächen (mehrere möglich)")

            c_add, c_sum = st.columns([1, 3])
            with c_add:
                if st.button("➕ Dachfläche hinzufügen"):
                    d["daecher"].append({
                        "name": f"Dachfläche {len(d['daecher'])+1}",
                        "form": "Satteldach",
                        "breite": 10.0,
                        "tiefe": 6.0,
                        "flaeche": 60.0,
                        "flaeche_auto": True,
                        "neigung": 35,
                        "azimut": 0,
                        "nutzfaktor": 0.80,
                        "hinweis": "",
                    })
                    st.rerun()

            with c_sum:
                total_area, total_eff, modules_max = pv_potential_from_roofs(d["daecher"], module_area_m2=2.0)
                st.info(f"Summe Fläche: **{total_area:.1f} m²**, effektiv: **{total_eff:.1f} m²** → grob **{modules_max} Module** möglich")

            for i, r in enumerate(list(d["daecher"])):
                with st.expander(f"🏠 {r.get('name','Dachfläche')} (#{i+1})", expanded=(i == 0)):
                    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
                    r["name"] = c1.text_input("Name", r.get("name", f"Dachfläche {i+1}"), key=f"roof_name_{i}")
                    r["form"] = c2.selectbox("Dachform", DACH_FORMS, index=safe_index(DACH_FORMS, r.get("form", "Satteldach"), 0), key=f"roof_form_{i}")
                    r["neigung"] = c3.number_input("Neigung (°)", 0, 90, int(r.get("neigung", 35)), key=f"roof_tilt_{i}")
                    r["azimut"] = c4.number_input("Azimut (°)", -180, 180, int(r.get("azimut", 0)), key=f"roof_az_{i}")

                    c5, c6, c7, c8 = st.columns([1, 1, 1, 1])
                    r["breite"] = c5.number_input("Breite (m)", 0.0, 500.0, float(r.get("breite", 10.0)), key=f"roof_w_{i}")
                    r["tiefe"] = c6.number_input("Tiefe/Höhe (m)", 0.0, 200.0, float(r.get("tiefe", 6.0)), key=f"roof_d_{i}")

                    r["flaeche_auto"] = c7.checkbox("Fläche auto", value=bool(r.get("flaeche_auto", True)), key=f"roof_auto_{i}")
                    auto_a = roof_area(r)
                    if r["flaeche_auto"]:
                        r["flaeche"] = auto_a
                        c8.number_input("Fläche (m²)", 0.0, 10000.0, float(r["flaeche"]), disabled=True, key=f"roof_area_{i}")
                    else:
                        r["flaeche"] = c8.number_input("Fläche (m²)", 0.0, 10000.0, float(r.get("flaeche", auto_a)), key=f"roof_area_{i}")

                    r["nutzfaktor"] = st.slider("Nutzfaktor (Belegbarkeit, Abstände, Sperrflächen)", 0.30, 0.95, float(r.get("nutzfaktor", 0.80)), 0.01, key=f"roof_nf_{i}")
                    r["hinweis"] = st.text_input("Hinweis (optional)", r.get("hinweis", ""), key=f"roof_note_{i}")

                    if st.button("🗑️ Dachfläche löschen", key=f"roof_del_{i}"):
                        d["daecher"].pop(i)
                        if len(d["daecher"]) == 0:
                            d["daecher"].append(DEFAULTS["daecher"][0])
                        st.rerun()

        st.divider()

        # ------------------ PV Modulfelder (Data Editor) ------------------
        st.subheader("🟦 PV-Modulfelder (Strings)")

        roof_names = [r.get("name", f"Dachfläche {i+1}") for i, r in enumerate(d["daecher"])]
        pv_cols = ["roof", "hersteller", "typ", "watt_pro_modul", "anzahl_module"]
        pv_df = list_of_dicts_to_df(d["pv"].get("felder", []), pv_cols)

        if pv_df.empty:
            pv_df = pd.DataFrame([{
                "roof": roof_names[0] if roof_names else "Dachfläche 1",
                "hersteller": "",
                "typ": "440W Standard",
                "watt_pro_modul": 440,
                "anzahl_module": 12
            }])

        pv_df = st.data_editor(
            pv_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "roof": st.column_config.SelectboxColumn("Dachfläche", options=roof_names or ["Dachfläche 1"]),
                "hersteller": st.column_config.TextColumn("Hersteller"),
                "typ": st.column_config.TextColumn("Typ"),
                "watt_pro_modul": st.column_config.NumberColumn("Watt/Modul", min_value=0, max_value=800, step=5),
                "anzahl_module": st.column_config.NumberColumn("Module", min_value=0, max_value=50000, step=1),
            },
            key="pv_fields_editor",
        )

        d["pv"]["felder"] = df_to_list_of_dicts(pv_df)

        # PV totals
        total_kwp = 0.0
        total_modules = 0
        for row in d["pv"]["felder"]:
            try:
                w = float(row.get("watt_pro_modul", 0) or 0)
                n = float(row.get("anzahl_module", 0) or 0)
                total_kwp += (w * n) / 1000.0
                total_modules += int(n)
            except Exception:
                pass
        d["pv"]["total_kwp"] = float(total_kwp)

        # Roof capacity sanity
        _, _, modules_max = pv_potential_from_roofs(d["daecher"], module_area_m2=2.0)
        cA, cB, cC = st.columns(3)
        cA.metric("Gesamtleistung PV", f"{total_kwp:.2f} kWp")
        cB.metric("Module gesamt", f"{total_modules:,}")
        cC.metric("Dach-Kapazität grob", f"{modules_max:,} Module")

        if total_modules > modules_max and modules_max > 0:
            st.warning("⚠️ Plausibilitätscheck: Modulanzahl liegt über der groben Dach-Kapazität (Nutzfaktor/Modulfläche prüfen).")

        st.divider()

        # ------------------ Wechselrichter (Data Editor) ------------------
        st.subheader("🚥 Wechselrichter (AC)")

        wr_cols = ["modell", "ac_kw"]
        wr_df = list_of_dicts_to_df(d["pv"].get("wr", []), wr_cols)
        if wr_df.empty:
            wr_df = pd.DataFrame([{"modell": "", "ac_kw": 10.0}])

        wr_df = st.data_editor(
            wr_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "modell": st.column_config.TextColumn("Modell"),
                "ac_kw": st.column_config.NumberColumn("AC-Nennleistung (kW)", min_value=0.0, max_value=50000.0, step=0.5),
            },
            key="wr_editor",
        )

        d["pv"]["wr"] = df_to_list_of_dicts(wr_df)
        # Map to internal keys (optional)
        # keep compatibility: store as dicts with p/h
        mapped_wr = []
        for row in d["pv"]["wr"]:
            mapped_wr.append({"h": row.get("modell", ""), "p": float(row.get("ac_kw", 0.0) or 0.0)})
        d["pv"]["wr"] = mapped_wr

        d["pv"]["total_ac_kw"] = sum_wr_ac_kw(d["pv"]["wr"])
        st.metric("Summe WR-AC", f"{d['pv']['total_ac_kw']:.1f} kW")

        st.divider()

        # ------------------ Einspeisekonzept ------------------
        cur_k = d["pv"].get("konzept", "Überschusseinspeisung")
        d["pv"]["konzept"] = st.radio("Einspeisekonzept", PV_KONZEPTE, index=safe_index(PV_KONZEPTE, cur_k, 0), horizontal=True)

        if d["pv"]["konzept"] == "Volleinspeisung":
            st.warning("⚠️ Hinweis: Volleinspeisung benötigt i. d. R. separates Messkonzept/Zählerplatz.")

    t_idx += 1

# ================================================================
# TAB: SPEICHER
# ================================================================
if "Speicher" in d["scope"]:
    with tabs[t_idx]:
        st.header("🔋 Speicher (mit KI-Empfehlung)")

        left, right = st.columns([1, 1])

        with left:
            s = d["speicher"]
            s["hersteller"] = st.text_input("Hersteller / System", s.get("hersteller", ""))
            s["objective"] = st.selectbox("Ziel / Optimierung", OBJECTIVES, index=safe_index(OBJECTIVES, s.get("objective", OBJECTIVES[0]), 0))

            s["kap"] = st.number_input("Kapazität (kWh)", 0.0, 500000.0, float(s.get("kap", 0.0)))
            s["p"] = st.number_input("Max. Lade/Entladeleistung (kW)", 0.0, 500000.0, float(s.get("p", 0.0)))

            cur_type = s.get("spannung", "Hochvolt")
            s["spannung"] = st.selectbox("Typ", BAT_TYPES, index=safe_index(BAT_TYPES, cur_type, 0))

        with right:
            st.subheader("🤖 KI-Speicherempfehlung (Auto)")

            # Region für Empfehlung (wird auch im ROI-Tab nochmal gesetzt)
            region_rec = st.selectbox("Region (für PV-Ertrag-Heuristik)", list(REGIONS.keys()), index=safe_index(list(REGIONS.keys()), "Mitte", 1), key="region_rec")

            rec = recommend_storage(d, hak_kw=hak_kw, objective=d["speicher"].get("objective", OBJECTIVES[0]), region=region_rec)

            c1, c2, c3 = st.columns(3)
            c1.metric("Empf. Kapazität", f"{rec['rec_kwh']} kWh")
            c2.metric("Empf. Leistung", f"{rec['rec_kw']} kW")
            c3.metric("C-Rate", f"{rec['c_rate']} C")
            st.write(f"**Empfohlener Typ:** {rec['empf_typ']}")
            if rec["ems_empfohlen"]:
                st.warning("**EMS/Lastmanagement empfohlen**")

            if rec["gruende"]:
                st.caption("**Begründung:**")
                for g in rec["gruende"]:
                    st.write(f"- {g}")

            if rec["hinweise"]:
                st.caption("**Hinweise:**")
                for h in rec["hinweise"]:
                    st.write(f"- {h}")

            with st.expander("Details / Inputs"):
                st.json(rec)

            if float(d["speicher"].get("kap", 0.0)) <= 0.0:
                if st.button("✅ Empfehlung übernehmen (Speicher füllen)", use_container_width=True):
                    d["speicher"]["kap"] = rec["rec_kwh"]
                    d["speicher"]["p"] = rec["rec_kw"]
                    d["speicher"]["spannung"] = rec["empf_typ"]
                    st.toast("Speicherempfehlung übernommen.")
                    st.rerun()

    t_idx += 1

# ================================================================
# TAB: E-MOBILITÄT
# ================================================================
if "Ladeinfrastruktur" in d["scope"]:
    with tabs[t_idx]:
        st.header("🔌 Ladeinfrastruktur & Fuhrpark")

        # ------------------ Wallboxen (Data Editor) ------------------
        st.subheader("1) Ladestationen")
        wb_cols = ["name", "leistung_kw"]
        wb_df = list_of_dicts_to_df(d["mobilität"].get("ladepunkte", []), wb_cols)
        if wb_df.empty:
            wb_df = pd.DataFrame([{"name": "Ladepunkt", "leistung_kw": 11}])

        wb_df = st.data_editor(
            wb_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Bezeichnung/Modell"),
                "leistung_kw": st.column_config.SelectboxColumn("Leistung (kW)", options=WB_POWER_OPTIONS),
            },
            key="wb_editor",
        )

        d["mobilität"]["ladepunkte"] = df_to_list_of_dicts(wb_df)

        wb_sum = 0.0
        for lp in d["mobilität"]["ladepunkte"]:
            try:
                wb_sum += float(lp.get("leistung_kw", 0.0) or 0.0)
            except Exception:
                pass

        # HAK CHECK (sinnvoller mit WR-AC statt kWp)
        pv_ac = float(d["pv"].get("total_ac_kw", 0.0))
        if (pv_ac + wb_sum) > hak_kw:
            st.error(f"🚨 KRITISCH: WR-AC ({pv_ac:.1f} kW) + Ladeleistung ({wb_sum:.1f} kW) > HAK ({hak_kw:.1f} kW).")
            st.info("💡 Empfehlung: EMS/Lastmanagement und Netzanschlussprüfung zwingend.")

        st.divider()

        # ------------------ Fuhrpark (Data Editor) ------------------
        st.subheader("2) TCO-Vergleich: Diesel vs. Elektro")

        fleet_cols = ["klasse", "anzahl", "km_pro_jahr", "kwh_pro_100km", "l_pro_100km"]
        fleet_df = list_of_dicts_to_df(d["mobilität"].get("fuhrpark", []), fleet_cols)
        if fleet_df.empty:
            fleet_df = pd.DataFrame([{
                "klasse": "PKW",
                "anzahl": 1,
                "km_pro_jahr": 20000,
                "kwh_pro_100km": 18.0,
                "l_pro_100km": 7.0,
            }])

        fleet_df = st.data_editor(
            fleet_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "klasse": st.column_config.SelectboxColumn("Klasse", options=FZ_KLASSEN),
                "anzahl": st.column_config.NumberColumn("Stück", min_value=0, max_value=100000, step=1),
                "km_pro_jahr": st.column_config.NumberColumn("km/Jahr", min_value=0, max_value=5_000_000, step=100),
                "kwh_pro_100km": st.column_config.NumberColumn("kWh/100km", min_value=0.0, max_value=300.0, step=0.5),
                "l_pro_100km": st.column_config.NumberColumn("L/100km", min_value=0.0, max_value=120.0, step=0.1),
            },
            key="fleet_editor",
        )

        d["mobilität"]["fuhrpark"] = df_to_list_of_dicts(fleet_df)

        strompreis = float(d["wirtschaft"].get("strompreis", 0.35))
        dieselpreis = float(d["wirtschaft"].get("dieselpreis", 1.70))

        total_ev_kwh = 0.0
        total_savings = 0.0

        for fz in d["mobilität"]["fuhrpark"]:
            try:
                anz = float(fz.get("anzahl", 0) or 0)
                km = float(fz.get("km_pro_jahr", 0) or 0)
                econs = float(fz.get("kwh_pro_100km", 0) or 0)
                dcons = float(fz.get("l_pro_100km", 0) or 0)

                ev_kwh = (km / 100.0) * econs * anz
                di_l  = (km / 100.0) * dcons * anz

                ev_cost = ev_kwh * strompreis
                di_cost = di_l * dieselpreis

                total_ev_kwh += ev_kwh
                total_savings += (di_cost - ev_cost)
            except Exception:
                pass

        d["mobilität"]["total_ev_kwh"] = float(total_ev_kwh)

        c1, c2 = st.columns(2)
        c1.metric("EV-Strombedarf Fuhrpark", f"{total_ev_kwh:,.0f} kWh/a")
        c2.metric("Grobe Energiekosten-Ersparnis", f"{total_savings:,.0f} €/a")

        st.caption("Hinweis: TCO hier nur Energiekosten (keine Wartung/Steuern/Invest/Restwerte).")

    t_idx += 1

# ================================================================
# TAB: KI-ANALYSE, ROI & ARBITRAGE
# ================================================================
with tabs[t_idx]:
    st.header("📊 KI-Analyse, ROI & Arbitrage")

    # Globale Parameter
    with st.expander("Globale Parameter (Preise/Verbrauch/ROI-Annahmen)", expanded=True):
        cw1, cw2, cw3, cw4 = st.columns(4)
        d["wirtschaft"]["strompreis"] = cw1.number_input("Strompreis Netz (€/kWh)", 0.05, 2.50, float(d["wirtschaft"].get("strompreis", 0.35)))
        d["wirtschaft"]["einspeise_v"] = cw2.number_input("Einspeisevergütung (€/kWh)", 0.00, 1.00, float(d["wirtschaft"].get("einspeise_v", 0.08)))
        d["wirtschaft"]["dieselpreis"] = cw3.number_input("Dieselpreis (€/L)", 0.50, 5.00, float(d["wirtschaft"].get("dieselpreis", 1.70)))
        d["wirtschaft"]["lastgang_jahr"] = cw4.number_input("Verbrauch Gebäude (kWh/a)", 0, 10_000_000, int(d["wirtschaft"].get("lastgang_jahr", 5000)))

        cA, cB, cC, cD = st.columns(4)
        d["wirtschaft"]["capex_pv_eur_per_kwp"] = cA.number_input("CAPEX PV (€/kWp)", 200.0, 4000.0, float(d["wirtschaft"].get("capex_pv_eur_per_kwp", 1100.0)))
        d["wirtschaft"]["capex_bat_eur_per_kwh"] = cB.number_input("CAPEX Speicher (€/kWh)", 50.0, 2000.0, float(d["wirtschaft"].get("capex_bat_eur_per_kwh", 450.0)))
        d["wirtschaft"]["capex_bat_eur_per_kw"] = cC.number_input("CAPEX Speicher (€/kW)", 0.0, 2000.0, float(d["wirtschaft"].get("capex_bat_eur_per_kw", 200.0)))
        d["wirtschaft"]["discount_rate"] = cD.number_input("Diskontsatz (p.a.)", 0.0, 0.25, float(d["wirtschaft"].get("discount_rate", 0.06)))

        region = st.selectbox("Standort-Modell (Ertrag)", list(REGIONS.keys()), index=safe_index(list(REGIONS.keys()), "Mitte", 1), key="region_roi")

    # Wenn kein Speicher eingegeben: Empfehlung anzeigen
    if ("Speicher" not in d["scope"]) or float(d["speicher"].get("kap", 0.0)) <= 0.0:
        st.subheader("🔋 Speicherempfehlung (weil kein Speicher hinterlegt)")
        obj = d["speicher"].get("objective", "Eigenverbrauch/Autarkie")
        rec = recommend_storage(d, hak_kw=hak_kw, objective=obj, region=region)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kapazität", f"{rec['rec_kwh']} kWh")
        c2.metric("Leistung", f"{rec['rec_kw']} kW")
        c3.metric("C-Rate", f"{rec['c_rate']} C")
        c4.metric("Typ", rec["empf_typ"])

        if rec["ems_empfohlen"]:
            st.warning("EMS/Lastmanagement empfohlen – siehe Details.")
        with st.expander("Details"):
            st.json(rec)

        if st.button("✅ Empfehlung übernehmen", use_container_width=True):
            d["speicher"]["kap"] = rec["rec_kwh"]
            d["speicher"]["p"] = rec["rec_kw"]
            d["speicher"]["spannung"] = rec["empf_typ"]
            if "Speicher" not in d["scope"]:
                d["scope"].append("Speicher")
            st.rerun()

    st.divider()

    # Simulation + KPIs
    pv_kwp = float(d["pv"].get("total_kwp", 0.0))
    load_building = float(d["wirtschaft"].get("lastgang_jahr", 0.0))
    ev_kwh = float(d["mobilität"].get("total_ev_kwh", 0.0))
    bat_kwh = float(d["speicher"].get("kap", 0.0))
    bat_kw = float(d["speicher"].get("p", 0.0))

    if pv_kwp <= 0:
        st.info("Bitte zuerst PV konfigurieren (Tab „PV & Dächer“).")
    else:
        seed = stable_seed(P_SLUG, region, pv_kwp, load_building, ev_kwh, bat_kwh, bat_kw)

        sim = simulate_hourly(
            pv_kwp=pv_kwp,
            load_building_kwh_year=load_building,
            ev_kwh_year=ev_kwh,
            bat_kwh=bat_kwh,
            bat_kw=bat_kw,
            region=region,
            seed=seed,
            roundtrip_eff=0.92,
        )

        # Baseline ohne Speicher (für Mehrwert Speicher)
        sim_no_bat = simulate_hourly(
            pv_kwp=pv_kwp,
            load_building_kwh_year=load_building,
            ev_kwh_year=ev_kwh,
            bat_kwh=0.0,
            bat_kw=0.0,
            region=region,
            seed=stable_seed(P_SLUG, region, pv_kwp, load_building, ev_kwh, 0.0, 0.0),
            roundtrip_eff=0.92,
        )

        st.subheader("Ertrag vs. Last (Stundenwerte, Jahresverlauf)")
        st.line_chart(sim[["PV", "Last", "Served"]])

        total_gen = float(sim["PV"].sum())
        total_load = float(sim["Last"].sum())
        total_served = float(sim["Served"].sum())
        total_import = float(sim["Import"].sum())
        total_export = float(sim["Export"].sum())

        autarkie = (total_served / total_load * 100.0) if total_load > 0 else 0.0
        eigenq = (total_served / total_gen * 100.0) if total_gen > 0 else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("PV-Ertrag", f"{total_gen:,.0f} kWh/a")
        k2.metric("Autarkiegrad", f"{autarkie:.1f} %")
        k3.metric("Eigenverbrauchsquote", f"{eigenq:.1f} %")
        k4.metric("Netzbezug", f"{total_import:,.0f} kWh/a")
        st.caption(f"Netzeinspeisung (grob): {total_export:,.0f} kWh/a")

        # SoC Visual
        with st.expander("Batterie-SoC & Netzflüsse"):
            st.line_chart(sim[["SoC"]])
            st.line_chart(sim[["Import", "Export"]])

        # ---------------- ROI: einfache Wirtschaftlichkeit ----------------
        st.subheader("💶 ROI (grob)")

        strompreis = float(d["wirtschaft"]["strompreis"])
        feedin = float(d["wirtschaft"]["einspeise_v"])

        baseline_cost = total_load * strompreis
        system_cost = (total_import * strompreis) - (total_export * feedin)
        annual_savings = baseline_cost - system_cost

        # Speicher-Mehrwert (gegen PV-only)
        total_import_no = float(sim_no_bat["Import"].sum())
        total_export_no = float(sim_no_bat["Export"].sum())
        system_cost_no = (total_import_no * strompreis) - (total_export_no * feedin)
        storage_delta = system_cost_no - system_cost  # positive => storage helps

        # CAPEX/OPEX (grob)
        capex_pv = pv_kwp * float(d["wirtschaft"]["capex_pv_eur_per_kwp"])
        capex_bat = (bat_kwh * float(d["wirtschaft"]["capex_bat_eur_per_kwh"])) + (bat_kw * float(d["wirtschaft"]["capex_bat_eur_per_kw"]))
        capex_total = capex_pv + capex_bat

        opex = capex_pv * float(d["wirtschaft"]["opex_pv_pct"]) + capex_bat * float(d["wirtschaft"]["opex_bat_pct"])

        net_savings = annual_savings - opex

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Baseline Energiekosten", f"{baseline_cost:,.0f} €/a")
        cB.metric("Kosten mit PV+Speicher", f"{system_cost:,.0f} €/a")
        cC.metric("Ersparnis (brutto)", f"{annual_savings:,.0f} €/a")
        cD.metric("Ersparnis netto (−OPEX)", f"{net_savings:,.0f} €/a")

        cE, cF, cG = st.columns(3)
        cE.metric("Speicher-Mehrwert ggü. PV-only", f"{storage_delta:,.0f} €/a")
        cF.metric("CAPEX PV", f"{capex_pv:,.0f} €")
        cG.metric("CAPEX Speicher", f"{capex_bat:,.0f} €")

        if net_savings > 0:
            payback = capex_total / net_savings
            st.info(f"Grobe Amortisation: **{payback:.1f} Jahre** (vereinfachtes Modell).")
        else:
            st.warning("Netto-Ersparnis ist ≤ 0 → CAPEX/OPEX/Preise/Dimensionierung prüfen.")

        # ---------------- Arbitrage ----------------
        st.divider()
        st.subheader("📈 Arbitrage (optional)")

        d["wirtschaft"]["arb_enabled"] = st.checkbox("Arbitrage berücksichtigen", value=bool(d["wirtschaft"].get("arb_enabled", False)))

        if d["wirtschaft"]["arb_enabled"]:
            c1, c2, c3, c4 = st.columns(4)
            d["wirtschaft"]["arb_low_price"] = c1.number_input("Niedrigpreis (€/kWh)", 0.00, 2.50, float(d["wirtschaft"].get("arb_low_price", 0.18)))
            d["wirtschaft"]["arb_high_price"] = c2.number_input("Hochpreis (€/kWh)", 0.00, 2.50, float(d["wirtschaft"].get("arb_high_price", 0.32)))
            d["wirtschaft"]["arb_cycles_per_year"] = c3.number_input("Zyklen/Jahr", 0, 1000, int(d["wirtschaft"].get("arb_cycles_per_year", 180)))
            d["wirtschaft"]["arb_roundtrip_eff"] = c4.number_input("Roundtrip-Effizienz", 0.50, 0.99, float(d["wirtschaft"].get("arb_roundtrip_eff", 0.90)))

            c5, c6 = st.columns(2)
            d["wirtschaft"]["arb_dod"] = c5.number_input("DoD (nutzbarer Anteil)", 0.10, 1.00, float(d["wirtschaft"].get("arb_dod", 0.90)))
            d["wirtschaft"]["arb_degradation_eur_per_kwh"] = c6.number_input("Degradation/Kosten je Durchsatz (€/kWh)", 0.00, 0.50, float(d["wirtschaft"].get("arb_degradation_eur_per_kwh", 0.03)))

            low = float(d["wirtschaft"]["arb_low_price"])
            high = float(d["wirtschaft"]["arb_high_price"])
            spread = max(high - low, 0.0)
            cycles = int(d["wirtschaft"]["arb_cycles_per_year"])
            eff = float(d["wirtschaft"]["arb_roundtrip_eff"])
            dod = float(d["wirtschaft"]["arb_dod"])
            deg = float(d["wirtschaft"]["arb_degradation_eur_per_kwh"])

            usable_kwh = max(bat_kwh * dod, 0.0)
            # Vereinfachte Arbitrage-Marge:
            # profit ≈ discharged_kWh * spread * eff  - throughput_kWh * degradation_cost
            discharged_kwh_year = usable_kwh * cycles
            throughput_kwh_year = usable_kwh * cycles * 2.0  # charge+discharge
            arb_profit = discharged_kwh_year * spread * eff - throughput_kwh_year * deg

            st.metric("Arbitrage-Potenzial (grob)", f"{arb_profit:,.0f} €/a")
            st.caption("Hinweis: stark abhängig von Markt/Regeln, Netzentgelten, Messkonzept, Vermarktungs-Setup, Steuerung/EMS.")

            # Gesamtbild mit Arbitrage
            net_savings_with_arb = net_savings + arb_profit
            st.info(f"Ersparnis netto inkl. Arbitrage: **{net_savings_with_arb:,.0f} €/a**")
            if net_savings_with_arb > 0:
                payback2 = capex_total / net_savings_with_arb
                st.info(f"Grobe Amortisation inkl. Arbitrage: **{payback2:.1f} Jahre**")

        # ---------------- KI Hinweise ----------------
        st.divider()
        st.subheader("🤖 KI-Hinweise (Plausibilität & nächste Schritte)")

        if float(d["speicher"].get("kap", 0.0)) <= 0:
            st.info("Kein Speicher hinterlegt → oben wird eine Empfehlung angezeigt (inkl. Übernehmen-Button).")

        if pv_kwp < (load_building + ev_kwh) / 1000.0 * 0.8:
            st.warning("PV wirkt im Verhältnis zum Jahresverbrauch eher klein → Dach-Vollausbau prüfen (wenn wirtschaftlich/technisch möglich).")

        # Anschluss & EMS Indikator
        wb_sum = sum(float(lp.get("leistung_kw", 0.0) or 0.0) for lp in d["mobilität"].get("ladepunkte", []))
        pv_ac = float(d["pv"].get("total_ac_kw", 0.0))
        if (pv_ac + wb_sum) > hak_kw:
            st.error("HAK-Konflikt: WR-AC + Ladeleistung über HAK → EMS/LM + Netzprüfung zwingend.")
        elif wb_sum > 0:
            st.info("Mit Ladeinfrastruktur: dynamisches Lastmanagement/EMS bringt i. d. R. deutliche Vorteile (HAK-Nutzung, PV-Überschussladung).")

# ================================================================
# 8) FINAL: AUTOSAVE NUR BEI ÄNDERUNG
# ================================================================
save_if_changed(STATE_FILE, d)
