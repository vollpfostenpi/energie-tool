import streamlit as st
import json
from pathlib import Path
from datetime import datetime, date, timedelta
import io
import zipfile

import pandas as pd
import numpy as np

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Lastganganalyse", layout="wide", page_icon="📊")

# ----------------------------
# Pfade
# ----------------------------
APP_DIR = Path(__file__).resolve().parents[1]  # .../energie-tool
PROJECTS_DIR = APP_DIR / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Defaults / State Schema
# ----------------------------
DEFAULTS = {
    "kunde": {"name": "—", "plz_ort": ""},
    "scope": ["PV-System", "Speicher", "Ladeinfrastruktur"],
    "tech": {"hak_mode": "Ampere", "hak_ampere": 63, "hak_kw": 43.6},
    "pv": {"total_kwp": 0.0, "total_ac_kw": 0.0},
    "speicher": {"kap": 0.0, "p": 0.0, "min_soc_pct": 10.0, "objective": "Eigenverbrauch/Autarkie"},
    "mobilität": {"total_ev_kwh": 0.0},
    "wirtschaft": {
        "lastgang_file": "",
        "lastgang_datetime_col": "timestamp",
        "lastgang_value_col": "power_kw",
        "lastgang_sep": ";",
        "lastgang_decimal": ".",
        "lastgang_resolution": "15min",
        "lastgang_is_power_kw": True,
        "lastgang_jahr": 0.0,
        "lastgang_meta_file": "",
        "lastgang_period_start": "",
        "lastgang_period_end": "",
        "lastgang_annualized_kwh": 0.0,
        "lastgang_source": "",  # Import / Muster / KI
    },
}

# ----------------------------
# Helpers
# ----------------------------
def deep_merge(target: dict, defaults: dict) -> None:
    for k, v in defaults.items():
        if k not in target:
            target[k] = v
        else:
            if isinstance(v, dict) and isinstance(target.get(k), dict):
                deep_merge(target[k], v)

def sanitize(obj):
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

def project_paths(slug: str):
    base = PROJECTS_DIR / slug
    docs = base / "documents"
    lastgang_dir = docs / "lastgang"
    inputs = docs / "inputs"
    reports = docs / "reports"
    for p in [docs, lastgang_dir, inputs, reports]:
        p.mkdir(parents=True, exist_ok=True)
    return base, docs, lastgang_dir, inputs, reports

def load_project(slug: str) -> tuple[dict, Path]:
    base = PROJECTS_DIR / slug
    state_file = base / "state.json"
    if not state_file.exists():
        st.error("⚠️ state.json nicht gefunden. Bitte Projekt im Dashboard anlegen/öffnen.")
        st.stop()
    d = json.loads(state_file.read_text(encoding="utf-8"))
    deep_merge(d, DEFAULTS)
    return d, state_file

def save_project(state_file: Path, d: dict):
    state_file.write_text(json.dumps(sanitize(d), indent=2, ensure_ascii=False), encoding="utf-8")

def robust_read_csv(uploaded_bytes: bytes, sep: str, decimal: str, encoding_hint: str):
    # Versuch: user hint > utf-8-sig > utf-8 > cp1252 > latin1
    encodings = []
    if encoding_hint and encoding_hint != "auto":
        encodings.append(encoding_hint)
    encodings += ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(uploaded_bytes), sep=sep, decimal=decimal, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"CSV konnte nicht gelesen werden. Letzter Fehler: {last_err}")

def make_time_index(mode: str, resolution: str, year: int, start_d: date, end_d: date) -> pd.DatetimeIndex:
    freq = "15T" if resolution == "15min" else "H"
    if mode == "Ganzes Jahr":
        start = datetime(year, 1, 1, 0, 0)
        end = datetime(year + 1, 1, 1, 0, 0)
    else:
        start = datetime(start_d.year, start_d.month, start_d.day, 0, 0)
        end = datetime(end_d.year, end_d.month, end_d.day, 0, 0) + timedelta(days=1)
    idx = pd.date_range(start, end, freq=freq, inclusive="left")
    return idx

def _shape_residential(idx: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    h = idx.hour.values + idx.minute.values / 60.0
    doy = idx.dayofyear.values
    seasonal = 0.95 + 0.20 * np.cos(2 * np.pi * (doy / 365.0))  # Winter höher
    morning = np.exp(-0.5 * ((h - 7.3) / 1.6) ** 2)
    evening = np.exp(-0.5 * ((h - 19.5) / 2.2) ** 2)
    base = 0.35 + 0.10 * np.cos(2 * np.pi * (h - 3) / 24)
    weekend = np.where(idx.dayofweek.values >= 5, 1.12, 1.0)
    noise = np.clip(rng.normal(1.0, 0.12, len(idx)), 0.65, 1.55)
    raw = (base + 1.1 * morning + 1.6 * evening) * seasonal * weekend * noise
    return raw

def _shape_commercial(idx: pd.DatetimeIndex, rng: np.random.Generator, sector: str, shifts: int) -> np.ndarray:
    h = idx.hour.values + idx.minute.values / 60.0
    dow = idx.dayofweek.values
    is_weekend = (dow >= 5)

    # Grundprofile pro Sektor
    if sector == "Logistik/Spedition":
        base = 0.85
        peak = 0.45
        night = 0.25
    elif sector == "Maschinenbau/Produktion":
        base = 0.55
        peak = 0.85
        night = 0.10
    elif sector == "Lebensmittel/Frische":
        base = 0.75
        peak = 0.55
        night = 0.25
    else:  # "Sonstiges Gewerbe"
        base = 0.50
        peak = 0.70
        night = 0.10

    # Schichtfenster
    def window(a, b):
        return ((h >= a) & (h < b)).astype(float)

    if shifts <= 1:
        work = window(7, 17)
        off = 1.0 - work
        prof = base + peak * work + night * off
        prof *= np.where(is_weekend, 0.35 if sector != "Logistik/Spedition" else 0.75, 1.0)
    elif shifts == 2:
        s1 = window(6, 14)
        s2 = window(14, 22)
        work = np.clip(s1 + s2, 0, 1)
        off = 1.0 - work
        prof = base + (peak * 0.9) * work + (night * 0.6) * off
        prof *= np.where(is_weekend, 0.30 if sector != "Logistik/Spedition" else 0.70, 1.0)
    else:  # 3 Schicht -> 24/7
        prof = base + peak * 0.55 + night * 0.35
        # leichtes Tagesmuster
        daily = 0.90 + 0.18 * np.cos(2 * np.pi * (h - 14) / 24)
        prof = prof * daily
        prof *= np.where(is_weekend, 0.95 if sector == "Logistik/Spedition" else 0.85, 1.0)

    # Saison (z.B. Heizung/Prozesswärme leicht winterlastig)
    doy = idx.dayofyear.values
    seasonal = 0.98 + 0.08 * np.cos(2 * np.pi * (doy / 365.0))
    noise = np.clip(rng.normal(1.0, 0.10, len(idx)), 0.70, 1.45)
    raw = prof * seasonal * noise
    return raw

def generate_load_kw(
    idx: pd.DatetimeIndex,
    annual_kwh: float,
    profile_type: str,
    sector: str,
    shifts: int,
    base_kw_min: float,
    peak_cap_kw: float | None,
    seed: int,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    if len(idx) == 0:
        return pd.Series(dtype=float)

    if profile_type == "Haushalt":
        raw = _shape_residential(idx, rng)
    else:
        raw = _shape_commercial(idx, rng, sector=sector, shifts=shifts)

    # in kW skalieren: Zielenergie = annual_kwh (für Ganzes Jahr) oder period_kwh (für custom)
    # Wir skalieren erst auf Energie über die Indexlänge:
    dt_h = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1.0
    raw_energy = raw.sum() * dt_h  # "kWh" wenn raw als kW interpretiert würde

    target_kwh = max(float(annual_kwh), 0.0)
    if raw_energy <= 0:
        scaled = np.zeros(len(idx))
    else:
        scaled = raw * (target_kwh / raw_energy)

    # Mindestgrundlast (kW)
    base_kw_min = max(float(base_kw_min), 0.0)
    scaled = np.maximum(scaled, base_kw_min)

    # Peak-Cap optional
    if peak_cap_kw is not None:
        scaled = np.minimum(scaled, max(float(peak_cap_kw), 0.0))

    return pd.Series(index=idx, data=scaled, name="power_kw")

def kpis_from_kw_series(s_kw: pd.Series) -> dict:
    if s_kw is None or s_kw.empty:
        return {"kwh_total": 0.0, "kw_peak": 0.0, "kw_mean": 0.0}
    dt_h = (s_kw.index[1] - s_kw.index[0]).total_seconds() / 3600.0 if len(s_kw) > 1 else 1.0
    kwh_total = float((s_kw * dt_h).sum())
    return {
        "kwh_total": kwh_total,
        "kw_peak": float(s_kw.max()),
        "kw_mean": float(s_kw.mean()),
    }

def save_lastgang_to_project(
    slug: str,
    d: dict,
    state_file: Path,
    series_kw: pd.Series,
    resolution: str,
    source: str,
    meta: dict,
):
    _, _, lastgang_dir, _, _ = project_paths(slug)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = lastgang_dir / f"lastgang_{source.lower()}_{ts}_{resolution}.csv"
    meta_path = lastgang_dir / f"lastgang_{source.lower()}_{ts}_{resolution}.json"

    df_out = pd.DataFrame({
        "timestamp": series_kw.index.astype("datetime64[ns]"),
        "power_kw": series_kw.values.astype(float),
    })
    # Speichern robust (German-friendly, aber eindeutig)
    df_out.to_csv(csv_path, index=False, sep=";", decimal=".")

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # state.json setzen, sodass Planung es sofort nutzt:
    d["wirtschaft"]["lastgang_file"] = str(csv_path.as_posix())
    d["wirtschaft"]["lastgang_datetime_col"] = "timestamp"
    d["wirtschaft"]["lastgang_value_col"] = "power_kw"
    d["wirtschaft"]["lastgang_sep"] = ";"
    d["wirtschaft"]["lastgang_decimal"] = "."
    d["wirtschaft"]["lastgang_resolution"] = resolution
    d["wirtschaft"]["lastgang_is_power_kw"] = True
    d["wirtschaft"]["lastgang_meta_file"] = str(meta_path.as_posix())
    d["wirtschaft"]["lastgang_source"] = source

    # KPI / Annualisierung:
    k = kpis_from_kw_series(series_kw)
    d["wirtschaft"]["lastgang_jahr"] = float(k["kwh_total"])
    d["wirtschaft"]["lastgang_annualized_kwh"] = float(k["kwh_total"])

    save_project(state_file, d)

# ----------------------------
# Active project
# ----------------------------
if "active_slug" not in st.session_state or not st.session_state["active_slug"]:
    st.error("⚠️ Kein aktives Projekt. Bitte im Dashboard ein Projekt auswählen.")
    st.stop()

SLUG = st.session_state["active_slug"]
d, STATE_FILE = load_project(SLUG)
_, _, LASTGANG_DIR, INPUTS_DIR, _ = project_paths(SLUG)

st.title("📊 PHASE 1 – Lastganganalyse")
st.caption(f"Aktives Projekt: **{SLUG}**")

# ----------------------------
# Aktueller Lastgang Status
# ----------------------------
w = d["wirtschaft"]
cur = w.get("lastgang_file", "")
colA, colB, colC = st.columns([2, 1, 1])
with colA:
    st.info(f"**Aktueller Lastgang:** {cur if cur else '— (noch keiner gesetzt)'}")
with colB:
    st.metric("Auflösung", str(w.get("lastgang_resolution", "15min")))
with colC:
    st.metric("Quelle", str(w.get("lastgang_source", "—")))

st.divider()

# ----------------------------
# Tabs
# ----------------------------
t_import, t_muster, t_ki = st.tabs(["📥 Import", "📦 Musterlastgänge", "🤖 KI-Lastgang"])

# =========================================================
# TAB Import
# =========================================================
with t_import:
    st.subheader("📥 CSV Import (lokal → Projekt)")
    st.caption("Du lädst die Datei hoch, sie wird ins Projekt nach `documents/inputs/` gespeichert und als Lastgang gesetzt.")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    sep = c1.text_input("Separator", value=w.get("lastgang_sep", ";"))
    dec = c2.text_input("Dezimal", value=w.get("lastgang_decimal", "."))
    res = c3.selectbox("Auflösung", ["15min", "hour"], index=0 if w.get("lastgang_resolution", "15min") == "15min" else 1)
    enc = c4.selectbox("Encoding", ["auto", "utf-8", "utf-8-sig", "cp1252", "latin1"], index=0)

    is_power = st.checkbox("Werte sind Leistung (kW) (nicht kWh je Schritt)", value=True)
    up = st.file_uploader("CSV hochladen", type=["csv"])

    if up is not None:
        try:
            df = robust_read_csv(up.getvalue(), sep=sep, decimal=dec, encoding_hint=enc)
            st.success(f"CSV gelesen: {df.shape[0]:,} Zeilen / {df.shape[1]} Spalten")
            st.dataframe(df.head(20), use_container_width=True)

            cols = list(df.columns)
            dt_col = st.selectbox("Datetime-Spalte", options=cols, index=0)
            val_col = st.selectbox("Werte-Spalte", options=cols, index=min(1, len(cols)-1))

            if st.button("✅ Import übernehmen (ins Projekt speichern)", use_container_width=True):
                # speichern
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = INPUTS_DIR / f"import_lastgang_{ts}_{up.name.replace('/','_').replace('\\','_')}"
                save_path.write_bytes(up.getvalue())

                # state setzen
                w["lastgang_file"] = str(save_path.as_posix())
                w["lastgang_datetime_col"] = dt_col
                w["lastgang_value_col"] = val_col
                w["lastgang_sep"] = sep
                w["lastgang_decimal"] = dec
                w["lastgang_resolution"] = res
                w["lastgang_is_power_kw"] = bool(is_power)
                w["lastgang_source"] = "Import"

                # KPI: kurz parsen
                df2 = df.copy()
                df2[dt_col] = pd.to_datetime(df2[dt_col], errors="coerce")
                df2 = df2.dropna(subset=[dt_col]).sort_values(dt_col)
                s = pd.to_numeric(df2[val_col], errors="coerce").fillna(0.0)
                idx = pd.DatetimeIndex(df2[dt_col].values)
                s.index = idx
                # resample
                s = s.sort_index()
                if res == "15min":
                    s = s.resample("15T").mean().interpolate(limit_direction="both")
                else:
                    s = s.resample("H").mean().interpolate(limit_direction="both")

                if is_power:
                    dt_h = (s.index[1] - s.index[0]).total_seconds()/3600.0 if len(s) > 1 else 1.0
                    kwh_total = float((s * dt_h).sum())
                else:
                    kwh_total = float(s.sum())

                w["lastgang_jahr"] = kwh_total
                w["lastgang_annualized_kwh"] = kwh_total
                w["lastgang_period_start"] = str(s.index.min())
                w["lastgang_period_end"] = str(s.index.max())

                save_project(STATE_FILE, d)
                st.success("✅ Import übernommen – Planung nutzt diesen Lastgang jetzt.")
                st.rerun()

        except Exception as e:
            st.error(str(e))

# =========================================================
# TAB Musterlastgänge
# =========================================================
with t_muster:
    st.subheader("📦 Musterlastgänge (schnell)")
    st.caption("Erzeugt ein plausibles synthetisches Profil (ohne externe Daten) und speichert es als Projekt-Lastgang.")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    resolution = c1.selectbox("Auflösung", ["15min", "hour"], index=0)
    mode = c2.selectbox("Dauer", ["Ganzes Jahr", "Zeitraum"], index=0)
    year = c3.number_input("Jahr", 2000, 2100, int(datetime.now().year), step=1)
    annual_kwh = c4.number_input("Zielverbrauch (kWh)", 0.0, 50_000_000.0, float(w.get("lastgang_jahr", 5000.0) or 5000.0), step=100.0)

    if mode == "Zeitraum":
        start_d = st.date_input("Startdatum", value=date(year, 1, 1))
        end_d = st.date_input("Enddatum", value=date(year, 12, 31))
    else:
        start_d = date(year, 1, 1)
        end_d = date(year, 12, 31)

    profile = st.selectbox("Profil", ["Haushalt", "Gewerbe (Standard)", "Logistik/Spedition", "Maschinenbau/Produktion"], index=0)

    base_kw = st.number_input("Mindest-Grundlast (kW)", 0.0, 100000.0, 0.2, step=0.1)
    peak_cap = st.checkbox("Peak begrenzen", value=False)
    peak_kw = st.number_input("Max Peak (kW)", 0.0, 200000.0, 50.0, step=1.0) if peak_cap else None

    if st.button("⚙️ Musterlastgang erzeugen & übernehmen", use_container_width=True):
        idx = make_time_index(mode, resolution, int(year), start_d, end_d)

        if profile == "Haushalt":
            ptype = "Haushalt"
            sector = "—"
            shifts = 1
        else:
            ptype = "Gewerbe"
            if profile == "Logistik/Spedition":
                sector = "Logistik/Spedition"
                shifts = 3
            elif profile == "Maschinenbau/Produktion":
                sector = "Maschinenbau/Produktion"
                shifts = 2
            else:
                sector = "Sonstiges Gewerbe"
                shifts = 1

        # Annualisierung bei Zeitraum:
        period_days = max((end_d - start_d).days + 1, 1)
        target_kwh = float(annual_kwh)
        if mode == "Zeitraum":
            target_kwh = float(annual_kwh) * (period_days / 365.0)

        s_kw = generate_load_kw(
            idx=idx,
            annual_kwh=target_kwh,
            profile_type=ptype,
            sector=sector,
            shifts=int(shifts),
            base_kw_min=float(base_kw),
            peak_cap_kw=peak_kw,
            seed=abs(hash((SLUG, profile, resolution, str(start_d), str(end_d)))) % (2**32),
        )

        k = kpis_from_kw_series(s_kw)
        annualized = float(k["kwh_total"]) * (365.0 / period_days) if mode == "Zeitraum" else float(k["kwh_total"])

        meta = {
            "source": "Muster",
            "profile": profile,
            "resolution": resolution,
            "duration_mode": mode,
            "year": int(year),
            "start": str(start_d),
            "end": str(end_d),
            "period_days": int(period_days),
            "kwh_total_period": float(k["kwh_total"]),
            "kwh_annualized": float(annualized),
            "kw_peak": float(k["kw_peak"]),
            "base_kw_min": float(base_kw),
            "peak_cap_kw": float(peak_kw) if peak_kw is not None else None,
        }

        save_lastgang_to_project(SLUG, d, STATE_FILE, s_kw, resolution, "Muster", meta)

        # state annualized
        d["wirtschaft"]["lastgang_period_start"] = str(start_d)
        d["wirtschaft"]["lastgang_period_end"] = str(end_d)
        d["wirtschaft"]["lastgang_jahr"] = float(k["kwh_total"])
        d["wirtschaft"]["lastgang_annualized_kwh"] = float(annualized)
        save_project(STATE_FILE, d)

        st.success(f"✅ Musterlastgang übernommen. Periodenverbrauch: {k['kwh_total']:,.0f} kWh | Peak: {k['kw_peak']:.1f} kW")
        st.line_chart(s_kw.tail(7*24*(4 if resolution=="15min" else 1)))
        st.rerun()

# =========================================================
# TAB KI
# =========================================================
with t_ki:
    st.subheader("🤖 KI-Lastgang (parameterbasiert, realistisch)")
    st.caption("Hier erzeugst du einen Lastgang für **ganzes Jahr** oder **benutzerdefinierten Zeitraum**. "
               "Die Daten werden im Projekt gespeichert und in der Planung/Simulation verwendet.")

    c1, c2, c3 = st.columns([1, 1, 1])
    resolution = c1.selectbox("Auflösung", ["15min", "hour"], index=0, key="ki_res")
    duration_mode = c2.selectbox("Dauer", ["Ganzes Jahr", "Zeitraum"], index=0, key="ki_dur")
    year = c3.number_input("Jahr", 2000, 2100, int(datetime.now().year), step=1, key="ki_year")

    if duration_mode == "Zeitraum":
        start_d = st.date_input("Startdatum", value=date(year, 1, 1), key="ki_start")
        end_d = st.date_input("Enddatum", value=date(year, 12, 31), key="ki_end")
    else:
        start_d = date(year, 1, 1)
        end_d = date(year, 12, 31)

    st.divider()

    colL, colR = st.columns([1.2, 1])
    with colL:
        ptype = st.selectbox("Nutzungsart", ["Haushalt", "Gewerbe"], index=1)
        if ptype == "Gewerbe":
            sector = st.selectbox("Branche", ["Logistik/Spedition", "Maschinenbau/Produktion", "Lebensmittel/Frische", "Sonstiges Gewerbe"], index=0)
            shift_enabled = st.checkbox("Schichtsystem?", value=True)
            if shift_enabled:
                shifts = st.selectbox("Schichten", [2, 3], index=0)
            else:
                shifts = 1
        else:
            sector = "—"
            shifts = 1

        annual_kwh_input = st.number_input("Zielverbrauch (kWh/a)", 0.0, 50_000_000.0, float(w.get("lastgang_annualized_kwh", 5000.0) or 5000.0), step=100.0)
        base_kw_min = st.number_input("Mindest-Grundlast (kW)", 0.0, 100000.0, 0.3 if ptype=="Haushalt" else 2.0, step=0.1)

        cap_peak = st.checkbox("Peak begrenzen (optional)", value=False)
        peak_cap_kw = st.number_input("Max Peak (kW)", 0.0, 200000.0, 80.0, step=1.0) if cap_peak else None

    with colR:
        st.markdown("### Vorschau-Logik")
        st.write("- Haushalt: Morgen/Abend-Peaks, Winter höher")
        st.write("- Gewerbe: Branche + Schichten beeinflussen Tages-/Wochenmuster")
        st.write("- Zeitraum: wird intern auf den Zeitraum skaliert und **für Speicher/ROI annualisiert**")

    if st.button("🚀 KI-Lastgang erzeugen & übernehmen", use_container_width=True):
        idx = make_time_index(duration_mode, resolution, int(year), start_d, end_d)

        period_days = max((end_d - start_d).days + 1, 1)

        # Zielenergie für Zeitraum:
        target_kwh_period = float(annual_kwh_input)
        if duration_mode == "Zeitraum":
            target_kwh_period = float(annual_kwh_input) * (period_days / 365.0)

        s_kw = generate_load_kw(
            idx=idx,
            annual_kwh=target_kwh_period,
            profile_type=ptype,
            sector=sector,
            shifts=int(shifts),
            base_kw_min=float(base_kw_min),
            peak_cap_kw=peak_cap_kw,
            seed=abs(hash((SLUG, "KI", ptype, sector, shifts, resolution, str(start_d), str(end_d)))) % (2**32),
        )

        k = kpis_from_kw_series(s_kw)
        annualized = float(k["kwh_total"]) * (365.0 / period_days) if duration_mode == "Zeitraum" else float(k["kwh_total"])

        meta = {
            "source": "KI",
            "profile_type": ptype,
            "sector": sector,
            "shifts": int(shifts),
            "resolution": resolution,
            "duration_mode": duration_mode,
            "year": int(year),
            "start": str(start_d),
            "end": str(end_d),
            "period_days": int(period_days),
            "kwh_total_period": float(k["kwh_total"]),
            "kwh_annualized": float(annualized),
            "kw_peak": float(k["kw_peak"]),
            "kw_mean": float(k["kw_mean"]),
            "base_kw_min": float(base_kw_min),
            "peak_cap_kw": float(peak_cap_kw) if peak_cap_kw is not None else None,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        save_lastgang_to_project(SLUG, d, STATE_FILE, s_kw, resolution, "KI", meta)

        # state annualized
        d["wirtschaft"]["lastgang_period_start"] = str(start_d)
        d["wirtschaft"]["lastgang_period_end"] = str(end_d)
        d["wirtschaft"]["lastgang_jahr"] = float(k["kwh_total"])
        d["wirtschaft"]["lastgang_annualized_kwh"] = float(annualized)
        save_project(STATE_FILE, d)

        st.success(f"✅ KI-Lastgang übernommen. Periodenverbrauch: {k['kwh_total']:,.0f} kWh | Annualisiert: {annualized:,.0f} kWh/a | Peak: {k['kw_peak']:.1f} kW")
        st.line_chart(s_kw.tail(7*24*(4 if resolution=="15min" else 1)))
        st.rerun()

st.divider()

# Quick-Navigation
cN1, cN2 = st.columns([1, 1])
with cN1:
    if st.button("➡️ Weiter zu Planung (Phase 2)", use_container_width=True):
        st.switch_page("pages/2_Planung.py")
with cN2:
    if st.button("⬅️ Zurück zum Dashboard", use_container_width=True):
        st.switch_page("app.py")
