import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import io
import hashlib
import zlib

import pandas as pd
import numpy as np

# Optional libs
try:
    import requests
except Exception:
    requests = None

st.set_page_config(page_title="Phase 1 – Lastganganalyse", layout="wide", page_icon="📊")


# ----------------------------
# Helpers / Defaults
# ----------------------------
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


def deep_merge(target: dict, defaults: dict) -> None:
    for k, v in defaults.items():
        if k not in target:
            target[k] = v
        else:
            if isinstance(v, dict) and isinstance(target.get(k), dict):
                deep_merge(target[k], v)


def dict_hash(obj: dict) -> str:
    s = json.dumps(sanitize(obj), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(s).hexdigest()


def stable_seed(*parts) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECTS_DIR = BASE_DIR / "projects"


DEFAULTS = {
    "kunde": {"name": "—"},
    "scope": ["PV-System", "Speicher", "Ladeinfrastruktur"],
    "tech": {"hak_mode": "Ampere", "hak_ampere": 63, "hak_kw": 43.6, "ems_required": False},
    "daecher": [],
    "pv": {"felder": [], "wr": [], "konzept": "Überschusseinspeisung", "total_kwp": 0.0, "total_ac_kw": 0.0},
    "speicher": {
        "hersteller": "",
        "kap": 0.0,
        "p": 0.0,
        "spannung": "Hochvolt",
        "datasheet": "",
        "objective": "Eigenverbrauch/Autarkie",
        "min_soc_pct": 10.0,
        "eta_roundtrip": 0.92,
        "arbitrage": {"enabled": False, "mode": "Manuell"},
    },
    "mobilität": {"ladepunkte": [], "fuhrpark": [], "total_ev_kwh": 0.0},
    "wirtschaft": {
        "strompreis": 0.35,
        "dieselpreis": 1.70,
        "einspeise_v": 0.08,
        "lastgang_jahr": 5000.0,
        "lastgang_file": "",
        "lastgang_datetime_col": "",
        "lastgang_value_col": "",
        "lastgang_sep": ";",
        "lastgang_decimal": ",",
        "lastgang_resolution": "15min",
        "lastgang_is_power_kw": True,
        "lastgang_source": "",
        "lastgang_ai": {},
    },
}


def project_paths(slug: str):
    base = PROJECTS_DIR / slug
    docs = base / "documents"
    inputs = docs / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    return base, docs, inputs, base / "state.json"


def load_project(slug: str) -> dict:
    _, _, _, state_file = project_paths(slug)
    if not state_file.exists():
        st.error("Projektstate nicht gefunden. Bitte Projekt über Dashboard anlegen/öffnen.")
        st.stop()
    with state_file.open("r", encoding="utf-8") as f:
        d = json.load(f)
    deep_merge(d, DEFAULTS)
    # Minimal: wenn daecher leer, ok – Planung kümmert sich.
    # Sicherstellen, dass wirtschaft keys vollständig sind:
    deep_merge(d.get("wirtschaft", {}), DEFAULTS["wirtschaft"])
    return d


def save_project(slug: str, d: dict):
    _, _, _, state_file = project_paths(slug)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(sanitize(d), f, indent=4, ensure_ascii=False)


def save_uploaded_file(folder: Path, uploaded_file, prefix: str = "") -> str:
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = uploaded_file.name.replace("/", "_").replace("\\", "_")
    filename = f"{prefix}{ts}_{safe_name}"
    path = folder / filename
    path.write_bytes(uploaded_file.getbuffer())
    return str(path)


@st.cache_data(show_spinner=False)
def _read_csv_bytes(content: bytes, sep: str, decimal: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), sep=sep, decimal=decimal)


def _ensure_datetime_index(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
    out = out.dropna(subset=[dt_col]).sort_values(dt_col).set_index(dt_col)
    return out


def _resample_series(s: pd.Series, resolution: str) -> pd.Series:
    if resolution == "15min":
        return s.resample("15T").mean().interpolate(limit_direction="both")
    return s.resample("H").mean().interpolate(limit_direction="both")


def write_lastgang_csv(path: Path, s_kwh_per_step: pd.Series):
    df = pd.DataFrame({"timestamp": s_kwh_per_step.index, "kwh": s_kwh_per_step.values})
    df.to_csv(path, index=False, sep=";", decimal=".")


def kwh_from_kw_series(s_kw: pd.Series) -> pd.Series:
    # s_kw hat DatetimeIndex und Werte in kW
    if len(s_kw.index) > 1:
        dt_h = (s_kw.index[1] - s_kw.index[0]).total_seconds() / 3600.0
    else:
        dt_h = 1.0
    return s_kw * dt_h


# ----------------------------
# KI-Lastgang Generator
# ----------------------------
def _daily_shape_household(hours: np.ndarray) -> np.ndarray:
    # morgens + abends
    h = hours
    m = np.exp(-0.5 * ((h - 7.5) / 1.6) ** 2)
    e = np.exp(-0.5 * ((h - 19.0) / 2.6) ** 2)
    base = 0.35 + 0.85 * (m + 1.2 * e)
    return base


def _daily_shape_office(hours: np.ndarray) -> np.ndarray:
    # 8-17 hoch
    h = hours
    work = ((h >= 7.0) & (h <= 18.0)).astype(float)
    ramp = 0.25 + 1.15 * work
    # mittags kleiner Dip
    dip = 1.0 - 0.15 * np.exp(-0.5 * ((h - 12.5) / 1.2) ** 2)
    return ramp * dip


def _daily_shape_logistics(hours: np.ndarray, shift: int) -> np.ndarray:
    # Logistik hat oft frühe Peaks, bei 2/3 Schicht deutlich breiter
    h = hours
    peak1 = np.exp(-0.5 * ((h - 6.0) / 2.0) ** 2)
    peak2 = np.exp(-0.5 * ((h - 15.0) / 3.0) ** 2)
    base = 0.45 + 0.9 * (peak1 + 0.7 * peak2)
    if shift == 2:
        base += 0.35 * ((h >= 5) & (h <= 23)).astype(float)
    elif shift == 3:
        base += 0.55 * np.ones_like(h)
    return base


def _daily_shape_machinebuilder(hours: np.ndarray, shift: int) -> np.ndarray:
    # Maschinenbauer: tagsüber hoch, 2/3 Schicht erweitert, Wochenende reduziert
    h = hours
    day = ((h >= 6.0) & (h <= 18.0)).astype(float)
    base = 0.30 + 1.25 * day
    if shift == 2:
        base += 0.35 * ((h >= 5) & (h <= 23)).astype(float)
    elif shift == 3:
        base += 0.55 * np.ones_like(h)
    return base


def _seasonal_factor(dayofyear: np.ndarray, kind: str) -> np.ndarray:
    # kind beeinflusst Winter/Sommer Last
    x = dayofyear / 365.0
    if kind == "logistics":
        return 1.0 + 0.05 * np.cos(2 * np.pi * (x - 0.1))
    if kind == "machine":
        return 1.0 + 0.08 * np.cos(2 * np.pi * (x - 0.1))
    if kind == "office":
        return 1.0 + 0.10 * np.cos(2 * np.pi * (x - 0.1))
    if kind == "agri":
        return 1.0 + 0.18 * np.cos(2 * np.pi * (x - 0.05))
    return 1.0 + 0.12 * np.cos(2 * np.pi * (x - 0.1))


def generate_ai_lastgang(
    index: pd.DatetimeIndex,
    annual_kwh: float,
    segment: str,
    branch: str,
    shift_system: int,
    weekend_reduction: float,
    seed: int,
    agri_phases: list[dict] | None = None,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    n = len(index)
    if n == 0:
        return pd.Series(index=index, data=np.zeros(0))

    hours = index.hour.values + index.minute.values / 60.0
    doy = index.dayofyear.values
    dow = index.dayofweek.values  # 0=Mon ... 6=Sun

    if segment == "Haushalt":
        base = _daily_shape_household(hours)
        season = _seasonal_factor(doy, "house")
    elif segment == "Gewerbe":
        if branch == "Büro/Verwaltung":
            base = _daily_shape_office(hours)
            season = _seasonal_factor(doy, "office")
        elif branch == "Logistik/Spedition":
            base = _daily_shape_logistics(hours, shift_system)
            season = _seasonal_factor(doy, "logistics")
        elif branch == "Maschinenbauer/Produktion":
            base = _daily_shape_machinebuilder(hours, shift_system)
            season = _seasonal_factor(doy, "machine")
        else:
            # "Sonstiges Gewerbe"
            base = 0.40 + 1.0 * ((hours >= 7) & (hours <= 19)).astype(float)
            if shift_system == 2:
                base += 0.25 * ((hours >= 6) & (hours <= 23)).astype(float)
            elif shift_system == 3:
                base += 0.45
            season = _seasonal_factor(doy, "generic")
    else:
        # Landwirtschaft/Mastställe
        # Basis: 24/7, saisonal und phases add-on (z.B. Lüftung/Heizung je Phase)
        base = 0.85 + 0.25 * np.ones_like(hours)
        season = _seasonal_factor(doy, "agri")

        if agri_phases:
            # phases: [{"start":"2026-02-10","weeks":10,"mult":1.2}, ...]
            add = np.zeros(n, dtype=float)
            for ph in agri_phases:
                try:
                    start = pd.to_datetime(ph["start"])
                    weeks = int(ph.get("weeks", 1))
                    mult = float(ph.get("mult", 1.0))
                    end = start + pd.Timedelta(days=7 * weeks)
                    mask = (index >= start) & (index < end)
                    # Phase intensiver: mehr Grundlast + stärker tagsüber
                    add[mask] += (mult - 1.0) * (0.60 + 0.40 * ((hours >= 8) & (hours <= 20)).astype(float))[mask]
                except Exception:
                    pass
            base = base * (1.0 + add)

    # Wochenende Reduktion für Gewerbe (nicht für Landwirtschaft)
    if segment == "Gewerbe":
        is_weekend = (dow >= 5).astype(float)
        base = base * (1.0 - is_weekend * np.clip(weekend_reduction, 0.0, 0.9))

    # Noise
    noise = np.clip(rng.normal(1.0, 0.10, n), 0.7, 1.4)
    raw = base * season * noise
    raw_sum = raw.sum()
    if raw_sum <= 0:
        return pd.Series(index=index, data=np.zeros(n))

    kwh_step = (raw / raw_sum) * max(float(annual_kwh), 0.0)
    return pd.Series(index=index, data=kwh_step)


# ----------------------------
# Page start
# ----------------------------
if "active_slug" not in st.session_state:
    st.error("Kein aktives Projekt. Bitte über Dashboard starten.")
    st.stop()

SLUG = st.session_state["active_slug"]
BASE, DOCS, INPUTS, STATE_FILE = project_paths(SLUG)
d = load_project(SLUG)

if "_last_saved_hash_phase1" not in st.session_state:
    st.session_state["_last_saved_hash_phase1"] = dict_hash(d)

st.title("📊 PHASE 1: Lastganganalyse")
st.caption(f"Projekt: **{SLUG}** – gespeicherte Lastgänge & KI-Modelle werden unter `projects/{SLUG}/documents/inputs/` abgelegt.")

with st.sidebar:
    st.header("💾 Projekt")
    st.write(f"**Slug:** {SLUG}")
    if st.button("Speichern", use_container_width=True):
        save_project(SLUG, d)
        st.session_state["_last_saved_hash_phase1"] = dict_hash(d)
        st.toast("Gespeichert.")

    st.divider()
    st.header("➡️ Navigation")
    if st.button("Phase 2: Planung", use_container_width=True):
        st.switch_page("pages/2_Planung.py")
    if st.button("Phase 3: Bericht", use_container_width=True):
        st.switch_page("pages/3_Bericht.py")


# ----------------------------
# Current Lastgang Status
# ----------------------------
w = d["wirtschaft"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Quelle", w.get("lastgang_source", "—") or "—")
c2.metric("Jahresverbrauch", f"{float(w.get('lastgang_jahr', 0.0)):,.0f} kWh/a")
c3.metric("Auflösung", w.get("lastgang_resolution", "—") or "—")
c4.metric("Format", "kW" if bool(w.get("lastgang_is_power_kw", True)) else "kWh/Step")

if w.get("lastgang_file"):
    st.success(f"Aktiver Lastgang: `{w['lastgang_file']}`")
else:
    st.warning("Noch kein Lastgang gesetzt. Bitte Upload/Musterprofil/KI nutzen.")


# ----------------------------
# Tabs: Upload / Muster / KI
# ----------------------------
tab_upload, tab_muster, tab_ki = st.tabs(["📥 Upload", "📁 Musterprofile", "🤖 KI-Lastgang erzeugen"])

with tab_upload:
    st.subheader("📥 Lastgang Upload (CSV / Excel)")
    st.write("Nach Upload wird die Datei im Projekt zwischengespeichert und in der Planung verwendet.")

    up = st.file_uploader("Datei hochladen", type=["csv", "xlsx", "xls"])
    if up is not None:
        # Speichern
        saved_path = save_uploaded_file(INPUTS, up, prefix="lastgang_")
        st.success(f"Gespeichert: `{saved_path}`")

        # Preview laden
        df = None
        try:
            if up.name.lower().endswith(".csv"):
                sep = st.text_input("CSV Separator", value=w.get("lastgang_sep", ";") or ";")
                dec = st.text_input("Dezimaltrennzeichen", value=w.get("lastgang_decimal", ",") or ",")
                df = _read_csv_bytes(up.getvalue(), sep=sep, decimal=dec)
            else:
                df = pd.read_excel(io.BytesIO(up.getvalue()))
        except Exception as e:
            st.error(f"Fehler beim Lesen: {e}")

        if df is not None and not df.empty:
            st.dataframe(df.head(30), use_container_width=True)

            cols = list(df.columns)
            dt_col = st.selectbox("Datetime-Spalte", cols, index=0 if cols else None)
            val_col = st.selectbox("Wert-Spalte", cols, index=1 if len(cols) > 1 else 0)

            resolution = st.selectbox("Ziel-Auflösung", ["15min", "hour"], index=0 if w.get("lastgang_resolution", "15min") == "15min" else 1)

            val_mode = st.radio("Werte sind …", ["Leistung (kW)", "Energie pro Schritt (kWh)"], index=0, horizontal=True)
            is_power_kw = (val_mode == "Leistung (kW)")

            if st.button("✅ Upload als aktiven Lastgang übernehmen", use_container_width=True):
                # Update state
                w["lastgang_file"] = saved_path
                w["lastgang_datetime_col"] = str(dt_col)
                w["lastgang_value_col"] = str(val_col)
                if up.name.lower().endswith(".csv"):
                    w["lastgang_sep"] = sep
                    w["lastgang_decimal"] = dec
                else:
                    # Excel → wir lesen später via pandas read_csv nicht; daher: in CSV umwandeln
                    # Konvertierung in standardisiertes CSV (timestamp/kwh)
                    try:
                        df2 = _ensure_datetime_index(df, dt_col)
                        s = pd.to_numeric(df2[val_col], errors="coerce").fillna(0.0)
                        s = _resample_series(s, resolution)

                        if is_power_kw:
                            s_kwh = kwh_from_kw_series(s)
                        else:
                            s_kwh = s

                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        out = INPUTS / f"lastgang_excel_norm_{ts}.csv"
                        write_lastgang_csv(out, s_kwh)
                        w["lastgang_file"] = str(out)
                        w["lastgang_datetime_col"] = "timestamp"
                        w["lastgang_value_col"] = "kwh"
                        w["lastgang_sep"] = ";"
                        w["lastgang_decimal"] = "."
                        w["lastgang_is_power_kw"] = False
                        w["lastgang_resolution"] = resolution
                        w["lastgang_source"] = "Upload (Excel→CSV normalisiert)"
                        w["lastgang_ai"] = {}
                        w["lastgang_jahr"] = float(s_kwh.sum())
                        save_project(SLUG, d)
                        st.success("Excel normalisiert & aktiv gesetzt.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Excel-Normalisierung fehlgeschlagen: {e}")
                        st.stop()

                w["lastgang_is_power_kw"] = is_power_kw
                w["lastgang_resolution"] = resolution
                w["lastgang_source"] = "Upload"
                w["lastgang_ai"] = {}

                # Jahresverbrauch berechnen (für Plausibilität + Speicherempfehlung)
                try:
                    if up.name.lower().endswith(".csv"):
                        df2 = _ensure_datetime_index(df, dt_col)
                        s = pd.to_numeric(df2[val_col], errors="coerce").fillna(0.0)
                        s = _resample_series(s, resolution)
                        s_kwh = kwh_from_kw_series(s) if is_power_kw else s
                        w["lastgang_jahr"] = float(s_kwh.sum())
                    # Excel war schon oben behandelt
                except Exception:
                    pass

                save_project(SLUG, d)
                st.success("Aktiver Lastgang gesetzt.")
                st.rerun()

with tab_muster:
    st.subheader("📁 Musterprofile (lokal im Repo)")
    profiles_dir = BASE_DIR / "Lastganganalyse" / "profiles"
    if not profiles_dir.exists():
        profiles_dir = BASE_DIR / "assets" / "profiles"

    if not profiles_dir.exists():
        st.warning("Kein profiles-Ordner gefunden. Lege z.B. `assets/profiles/` oder `Lastganganalyse/profiles/` an.")
    else:
        files = sorted([p for p in profiles_dir.rglob("*") if p.is_file() and p.suffix.lower() in [".csv", ".xlsx", ".xls"]])
        if not files:
            st.info("Keine Profile gefunden.")
        else:
            sel = st.selectbox("Profil wählen", files, format_func=lambda p: str(p.relative_to(profiles_dir)))
            if sel:
                st.write(f"Datei: `{sel}`")
                df = None
                try:
                    if sel.suffix.lower() == ".csv":
                        sep = st.text_input("CSV Separator", value=";")
                        dec = st.text_input("Dezimaltrennzeichen", value=",")
                        df = pd.read_csv(sel, sep=sep, decimal=dec)
                    else:
                        df = pd.read_excel(sel)
                except Exception as e:
                    st.error(f"Lesefehler: {e}")

                if df is not None and not df.empty:
                    st.dataframe(df.head(30), use_container_width=True)
                    cols = list(df.columns)
                    dt_col = st.selectbox("Datetime-Spalte", cols, index=0)
                    val_col = st.selectbox("Wert-Spalte", cols, index=1 if len(cols) > 1 else 0)
                    resolution = st.selectbox("Ziel-Auflösung", ["15min", "hour"], index=0)
                    val_mode = st.radio("Werte sind …", ["Leistung (kW)", "Energie pro Schritt (kWh)"], index=0, horizontal=True)
                    is_power_kw = (val_mode == "Leistung (kW)")

                    if st.button("✅ Musterprofil übernehmen", use_container_width=True):
                        # Profil in Projekt kopieren (damit es dauerhaft/versionssicher ist)
                        content = sel.read_bytes()
                        up_fake = type("UF", (), {"name": sel.name, "getbuffer": lambda self=content: io.BytesIO(content).getbuffer()})()
                        saved_path = save_uploaded_file(INPUTS, up_fake, prefix="lastgang_profile_")

                        try:
                            df2 = _ensure_datetime_index(df, dt_col)
                            s = pd.to_numeric(df2[val_col], errors="coerce").fillna(0.0)
                            s = _resample_series(s, resolution)
                            s_kwh = kwh_from_kw_series(s) if is_power_kw else s
                            # Wir speichern immer normalisiert als timestamp/kwh (für Planung)
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            out = INPUTS / f"lastgang_profile_norm_{ts}.csv"
                            write_lastgang_csv(out, s_kwh)

                            w["lastgang_file"] = str(out)
                            w["lastgang_datetime_col"] = "timestamp"
                            w["lastgang_value_col"] = "kwh"
                            w["lastgang_sep"] = ";"
                            w["lastgang_decimal"] = "."
                            w["lastgang_is_power_kw"] = False
                            w["lastgang_resolution"] = resolution
                            w["lastgang_source"] = f"Musterprofil: {sel.name}"
                            w["lastgang_ai"] = {}
                            w["lastgang_jahr"] = float(s_kwh.sum())

                            save_project(SLUG, d)
                            st.success("Musterprofil normalisiert & aktiv gesetzt.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Übernahme fehlgeschlagen: {e}")

with tab_ki:
    st.subheader("🤖 KI-Lastgang erzeugen (wird als Datei gespeichert und in Planung übernommen)")

    resolution = st.selectbox("Zeitauflösung", ["15min", "hour"], index=0)
    freq = "15T" if resolution == "15min" else "H"

    year = st.number_input("Jahr (für Index)", 2000, 2100, int(datetime.now().year), step=1)
    idx = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq=freq, inclusive="left")

    segment = st.selectbox("Segment", ["Gewerbe", "Haushalt", "Landwirtschaft/Mastställe"], index=0)
    annual_kwh = st.number_input("Jahresverbrauch (kWh/a)", 0.0, 1e9, float(w.get("lastgang_jahr", 5000.0)), step=100.0)

    branch = "—"
    shift_system = 0
    weekend_reduction = 0.5
    agri_phases = []

    if segment == "Gewerbe":
        branch = st.selectbox("Branche", ["Logistik/Spedition", "Maschinenbauer/Produktion", "Büro/Verwaltung", "Sonstiges Gewerbe"], index=0)
        shift_q = st.radio("Schichtsystem?", ["Nein", "2-Schicht", "3-Schicht"], index=0, horizontal=True)
        shift_system = 0 if shift_q == "Nein" else (2 if "2" in shift_q else 3)
        weekend_reduction = st.slider("Wochenend-Reduktion (Sa/So)", 0.0, 0.9, 0.6, 0.05)

    if segment == "Landwirtschaft/Mastställe":
        st.caption("Optional: Phasen (z.B. Einstallen → Wachstum → Endmast) erhöhen temporär die Last.")
        nph = st.number_input("Anzahl Phasen", 0, 8, 3, step=1)
        for i in range(nph):
            c1, c2, c3 = st.columns([1.3, 1, 1])
            start = c1.text_input(f"Phase {i+1} Start (YYYY-MM-DD)", value=f"{year}-02-01", key=f"phs_{i}_s")
            weeks = c2.number_input(f"Phase {i+1} Dauer (Wochen)", 1, 52, 8, key=f"phs_{i}_w")
            mult = c3.number_input(f"Phase {i+1} Intensität (z.B. 1.2)", 1.0, 3.0, 1.2, 0.05, key=f"phs_{i}_m")
            agri_phases.append({"start": start, "weeks": int(weeks), "mult": float(mult)})

    seed = stable_seed(SLUG, segment, branch, shift_system, annual_kwh, resolution, year)
    s_kwh = generate_ai_lastgang(
        index=idx,
        annual_kwh=annual_kwh,
        segment="Haushalt" if segment == "Haushalt" else ("Gewerbe" if segment == "Gewerbe" else "Landwirtschaft"),
        branch=branch,
        shift_system=shift_system,
        weekend_reduction=weekend_reduction,
        seed=seed,
        agri_phases=agri_phases if segment == "Landwirtschaft/Mastställe" else None,
    )

    st.line_chart(s_kwh.tail(7 * (24 if resolution == "hour" else 24 * 4)))
    st.write(f"Summe: **{float(s_kwh.sum()):,.0f} kWh/a**")

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("✅ KI-Lastgang speichern & als aktiv setzen", use_container_width=True):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = INPUTS / f"lastgang_ai_{SLUG}_{ts}.csv"
            write_lastgang_csv(out_csv, s_kwh)

            meta = {
                "created": ts,
                "segment": segment,
                "branch": branch,
                "shift_system": shift_system,
                "weekend_reduction": weekend_reduction,
                "annual_kwh_target": float(annual_kwh),
                "annual_kwh_result": float(s_kwh.sum()),
                "resolution": resolution,
                "year": int(year),
                "seed": int(seed),
                "phases": agri_phases if segment == "Landwirtschaft/Mastställe" else [],
                "file": str(out_csv),
            }
            out_meta = INPUTS / f"lastgang_ai_{SLUG}_{ts}.json"
            out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

            w["lastgang_file"] = str(out_csv)
            w["lastgang_datetime_col"] = "timestamp"
            w["lastgang_value_col"] = "kwh"
            w["lastgang_sep"] = ";"
            w["lastgang_decimal"] = "."
            w["lastgang_is_power_kw"] = False  # normalisiert!
            w["lastgang_resolution"] = resolution
            w["lastgang_source"] = "KI"
            w["lastgang_ai"] = meta
            w["lastgang_jahr"] = float(s_kwh.sum())

            save_project(SLUG, d)
            st.success("KI-Lastgang gespeichert und aktiv gesetzt. Planung nutzt ihn jetzt automatisch.")
            st.rerun()

    with colB:
        if st.button("➡️ Weiter zu Phase 2 (Planung)", use_container_width=True):
            st.switch_page("pages/2_Planung.py")

# Auto-save if changed
h = dict_hash(d)
if st.session_state.get("_last_saved_hash_phase1") != h:
    save_project(SLUG, d)
    st.session_state["_last_saved_hash_phase1"] = h
