import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from pathlib import Path
from datetime import datetime, date, timedelta
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Lastganganalyse", page_icon="📊", layout="wide")
st.title("📊 PHASE 1 – Lastganganalyse")

# ============================================================
# PROJECT CONTEXT
# ============================================================
def get_project_context():
    slug = st.session_state.get("active_slug")
    if not slug:
        return None, None, None, None
    base = Path("projects") / slug
    state_file = base / "state.json"
    docs = base / "documents"
    inputs = docs / "inputs"
    lg_dir = inputs / "lastgang"
    lg_dir.mkdir(parents=True, exist_ok=True)
    return slug, base, state_file, lg_dir

def load_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {}
    with state_file.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state_file: Path, data: dict):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ============================================================
# ROBUST CSV READER (ENCODING FIX)
# ============================================================
def read_csv_flexible(content: bytes, sep=None, decimal=None) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            if sep:
                return pd.read_csv(io.BytesIO(content), sep=sep, decimal=decimal, encoding=enc)
            return pd.read_csv(io.BytesIO(content), sep=None, engine="python", decimal=decimal, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def read_profile_any(path_or_file, csv_sep=None, csv_decimal=None) -> pd.DataFrame:
    # path_or_file may be Path/str or UploadedFile
    name = getattr(path_or_file, "name", "")
    if isinstance(path_or_file, (str, Path)):
        p = Path(path_or_file)
        if p.suffix.lower() == ".csv":
            content = p.read_bytes()
            return read_csv_flexible(content, sep=csv_sep, decimal=csv_decimal)
        return pd.read_excel(p)

    # UploadedFile
    content = path_or_file.getvalue()
    name = (name or "").lower()
    if name.endswith(".csv"):
        return read_csv_flexible(content, sep=csv_sep, decimal=csv_decimal)
    return pd.read_excel(io.BytesIO(content))

# ============================================================
# AUTO COLUMN GUESS + NORMALIZE
# ============================================================
def guess_datetime_col(cols):
    candidates = []
    for c in cols:
        cl = str(c).lower()
        score = 0
        if any(k in cl for k in ["time", "datum", "date", "timestamp", "zeit"]):
            score += 3
        if "start" in cl:
            score += 1
        candidates.append((score, c))
    candidates.sort(reverse=True, key=lambda x: x[0])
    if candidates and candidates[0][0] > 0:
        return candidates[0][1]
    return cols[0] if cols else None

def guess_value_col(cols, dt_col):
    candidates = []
    for c in cols:
        if c == dt_col:
            continue
        cl = str(c).lower()
        score = 0
        if any(k in cl for k in ["kw", "leistung", "power"]):
            score += 3
        if any(k in cl for k in ["kwh", "energy"]):
            score += 2
        if any(k in cl for k in ["value", "wert"]):
            score += 1
        candidates.append((score, c))
    candidates.sort(reverse=True, key=lambda x: x[0])
    if candidates:
        return candidates[0][1]
    for c in cols:
        if c != dt_col:
            return c
    return dt_col

def normalize_df(df: pd.DataFrame, dt_col: str, val_col: str, assume_kw=True):
    out = df.copy()
    out = out[[dt_col, val_col]].rename(columns={dt_col: "timestamp", val_col: "value"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0.0)

    if len(out) >= 2:
        dt = (out["timestamp"].iloc[1] - out["timestamp"].iloc[0]).total_seconds()
        step_hours = max(dt / 3600.0, 1e-9)
    else:
        step_hours = 1.0

    if assume_kw:
        out["kw"] = out["value"].astype(float)
        out["kwh_interval"] = out["kw"] * step_hours
    else:
        out["kwh_interval"] = out["value"].astype(float)
        out["kw"] = out["kwh_interval"] / step_hours

    out["date"] = out["timestamp"].dt.date
    out["hour"] = out["timestamp"].dt.hour
    return out[["timestamp", "kw", "kwh_interval", "date", "hour"]], step_hours

# ============================================================
# KI LASTGANG GENERATOR (Gewerbe + Branchen + Schichtsystem)
# ============================================================
def make_index(start_date: date, end_date: date, resolution: str):
    freq = "15min" if resolution == "15min" else "H"
    return pd.date_range(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date) + pd.Timedelta(days=1),
        freq=freq,
        inclusive="left"
    )

def _daily_peaks(hour, peaks):
    out = np.zeros_like(hour, dtype=float)
    for mu, sigma, amp in peaks:
        out += amp * np.exp(-0.5 * ((hour - mu) / sigma) ** 2)
    return out

def _weekday_mask(idx: pd.DatetimeIndex, saturday_active: bool, sunday_active: bool):
    wd = idx.dayofweek.values  # 0=Mon
    mask = (wd <= 4).astype(float)  # Mon-Fri
    if saturday_active:
        mask = np.where(wd == 5, 1.0, mask)
    if sunday_active:
        mask = np.where(wd == 6, 1.0, mask)
    return mask

def ki_profile_gewerbe(idx: pd.DatetimeIndex, seed: int, branche: str, schicht: str,
                       saturday_active: bool, sunday_active: bool, seasonality_level: float = 0.08):
    rng = np.random.default_rng(seed)
    n = len(idx)
    hour = idx.hour.values + idx.minute.values / 60.0
    doy = idx.dayofyear.values

    presets = {
        "Logistik/Lager": {
            "night_floor": 0.70,
            "peaks": [(7.5, 1.3, 0.35), (15.5, 1.6, 0.25)],
            "season_boost": 0.05,
        },
        "Maschinenbauer/Produktion": {
            "night_floor": 0.35,
            "peaks": [(6.5, 1.2, 0.55), (13.0, 1.8, 0.35)],
            "season_boost": 0.08,
        },
        "Lebensmittel/Verarbeitung": {
            "night_floor": 0.55,
            "peaks": [(5.5, 1.4, 0.45), (12.5, 2.0, 0.35), (19.0, 1.8, 0.15)],
            "season_boost": 0.10,
        },
        "Kälte/Frische/Logistik": {
            "night_floor": 0.80,
            "peaks": [(8.0, 1.8, 0.18), (15.0, 2.2, 0.12)],
            "season_boost": 0.18,
        },
        "Werkstatt/Handwerk": {
            "night_floor": 0.15,
            "peaks": [(7.0, 1.1, 0.60), (12.0, 1.8, 0.25)],
            "season_boost": 0.06,
        },
        "Büro/Verwaltung": {
            "night_floor": 0.05,
            "peaks": [(9.5, 1.8, 0.65), (14.0, 1.8, 0.35)],
            "season_boost": 0.04,
        },
    }
    p = presets.get(branche, presets["Maschinenbauer/Produktion"])

    if schicht == "1-Schicht":
        active = ((hour >= 6.0) & (hour < 18.0)).astype(float)
        active = 0.25 + 0.75 * active
    elif schicht == "2-Schicht":
        active = ((hour >= 6.0) & (hour < 22.0)).astype(float)
        active = 0.35 + 0.65 * active
    else:  # 3-Schicht
        active = np.ones(n, dtype=float) * 0.95

    peaks = _daily_peaks(hour, p["peaks"])
    daily = p["night_floor"] + (1.0 - p["night_floor"]) * active + peaks

    wmask = _weekday_mask(idx, saturday_active=saturday_active, sunday_active=sunday_active)
    daily = daily * (0.55 + 0.45 * wmask)

    seasonal = 1.0 + (seasonality_level + p["season_boost"]) * np.cos(2 * np.pi * (doy / 365.0))
    seasonal = np.clip(seasonal, 0.85, 1.25)

    noise = np.clip(rng.normal(1.0, 0.06, n), 0.75, 1.35)

    spikes = np.zeros(n, dtype=float)
    spike_count = int(n * 0.0025)
    if spike_count > 0:
        pos = rng.integers(0, n, size=spike_count)
        spikes[pos] = rng.uniform(0.05, 0.25, size=spike_count)

    shape = (daily * seasonal * noise) * (1.0 + spikes)
    shape = np.clip(shape, 0.02, None)
    return shape

def ki_profile_maststall(idx: pd.DatetimeIndex, seed: int, phases: list[dict], summer_vent_boost: float = 0.35):
    rng = np.random.default_rng(seed + 42)
    n = len(idx)
    hour = idx.hour.values + idx.minute.values / 60.0
    doy = idx.dayofyear.values

    morning = np.exp(-0.5 * ((hour - 6.5) / 1.2) ** 2)
    evening = np.exp(-0.5 * ((hour - 18.0) / 1.8) ** 2)
    daily = 0.90 + 0.25 * morning + 0.18 * evening

    summer = np.exp(-0.5 * ((doy - 200) / 45) ** 2)
    seasonal = 1.0 + summer_vent_boost * summer

    intensity = np.zeros(n) + 0.15
    ts = idx.date
    for ph in phases:
        s = ph["start"]
        e = ph["end"]
        inten = float(ph.get("intensity", 1.0))
        mask = (ts >= s) & (ts < e)
        if mask.any():
            phase_idx = np.where(mask)[0]
            k = len(phase_idx)
            ramp = np.linspace(0.75, 1.05, k)
            intensity[phase_idx] = inten * ramp

    noise = np.clip(rng.normal(1.0, 0.06, n), 0.80, 1.35)
    shape = daily * seasonal * intensity * noise
    shape = np.clip(shape, 0.05, None)
    return shape

def scale_shape_to_target_kwh(idx: pd.DatetimeIndex, shape: np.ndarray, target_kwh: float):
    if len(idx) < 2:
        step_h = 1.0
    else:
        step_h = (idx[1] - idx[0]).total_seconds() / 3600.0
    raw_kwh = float(np.sum(shape) * step_h)
    scale = 0.0 if raw_kwh <= 0 else float(target_kwh) / raw_kwh
    kw = shape * scale
    kwh_interval = kw * step_h
    df = pd.DataFrame({"timestamp": idx, "kw": kw, "kwh_interval": kwh_interval})
    return df, step_h

def build_phases_auto(start: date, end: date, n_phases: int, phase_days: int, pause_days: int):
    phases = []
    cur = start
    for _ in range(n_phases):
        ph_start = cur
        ph_end = min(ph_start + timedelta(days=phase_days), end + timedelta(days=1))
        phases.append({"start": ph_start, "end": ph_end, "intensity": 1.0})
        cur = ph_end + timedelta(days=pause_days)
        if cur > end:
            break
    return phases

# ============================================================
# SIDEBAR – SOURCE SELECTION
# ============================================================
assets_path = Path("assets") / "profiles"

st.sidebar.header("Datenquelle")
source_option = st.sidebar.radio(
    "Quelle wählen",
    ["Standard-Musterprofil", "Eigenes Profil hochladen", "KI-Lastgang generieren"],
    index=0,
)

with st.sidebar.expander("CSV-Optionen (optional)"):
    csv_sep = st.text_input("Separator (leer = auto)", value="")
    csv_decimal = st.text_input("Dezimal (leer = auto)", value="")
    csv_sep = csv_sep if csv_sep.strip() else None
    csv_decimal = csv_decimal if csv_decimal.strip() else None

df_raw = None
selected_file_name = ""
meta_source = {}
step_hours = 1.0

# ============================================================
# A) Musterprofile
# ============================================================
if source_option == "Standard-Musterprofil":
    if assets_path.exists():
        muster_files = sorted([f.name for f in assets_path.iterdir() if f.suffix.lower() in [".csv", ".xlsx", ".xls"]])
        if muster_files:
            selected_muster = st.sidebar.selectbox("Muster auswählen", muster_files, index=0)
            full_path = assets_path / selected_muster
            selected_file_name = selected_muster
            try:
                df_raw = read_profile_any(full_path, csv_sep=csv_sep, csv_decimal=csv_decimal)
                meta_source = {"type": "muster", "file": selected_muster}
            except Exception as e:
                st.sidebar.error(f"Fehler beim Laden: {e}")
        else:
            st.sidebar.warning("Keine Musterdateien in 'assets/profiles' gefunden.")
    else:
        st.sidebar.error("Ordner 'assets/profiles' nicht gefunden.")

# ============================================================
# B) Upload
# ============================================================
elif source_option == "Eigenes Profil hochladen":
    uploaded_file = st.sidebar.file_uploader("Datei wählen (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        selected_file_name = uploaded_file.name
        try:
            df_raw = read_profile_any(uploaded_file, csv_sep=csv_sep, csv_decimal=csv_decimal)
            meta_source = {"type": "upload", "file": uploaded_file.name}
            st.sidebar.success("Datei geladen!")
        except Exception as e:
            st.sidebar.error(f"Fehler beim Upload: {e}")
    else:
        st.info("Bitte lade eine Datei in der Sidebar hoch.")

# ============================================================
# C) KI Generator
# ============================================================
else:
    st.sidebar.subheader("KI-Lastgang Generator")
    resolution = st.sidebar.selectbox("Auflösung", ["15min", "hour"], index=0)
    year = st.sidebar.number_input("Jahr", min_value=2000, max_value=2100, value=2026, step=1)
    start_d = st.sidebar.date_input("Startdatum", value=date(int(year), 1, 1))
    end_d = st.sidebar.date_input("Enddatum", value=date(int(year), 12, 31))
    if end_d < start_d:
        st.sidebar.error("Enddatum muss nach Startdatum liegen.")

    target_kwh = st.sidebar.number_input(
        "Energie im Zeitraum (kWh)",
        min_value=0.0,
        max_value=1e12,
        value=500000.0,
        step=1000.0,
        format="%.0f"
    )

    profiltyp = st.sidebar.selectbox("Profiltyp", ["Gewerbe (KI)", "Landwirtschaft – Maststall"], index=0)
    seed = st.sidebar.number_input("Seed (optional)", min_value=0, max_value=999999, value=12345, step=1)

    branche = "Maschinenbauer/Produktion"
    schicht = "1-Schicht"
    saturday_active = False
    sunday_active = False
    seasonality_level = 0.08

    phases = []
    summer_boost = 0.35

    if profiltyp.startswith("Gewerbe"):
        st.sidebar.markdown("**Gewerbe-Parameter**")
        branche = st.sidebar.selectbox(
            "Branche",
            ["Logistik/Lager", "Maschinenbauer/Produktion", "Lebensmittel/Verarbeitung", "Kälte/Frische/Logistik", "Werkstatt/Handwerk", "Büro/Verwaltung"],
            index=1
        )
        # Schichtsystem inkl. Abfrage
        schicht_yn = st.sidebar.checkbox("Schichtsystem?", value=True)
        if schicht_yn:
            schicht = st.sidebar.selectbox("Schichtsystem", ["1-Schicht", "2-Schicht", "3-Schicht"], index=0)
        else:
            schicht = "1-Schicht"

        cW1, cW2 = st.sidebar.columns(2)
        saturday_active = cW1.checkbox("Samstag aktiv", value=(branche in ["Logistik/Lager", "Kälte/Frische/Logistik"]))
        sunday_active = cW2.checkbox("Sonntag aktiv", value=(branche in ["Kälte/Frische/Logistik"]))

        seasonality_level = st.sidebar.slider("Saison-Einfluss (Heizung/Kälte)", 0.0, 0.30, float(seasonality_level), 0.01)

    else:
        st.sidebar.markdown("**Maststall-Parameter**")
        n_phases = st.sidebar.number_input("Anzahl Einstallungsphasen", min_value=1, max_value=50, value=4, step=1)
        phase_days = st.sidebar.number_input("Dauer je Phase (Tage)", min_value=1, max_value=365, value=80, step=1)
        pause_days = st.sidebar.number_input("Pause/Leerstand (Tage)", min_value=0, max_value=180, value=7, step=1)
        summer_boost = st.sidebar.slider("Sommerlüftung-Boost", 0.0, 1.0, float(summer_boost), 0.05)

        auto_mode = st.sidebar.checkbox("Phasen automatisch verteilen", value=True)
        if auto_mode:
            phases = build_phases_auto(start_d, end_d, int(n_phases), int(phase_days), int(pause_days))
            st.sidebar.caption(f"Erzeugt: {len(phases)} Phasen im Zeitraum")
        else:
            phases = []
            st.sidebar.info("Manuelle Phasen: pro Phase Start/Ende setzen.")
            for i in range(int(n_phases)):
                s_i = st.sidebar.date_input(f"Phase {i+1} Start", value=start_d + timedelta(days=i*(phase_days+pause_days)), key=f"ph_s_{i}")
                e_i = st.sidebar.date_input(f"Phase {i+1} Ende", value=min(s_i + timedelta(days=int(phase_days)), end_d), key=f"ph_e_{i}")
                inten = st.sidebar.slider(f"Phase {i+1} Intensität", 0.2, 1.5, 1.0, 0.05, key=f"ph_int_{i}")
                phases.append({"start": s_i, "end": e_i, "intensity": float(inten)})

    gen_btn = st.sidebar.button("🤖 KI Lastgang erzeugen", use_container_width=True)

    if gen_btn and end_d >= start_d:
        idx = make_index(start_d, end_d, resolution)
        if len(idx) < 2:
            st.error("Zeitraum/Auflösung führt zu leerem Index.")
        else:
            if profiltyp.startswith("Landwirtschaft"):
                shape = ki_profile_maststall(idx, seed=int(seed), phases=phases, summer_vent_boost=float(summer_boost))
                selected_file_name = f"KI_Maststall_{start_d}_{end_d}_{resolution}.csv"
                meta_source = {
                    "type": "ki",
                    "profiltyp": "maststall",
                    "resolution": resolution,
                    "start": str(start_d),
                    "end": str(end_d),
                    "target_kwh": float(target_kwh),
                    "seed": int(seed),
                    "summer_boost": float(summer_boost),
                    "phases": phases,
                }
            else:
                shape = ki_profile_gewerbe(
                    idx,
                    seed=int(seed),
                    branche=str(branche),
                    schicht=str(schicht),
                    saturday_active=bool(saturday_active),
                    sunday_active=bool(sunday_active),
                    seasonality_level=float(seasonality_level),
                )
                selected_file_name = f"KI_Gewerbe_{branche.replace('/','-')}_{schicht}_{start_d}_{end_d}_{resolution}.csv"
                meta_source = {
                    "type": "ki",
                    "profiltyp": "gewerbe",
                    "branche": branche,
                    "schicht": schicht,
                    "resolution": resolution,
                    "start": str(start_d),
                    "end": str(end_d),
                    "target_kwh": float(target_kwh),
                    "seed": int(seed),
                    "saturday_active": bool(saturday_active),
                    "sunday_active": bool(sunday_active),
                    "seasonality_level": float(seasonality_level),
                }

            df_gen, step_hours = scale_shape_to_target_kwh(idx, shape, float(target_kwh))
            df_raw = df_gen.copy()

# ============================================================
# MAIN VIEW
# ============================================================
slug, base, state_file, lg_dir = get_project_context()
if not slug:
    st.warning("Kein aktives Projekt gefunden. Bitte zuerst im Dashboard ein Projekt auswählen/öffnen.")
    st.info("Du kannst Lastgänge trotzdem ansehen – Speichern/Übernehmen in die Planung geht erst mit aktivem Projekt.")

if df_raw is not None:
    st.subheader(f"Aktives Profil: {selected_file_name or '—'}")

    cols = df_raw.columns.tolist()
    # guesses
    if "timestamp" in cols and ("kw" in cols or "kwh_interval" in cols):
        dt_guess = "timestamp"
        val_guess = "kw" if "kw" in cols else "kwh_interval"
    else:
        dt_guess = guess_datetime_col(cols)
        val_guess = guess_value_col(cols, dt_guess)

    col_plot, col_cfg = st.columns([2, 1])

    with col_cfg:
        st.write("### Einstellungen")
        dt_col = st.selectbox("Zeitachse (X)", cols, index=cols.index(dt_guess) if dt_guess in cols else 0)
        y_col = st.selectbox("Wert (Y)", cols, index=cols.index(val_guess) if val_guess in cols else (1 if len(cols) > 1 else 0))
        assume_kw = st.radio("Interpretation Y", ["kW (Leistung)", "kWh/Intervall (Energie)"], index=0, horizontal=True) == "kW (Leistung)"

        try:
            df_norm, step_hours = normalize_df(df_raw, dt_col, y_col, assume_kw=assume_kw)
        except Exception as e:
            st.error(f"Normalisierung fehlgeschlagen: {e}")
            df_norm = None

        if df_norm is not None and not df_norm.empty:
            peak_kw = float(df_norm["kw"].max())
            avg_kw = float(df_norm["kw"].mean())
            energy_kwh = float(df_norm["kwh_interval"].sum())
            st.metric("Spitzenlast", f"{peak_kw:,.2f} kW")
            st.metric("Durchschnitt", f"{avg_kw:,.2f} kW")
            st.metric("Energie im Zeitraum", f"{energy_kwh:,.0f} kWh")

            with st.expander("Erweiterte Kennzahlen"):
                st.write(f"- Schrittweite: **{step_hours:.2f} h**")
                st.write(f"- Zeitraum: **{df_norm['timestamp'].min()}** bis **{df_norm['timestamp'].max()}**")
                st.write(f"- Datenpunkte: **{len(df_norm):,}**")
                st.write(f"- Tagesenergie (Ø): **{energy_kwh / max(df_norm['timestamp'].dt.date.nunique(),1):,.0f} kWh/Tag**")

            # --- Zwischenspeichern (Draft) ---
            if slug and lg_dir and state_file:
                st.divider()
                st.write("### 💾 Zwischenspeichern / In Planung übernehmen")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")

                if st.button("💾 Normalisiertes Profil zwischenspeichern (Draft)", use_container_width=True):
                    out_csv = lg_dir / f"draft_lastgang_norm_{slug}_{ts}.csv"
                    df_norm[["timestamp", "kw", "kwh_interval"]].to_csv(out_csv, index=False, sep=";", decimal=".", encoding="utf-8")

                    if meta_source:
                        out_meta = lg_dir / f"draft_lastgang_meta_{slug}_{ts}.json"
                        out_meta.write_text(json.dumps(meta_source, indent=2, ensure_ascii=False), encoding="utf-8")

                    st.success(f"Gespeichert: {out_csv}")

                if st.button("✅ In Planung übernehmen (state.json setzen)", use_container_width=True):
                    # Write normalized file
                    out_csv = lg_dir / f"lastgang_norm_{slug}_{ts}.csv"
                    df_norm[["timestamp", "kw", "kwh_interval"]].to_csv(out_csv, index=False, sep=";", decimal=".", encoding="utf-8")

                    # write meta
                    if meta_source:
                        out_meta = lg_dir / f"lastgang_meta_{slug}_{ts}.json"
                        out_meta.write_text(json.dumps(meta_source, indent=2, ensure_ascii=False), encoding="utf-8")

                    state = load_state(state_file)
                    state.setdefault("wirtschaft", {})
                    state["wirtschaft"]["lastgang_file"] = str(out_csv)
                    state["wirtschaft"]["lastgang_datetime_col"] = "timestamp"
                    state["wirtschaft"]["lastgang_value_col"] = "kw"
                    state["wirtschaft"]["lastgang_sep"] = ";"
                    state["wirtschaft"]["lastgang_decimal"] = "."
                    state["wirtschaft"]["lastgang_resolution"] = "15min" if step_hours <= 0.26 else "hour"
                    state["wirtschaft"]["lastgang_is_power_kw"] = True
                    state["wirtschaft"]["lastgang_source"] = meta_source

                    # optional: store lastgang KPI
                    state["wirtschaft"]["lastgang_jahr"] = float(df_norm["kwh_interval"].sum())

                    save_state(state_file, state)
                    st.success("✅ Übernommen: Lastgang wurde in state.json gesetzt (Planung nutzt ihn jetzt).")

    with col_plot:
        if df_norm is not None and not df_norm.empty:
            plot_mode = st.radio("Plot", ["kW Verlauf", "kWh/Intervall Verlauf"], horizontal=True)
            y_plot = "kw" if plot_mode == "kW Verlauf" else "kwh_interval"
            fig = px.line(df_norm, x="timestamp", y=y_plot, title="Lastgang Verlauf")
            fig.update_layout(hovermode="x unified", template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Tabellenansicht (normalisiert)"):
                st.dataframe(df_norm.head(500), use_container_width=True)

            csv_bytes = df_norm[["timestamp", "kw", "kwh_interval"]].to_csv(index=False, sep=";", decimal=".").encode("utf-8")
            st.download_button(
                "⬇️ Normalisierte CSV herunterladen (timestamp;kw;kwh_interval)",
                data=csv_bytes,
                file_name=f"lastgang_normalisiert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                use_container_width=True,
            )

else:
    if source_option == "Standard-Musterprofil":
        st.info("Bitte Musterprofil auswählen.")
    elif source_option == "KI-Lastgang generieren":
        st.info("Parameter setzen und **„KI Lastgang erzeugen“** klicken.")
    else:
        st.info("Bitte Datei hochladen.")
