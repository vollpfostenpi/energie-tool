import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import plotly.express as px
from datetime import datetime, date, timedelta
import io

# ============================================================
# 1) Seitenkonfiguration
# ============================================================
st.set_page_config(page_title="Lastgang Analyse", page_icon="📊", layout="wide")
st.title("📊 Lastgang Analyse")

# ============================================================
# 2) Projekt-Kontext (optional) – für "In Planung übernehmen"
# ============================================================
def get_project_context():
    slug = st.session_state.get("active_slug")
    if not slug:
        return None, None, None
    base = Path("projects") / slug
    state_file = base / "state.json"
    docs = base / "documents"
    inputs = docs / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    return slug, state_file, inputs

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
# 3) Datenquellen / Assets
# ============================================================
assets_path = Path("assets") / "profiles"

# ============================================================
# 4) Helper: Datei lesen + Auto-Spaltenerkennung
# ============================================================
def guess_datetime_col(cols):
    candidates = []
    for c in cols:
        cl = str(c).lower()
        score = 0
        if "time" in cl or "datum" in cl or "date" in cl or "timestamp" in cl or "zeit" in cl:
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
        if "kw" in cl or "leistung" in cl or "power" in cl:
            score += 3
        if "kwh" in cl or "energy" in cl:
            score += 2
        if "value" in cl or "wert" in cl:
            score += 1
        candidates.append((score, c))
    candidates.sort(reverse=True, key=lambda x: x[0])
    if candidates:
        return candidates[0][1]
    # Fallback: zweite Spalte
    for c in cols:
        if c != dt_col:
            return c
    return dt_col

def read_profile_any(path_or_file, csv_sep=None, csv_decimal=None):
    # path_or_file: Path oder UploadedFile
    name = getattr(path_or_file, "name", "")
    if isinstance(path_or_file, (str, Path)):
        name = str(path_or_file)
        p = Path(path_or_file)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p, sep=csv_sep, decimal=csv_decimal) if csv_sep else pd.read_csv(p, sep=None, engine="python", decimal=csv_decimal)
        return pd.read_excel(p)
    else:
        # UploadedFile
        if name.lower().endswith(".csv"):
            if csv_sep:
                return pd.read_csv(path_or_file, sep=csv_sep, decimal=csv_decimal)
            return pd.read_csv(path_or_file, sep=None, engine="python", decimal=csv_decimal)
        return pd.read_excel(path_or_file)

def normalize_df(df: pd.DataFrame, dt_col: str, val_col: str, assume_kw=True):
    out = df.copy()
    out = out[[dt_col, val_col]].rename(columns={dt_col: "timestamp", val_col: "value"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0.0)

    # Frequenz ableiten
    if len(out) >= 2:
        dt = (out["timestamp"].iloc[1] - out["timestamp"].iloc[0]).total_seconds()
        step_hours = max(dt / 3600.0, 1e-9)
    else:
        step_hours = 1.0

    if assume_kw:
        out["kw"] = out["value"].astype(float)
        out["kwh_interval"] = out["kw"] * step_hours
    else:
        # wenn value bereits kWh/Intervall ist
        out["kwh_interval"] = out["value"].astype(float)
        out["kw"] = out["kwh_interval"] / step_hours

    out["date"] = out["timestamp"].dt.date
    out["hour"] = out["timestamp"].dt.hour
    return out, step_hours

# ============================================================
# 5) KI-Lastgang Generator (mit Maststall-Option)
# ============================================================
def make_index(start_date: date, end_date: date, resolution: str):
    freq = "15min" if resolution == "15min" else "H"
    idx = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1), freq=freq, inclusive="left")
    return idx

def ki_profile_general(idx: pd.DatetimeIndex, seed: int):
    rng = np.random.default_rng(seed)
    n = len(idx)
    hour = idx.hour.values + idx.minute.values / 60.0
    doy = idx.dayofyear.values

    # Tagesgang (morgens/abends leicht höher)
    daily = 0.95 + 0.15 * np.cos(2 * np.pi * (hour - 19) / 24.0)
    # Saisonal (Winter minimal höher)
    seasonal = 1.00 + 0.08 * np.cos(2 * np.pi * (doy / 365.0))
    # Rauschen
    noise = np.clip(rng.normal(1.0, 0.05, n), 0.85, 1.25)
    shape = daily * seasonal * noise
    return shape

def ki_profile_maststall(idx: pd.DatetimeIndex, seed: int, phases: list[dict], summer_vent_boost: float = 0.35):
    """
    phases: list of dict {start: date, end: date, intensity: float}
    intensity ~ Besatz/Phasefaktor (z.B. 0.2..1.0)
    """
    rng = np.random.default_rng(seed + 42)
    n = len(idx)
    hour = idx.hour.values + idx.minute.values / 60.0
    doy = idx.dayofyear.values

    # Baseline: Stall hat Grundlast (Lüftung, Steuerung)
    base = np.ones(n) * 1.0

    # Tagesgang: Fütterung/Technik Peaks morgens und abends
    morning = np.exp(-0.5 * ((hour - 6.5) / 1.2) ** 2)
    evening = np.exp(-0.5 * ((hour - 18.0) / 1.8) ** 2)
    daily = 0.90 + 0.25 * morning + 0.18 * evening

    # Sommerlüftung (Juni–Aug) höher
    # peak um Tag 200
    summer = np.exp(-0.5 * ((doy - 200) / 45) ** 2)
    seasonal = 1.0 + summer_vent_boost * summer

    # Phasenintensität über Zeit
    intensity = np.zeros(n) + 0.15  # "Leerstand/Minimalbetrieb"
    ts = idx.date

    for ph in phases:
        s = ph["start"]
        e = ph["end"]
        inten = float(ph.get("intensity", 1.0))
        mask = (ts >= s) & (ts < e)
        if mask.any():
            # innerhalb der Phase leicht ansteigend (Tierwachstum)
            # linear von 0.75..1.05 des intensity
            phase_idx = np.where(mask)[0]
            k = len(phase_idx)
            ramp = np.linspace(0.75, 1.05, k)
            intensity[phase_idx] = inten * ramp

    noise = np.clip(rng.normal(1.0, 0.06, n), 0.80, 1.35)
    shape = base * daily * seasonal * intensity * noise

    # immer positive Werte
    shape = np.clip(shape, 0.05, None)
    return shape

def scale_to_annual_kwh(idx: pd.DatetimeIndex, shape: np.ndarray, target_kwh: float):
    if len(idx) < 2:
        step_h = 1.0
    else:
        step_h = (idx[1] - idx[0]).total_seconds() / 3600.0

    # shape entspricht "kW relativ" -> skaliere so, dass Jahresenergie passt
    raw_kwh = float(np.sum(shape) * step_h)
    if raw_kwh <= 0:
        scale = 0.0
    else:
        scale = float(target_kwh) / raw_kwh

    kw = shape * scale
    kwh_interval = kw * step_h
    df = pd.DataFrame({"timestamp": idx, "kw": kw, "kwh_interval": kwh_interval})
    return df, step_h

def build_phases_auto(start: date, end: date, n_phases: int, phase_days: int, pause_days: int):
    phases = []
    cur = start
    for i in range(n_phases):
        ph_start = cur
        ph_end = min(ph_start + timedelta(days=phase_days), end + timedelta(days=1))
        phases.append({"start": ph_start, "end": ph_end, "intensity": 1.0})
        cur = ph_end + timedelta(days=pause_days)
        if cur > end:
            break
    return phases

# ============================================================
# 6) Sidebar: Auswahl & Upload & KI
# ============================================================
st.sidebar.header("Daten-Einstellungen")
source_option = st.sidebar.radio(
    "Datenquelle wählen:",
    ["Standard-Musterprofil", "Eigenes Profil hochladen", "KI-Lastgang generieren"],
    index=0,
)

df_raw = None
df_norm = None
selected_file_name = ""
step_hours = 1.0

# Erweiterte CSV Optionen
with st.sidebar.expander("CSV-Optionen (optional)"):
    csv_sep = st.text_input("Separator (leer = auto)", value="")
    csv_decimal = st.text_input("Dezimal (leer = auto)", value="")
    csv_sep = csv_sep if csv_sep.strip() else None
    csv_decimal = csv_decimal if csv_decimal.strip() else None

# -------------------------
# A) Musterprofile
# -------------------------
if source_option == "Standard-Musterprofil":
    if assets_path.exists():
        muster_files = [f.name for f in assets_path.iterdir() if f.suffix.lower() in [".csv", ".xlsx", ".xls"]]
        muster_files = sorted(muster_files)
        if muster_files:
            selected_muster = st.sidebar.selectbox("Muster auswählen:", muster_files, index=0)
            full_path = assets_path / selected_muster
            selected_file_name = selected_muster
            try:
                df_raw = read_profile_any(full_path, csv_sep=csv_sep, csv_decimal=csv_decimal)
            except Exception as e:
                st.error(f"Fehler beim Laden des Musters: {e}")
        else:
            st.sidebar.warning("Keine Musterdateien in 'assets/profiles' gefunden.")
    else:
        st.sidebar.error("Ordner 'assets/profiles' nicht gefunden.")

# -------------------------
# B) Upload
# -------------------------
elif source_option == "Eigenes Profil hochladen":
    uploaded_file = st.sidebar.file_uploader("Eigene Datei wählen", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        selected_file_name = uploaded_file.name
        try:
            df_raw = read_profile_any(uploaded_file, csv_sep=csv_sep, csv_decimal=csv_decimal)
            st.sidebar.success("Datei geladen!")
        except Exception as e:
            st.sidebar.error(f"Fehler beim Upload: {e}")
    else:
        st.info("Bitte laden Sie eine Datei in der Sidebar hoch.")

# -------------------------
# C) KI Generator
# -------------------------
else:
    st.sidebar.subheader("KI-Lastgang Generator")

    resolution = st.sidebar.selectbox("Auflösung", ["15min", "hour"], index=0)
    year = st.sidebar.number_input("Jahr", min_value=2000, max_value=2100, value=2026, step=1)

    # Zeitraum
    start_d = st.sidebar.date_input("Startdatum", value=date(int(year), 1, 1))
    end_d = st.sidebar.date_input("Enddatum", value=date(int(year), 12, 31))
    if end_d < start_d:
        st.sidebar.error("Enddatum muss nach Startdatum liegen.")

    target_kwh = st.sidebar.number_input("Jahresmenge (kWh) im Zeitraum", min_value=0.0, max_value=1e12, value=500000.0, step=1000.0, format="%.0f")

    profiltyp = st.sidebar.selectbox("Profiltyp", ["Allgemein (Gewerbe/Standard)", "Landwirtschaft – Maststall"], index=0)

    # Maststall-Inputs
    phases = []
    summer_boost = 0.35
    if profiltyp.startswith("Landwirtschaft"):
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
            st.sidebar.info("Manuelle Phasen: pro Phase Start/Ende setzen.")
            phases = []
            for i in range(int(n_phases)):
                s_i = st.sidebar.date_input(f"Phase {i+1} Start", value=start_d + timedelta(days=i*(phase_days+pause_days)), key=f"ph_s_{i}")
                e_i = st.sidebar.date_input(f"Phase {i+1} Ende", value=min(s_i + timedelta(days=int(phase_days)), end_d), key=f"ph_e_{i}")
                inten = st.sidebar.slider(f"Phase {i+1} Intensität", 0.2, 1.5, 1.0, 0.05, key=f"ph_int_{i}")
                phases.append({"start": s_i, "end": e_i, "intensity": float(inten)})

    seed = st.sidebar.number_input("Seed (optional)", min_value=0, max_value=999999, value=12345, step=1)
    gen_btn = st.sidebar.button("🤖 KI Lastgang erzeugen", use_container_width=True)

    if gen_btn and end_d >= start_d:
        idx = make_index(start_d, end_d, resolution)
        if len(idx) < 2:
            st.error("Zeitraum/Auflösung führt zu leerem Index.")
        else:
            if profiltyp.startswith("Landwirtschaft"):
                shape = ki_profile_maststall(idx, seed=int(seed), phases=phases, summer_vent_boost=float(summer_boost))
                selected_file_name = f"KI_Maststall_{start_d}_{end_d}_{resolution}.csv"
            else:
                shape = ki_profile_general(idx, seed=int(seed))
                selected_file_name = f"KI_Standard_{start_d}_{end_d}_{resolution}.csv"

            df_gen, step_hours = scale_to_annual_kwh(idx, shape, float(target_kwh))
            df_raw = df_gen.copy()

# ============================================================
# 7) Anzeige & Analyse
# ============================================================
if df_raw is not None:
    st.subheader(f"Aktives Profil: {selected_file_name or '—'}")

    cols = df_raw.columns.tolist()

    # Auto-Erkennung (nur wenn nicht KI-DF mit timestamp/kw)
    if "timestamp" in cols and ("kw" in cols or "kwh_interval" in cols):
        dt_guess = "timestamp"
        val_guess = "kw" if "kw" in cols else "kwh_interval"
    else:
        dt_guess = guess_datetime_col(cols)
        val_guess = guess_value_col(cols, dt_guess)

    col1, col2 = st.columns([2, 1])

    with col2:
        st.write("### Einstellungen")
        dt_col = st.selectbox("Zeitachse (X)", cols, index=cols.index(dt_guess) if dt_guess in cols else 0)
        y_col = st.selectbox("Wert (Y)", cols, index=cols.index(val_guess) if val_guess in cols else (1 if len(cols) > 1 else 0))

        assume_kw = st.radio("Interpretation Y", ["kW (Leistung)", "kWh/Intervall (Energie)"], index=0, horizontal=True) == "kW (Leistung)"

        # Normalisieren
        try:
            df_norm, step_hours = normalize_df(df_raw, dt_col, y_col, assume_kw=assume_kw)
        except Exception as e:
            st.error(f"Normalisierung fehlgeschlagen: {e}")
            df_norm = None

        if df_norm is not None and not df_norm.empty:
            peak_kw = float(df_norm["kw"].max())
            avg_kw = float(df_norm["kw"].mean())
            annual_kwh = float(df_norm["kwh_interval"].sum())

            st.metric("Spitzenlast", f"{peak_kw:,.2f} kW")
            st.metric("Durchschnitt", f"{avg_kw:,.2f} kW")
            st.metric("Energie im Zeitraum", f"{annual_kwh:,.0f} kWh")

            # Quick plausibility
            if peak_kw <= 0:
                st.warning("⚠️ Spitzenlast ist 0 – prüfen, ob Spalten korrekt gewählt sind.")
            if annual_kwh <= 0:
                st.warning("⚠️ Energie ist 0 – prüfen, ob Spalten korrekt gewählt sind.")

            with st.expander("Erweiterte Kennzahlen"):
                st.write(f"- Schrittweite: **{step_hours:.2f} h**")
                st.write(f"- Zeitraum: **{df_norm['timestamp'].min()}** bis **{df_norm['timestamp'].max()}**")
                st.write(f"- Datenpunkte: **{len(df_norm):,}**")
                st.write(f"- Tagesenergie (Ø): **{annual_kwh / max((df_norm['timestamp'].dt.date.nunique()),1):,.0f} kWh/Tag**")

    with col1:
        if df_norm is not None and not df_norm.empty:
            # Plot wählen: kW oder kWh/Intervall
            plot_mode = st.radio("Plot", ["kW Verlauf", "kWh/Intervall Verlauf"], horizontal=True)
            y_plot = "kw" if plot_mode == "kW Verlauf" else "kwh_interval"

            try:
                fig = px.line(df_norm, x="timestamp", y=y_plot, title="Lastgang Verlauf")
                fig.update_layout(
                    hovermode="x unified",
                    template="plotly_white",
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Grafik-Fehler: {e}")

            with st.expander("Tabellenansicht (normalisiert)"):
                st.dataframe(df_norm.head(200), use_container_width=True)

            # Download normalisiert
            csv_bytes = df_norm[["timestamp", "kw", "kwh_interval"]].to_csv(index=False, sep=";", decimal=".").encode("utf-8")
            st.download_button(
                "⬇️ Normalisierte CSV herunterladen (timestamp;kw;kwh_interval)",
                data=csv_bytes,
                file_name=f"lastgang_normalisiert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                use_container_width=True,
            )

            # In Projekt übernehmen (optional)
            slug, state_file, inputs_dir = get_project_context()
            if slug and state_file and inputs_dir:
                st.divider()
                st.subheader("➡️ In Planung übernehmen (Projektstate.json aktualisieren)")
                if st.button("✅ Profil in Projekt speichern & für Planung setzen", use_container_width=True):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = inputs_dir / f"lastgang_{slug}_{ts}.csv"
                    # wir speichern bewusst kW als value (Planung kann in kWh/step umrechnen)
                    df_norm[["timestamp", "kw"]].to_csv(out_path, index=False, sep=";", decimal=".")

                    state = load_state(state_file)
                    state.setdefault("wirtschaft", {})
                    state["wirtschaft"]["lastgang_file"] = str(out_path)
                    state["wirtschaft"]["lastgang_datetime_col"] = "timestamp"
                    state["wirtschaft"]["lastgang_value_col"] = "kw"
                    state["wirtschaft"]["lastgang_sep"] = ";"
                    state["wirtschaft"]["lastgang_decimal"] = "."
                    # Auflösung ableiten
                    state["wirtschaft"]["lastgang_resolution"] = "15min" if step_hours <= 0.26 else "hour"
                    state["wirtschaft"]["lastgang_is_power_kw"] = True

                    save_state(state_file, state)
                    st.success(f"Gespeichert: {out_path}")
                    st.caption("Die Planung-Seite kann jetzt dieses Profil direkt nutzen (aus state.json).")
            else:
                st.caption("Hinweis: Kein aktives Projekt erkannt (active_slug fehlt) – Projekt-Übernahme deaktiviert.")

else:
    if source_option == "Standard-Musterprofil":
        st.warning("Es konnte kein Musterprofil geladen werden. Prüfe den Ordner 'assets/profiles'.")
    elif source_option == "KI-Lastgang generieren":
        st.info("Stelle Parameter in der Sidebar ein und klicke **„KI Lastgang erzeugen“**.")
    else:
        st.info("Bitte lade eine Datei in der Sidebar hoch.")
