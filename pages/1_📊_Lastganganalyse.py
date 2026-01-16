import streamlit as st
import pandas as pd
import os
import plotly.express as px

# 1. Seitenkonfiguration
st.set_page_config(page_title="Lastgang Analyse", page_icon="📊", layout="wide")

st.title("📊 Lastgang Analyse")

# 2. Pfad zu den Musterprofilen
assets_path = os.path.join("assets", "profiles")

# 3. Sidebar: Auswahl & Upload
st.sidebar.header("Daten-Einstellungen")
source_option = st.sidebar.radio(
    "Datenquelle wählen:",
    ["Standard-Musterprofil", "Eigenes Profil hochladen"]
)

df = None
selected_file_name = ""

# --- LOGIK: LADEN DER DATEN ---

# FALL A: Musterprofile (Standard-Start)
if source_option == "Standard-Musterprofil":
    if os.path.exists(assets_path):
        muster_files = [f for f in os.listdir(assets_path) if f.endswith(('.csv', '.xlsx', '.xls'))]
        if muster_files:
            # Dropdown für Muster, aber das erste ist standardmäßig ausgewählt
            selected_muster = st.sidebar.selectbox("Muster auswählen:", muster_files, index=0)
            full_path = os.path.join(assets_path, selected_muster)
            selected_file_name = selected_muster
            
            try:
                if selected_muster.endswith('.csv'):
                    df = pd.read_csv(full_path, sep=None, engine='python')
                else:
                    df = pd.read_excel(full_path)
            except Exception as e:
                st.error(f"Fehler beim Laden des Musters: {e}")
        else:
            st.sidebar.warning("Keine Musterdateien in 'assets/profiles' gefunden.")
    else:
        st.sidebar.error("Ordner 'assets/profiles' nicht gefunden.")

# FALL B: Manueller Upload
else:
    uploaded_file = st.sidebar.file_uploader("Eigene Datei wählen", type=["csv", "xlsx"])
    if uploaded_file:
        selected_file_name = uploaded_file.name
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
            else:
                df = pd.read_excel(uploaded_file)
            st.sidebar.success("Datei geladen!")
        except Exception as e:
            st.sidebar.error(f"Fehler beim Upload: {e}")
    else:
        st.info("Bitte laden Sie eine Datei in der Sidebar hoch.")

# --- ANZEIGE & ANALYSE ---

if df is not None:
    st.subheader(f"Aktives Profil: {selected_file_name}")
    
    # Automatische Spaltenerkennung
    cols = df.columns.tolist()
    
    col1, col2 = st.columns([2, 1])
    with col2:
        st.write("### Einstellungen")
        x_col = st.selectbox("Zeitachse (X)", cols, index=0)
        y_col = st.selectbox("Leistung in kW (Y)", cols, index=1 if len(cols) > 1 else 0)
        
        # Kleine Statistik-Box
        st.metric("Spitzenlast", f"{df[y_col].max():.2f} kW")
        st.metric("Durchschnitt", f"{df[y_col].mean():.2f} kW")

    with col1:
        # Visualisierung
        try:
            fig = px.line(df, x=x_col, y=y_col, title="Lastgang Verlauf")
            fig.update_layout(
                hovermode="x unified",
                template="plotly_white",
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Grafik-Fehler: {e}")

    with st.expander("Tabellenansicht (Rohdaten)"):
        st.dataframe(df.head(100), use_container_width=True)

else:
    if source_option == "Standard-Musterprofil":
        st.warning("Es konnte kein Musterprofil geladen werden. Prüfen Sie Ihren 'assets/profiles' Ordner auf GitHub.")
