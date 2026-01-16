import streamlit as st
import pandas as pd
import os
import plotly.express as px

# 1. Seitenkonfiguration
st.set_page_config(page_title="Lastgang Analyse", page_icon="📊", layout="wide")

st.title("📊 Lastgang Analyse")

# 2. Pfad zu den Musterprofilen definieren
assets_path = "assets"

# 3. Auswahl der Datenquelle in der Sidebar
st.sidebar.header("Datenquelle")
source_option = st.sidebar.radio(
    "Wie möchten Sie Daten laden?",
    ["Eigenes Profil hochladen", "Musterprofil auswählen"]
)

df = None

# --- LOGIK: EIGENER UPLOAD ---
if source_option == "Eigenes Profil hochladen":
    uploaded_file = st.sidebar.file_uploader("Datei wählen (.csv, .xlsx)", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                # sep=None mit engine='python' erkennt automatisch ; oder ,
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
            else:
                df = pd.read_excel(uploaded_file)
            st.sidebar.success("Datei erfolgreich geladen!")
        except Exception as e:
            st.sidebar.error(f"Fehler beim Laden: {e}")

# --- LOGIK: MUSTERPROFILE ---
else:
    if os.path.exists(assets_path):
        # Liste alle Excel und CSV Dateien im assets Ordner
        muster_files = [f for f in os.listdir(assets_path) if f.endswith(('.csv', '.xlsx', '.xls'))]
        
        if muster_files:
            selected_muster = st.sidebar.selectbox("Muster auswählen:", muster_files)
            full_path = os.path.join(assets_path, selected_muster)
            try:
                if selected_muster.endswith('.csv'):
                    df = pd.read_csv(full_path, sep=None, engine='python')
                else:
                    df = pd.read_excel(full_path)
                st.sidebar.info(f"Profil geladen: {selected_muster}")
            except Exception as e:
                st.sidebar.error(f"Fehler beim Laden des Musters: {e}")
        else:
            st.sidebar.warning("Ordner 'assets' ist leer.")
    else:
        st.sidebar.error("Ordner 'assets' nicht gefunden. Bitte auf GitHub erstellen.")

# --- DATENANZEIGE & VISUALISIERUNG ---
if df is not None:
    # Vorschau der Daten
    with st.expander("Vorschau der Rohdaten"):
        st.dataframe(df.head(10), use_container_width=True)

    # Spaltenauswahl für die Grafik
    cols = df.columns.tolist()
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("Zeitachse (X-Achse)", cols, index=0)
    with col2:
        # Falls eine zweite Spalte existiert, diese wählen, sonst die erste
        y_col = st.selectbox("Leistung in kW (Y-Achse)", cols, index=1 if len(cols) > 1 else 0)

    # Grafik erstellen mit Plotly (vermeidet Altair-Probleme in Python 3.13)
    st.subheader("Visualisierung des Lastgangs")
    try:
        fig = px.line(df, x=x_col, y=y_col, title=f"Lastgang: {y_col}")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Kleine Statistik-Kacheln
        m1, m2, m3 = st.columns(3)
        m1.metric("Max. Last", f"{df[y_col].max():.2f} kW")
        m2.metric("Durchschnitt", f"{
