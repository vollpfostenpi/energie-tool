import streamlit as st
import pandas as pd
import os
import plotly.express as px

# Konfiguration der Seite
st.set_page_config(page_title="Lastgang Analyse", page_icon="📊", layout="wide")

st.title("📊 Lastgang Analyse")
st.markdown("""
Analysieren Sie hier Ihren Stromverbrauch. Sie können entweder eigene Daten hochladen 
oder auf vorbereitete Musterprofile aus unserem System zurückgreifen.
""")

# --- NAVIGATION / DATENAUSWAHL ---
st.sidebar.header("Datenquelle")
source_option = st.sidebar.radio(
    "Quelle wählen:",
    ["Eigenes Profil (.csv, .xlsx)", "Musterprofile"]
)

assets_path = "assets"
df = None

if source_option == "Eigenes Profil (.csv, .xlsx)":
    uploaded_file = st.sidebar.file_uploader("Datei hochladen", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
            else:
                df = pd.read_excel(uploaded_file)
            st.sidebar.success("Datei erfolgreich geladen!")
        except Exception as e:
            st.sidebar.error(f"Fehler beim Laden: {e}")

else: # Musterprofile aus assets/
    if os.path.exists(assets_path):
        muster_files = [f for f in os.listdir(assets_path) if f.endswith(('.csv', '.xlsx', '.xls'))]
        if muster_files:
            selected_muster = st.sidebar.selectbox("Muster auswählen:", muster_files)
            full_path = os.path.join(assets_path, selected_muster)
            try:
                if selected_muster.endswith('.csv'):
                    df = pd.read_csv(full_path, sep=None, engine='python')
                else:
                    df = pd.read_excel(full_path)
                st.sidebar.info(f"Musterprofil geladen: {selected
