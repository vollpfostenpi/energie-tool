import streamlit as st
import pandas as pd
import plotly.express as px
from core.core import auto_read, try_parse_datetime, clean_numeric, get_official_sources

st.set_page_config(page_title="Lastgang Analyse", layout="wide")

st.title("📊 Lastgang-Analyse & Energiedaten")
st.markdown("""
Analysieren Sie Ihre Verbrauchsdaten aus synPRO oder RLM-Ausläsen. 
Diese Daten bilden die Basis für die PV-Dimensionierung und THG-Flottenplanung.
""")

# Sidebar für Einstellungen
with st.sidebar:
    st.header("⚙️ Import-Optionen")
    up = st.file_uploader("Datei hochladen (.dat, .csv, .xlsx)", type=["dat", "csv", "xlsx"])
    work_price = st.number_input("Arbeitspreis Netz (ct/kWh)", value=28.5, step=0.5)

if up:
    df = auto_read(up)
    
    if not df.empty:
        st.success(f"✅ Datei '{up.name}' erfolgreich geladen.")
        
        # Spalten-Mapping
        cols = df.columns.tolist()
        col_1, col_2 = st.columns(2)
        with col_1:
            time_col = st.selectbox("Zeitstempel-Spalte", cols, index=0)
        with col_2:
            val_col = st.selectbox("Leistungs-Spalte (kW/W)", cols, index=len(cols)-1)
        
        # Daten-Konvertierung
        df[time_col] = df[time_col].apply(try_parse_datetime)
        df = df.dropna(subset=[time_col])
        df[val_col] = df[val_col].apply(clean_numeric)
        
        # Automatische Watt/Kilowatt Erkennung
        if df[val_col].max() > 5000:
            df[val_col] = df[val_col] / 1000.0
            st.info("💡 Werte wurden automatisch von Watt in kW umgerechnet.")

        df = df.set_index(time_col).sort_index()
        series = df[val_col]

        # Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        # Grobe Schätzung des Jahresverbrauchs basierend auf der Durchschnittslast
        annual_energy = series.mean() * 8760 
        
        m1.metric("Spitzenlast", f"{series.max():.2f} kW")
        m2.metric("Grundlast", f"{series.min():.2f} kW")
        m3.metric("Jahresverbrauch (ca.)", f"{annual_energy:,.0f} kWh".replace(",", "."))
        m4.metric("Stromkosten (Netz)", f"{(annual_energy * work_price / 100):,.0f} €".replace(",", "."))

        # Chart
        st.subheader("📈 Lastprofil im Zeitverlauf")
        fig = px.line(df, y=val_col, labels={val_col: "Leistung [kW]", time_col: "Zeitstempel"})
        fig.update_traces(line_color='#0E7C86')
        st.plotly_chart(fig, use_container_width=True)
        
        # Speichern für andere Seiten
        st.session_state['lastgang_data'] = series
        
        st.caption(f"Datenbasis zur Verifizierung: [SMARD Marktdaten]({get_official_sources()['Marktdaten']})")
    else:
        st.error("Format nicht erkannt. Bitte prüfen Sie die Datei.")
else:
    st.info("Bitte laden Sie eine Lastgang-Datei hoch, um mit der Analyse zu beginnen.")
