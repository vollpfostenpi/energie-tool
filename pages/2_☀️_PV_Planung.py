import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="PV Planung", page_icon="☀️", layout="wide")

st.title("☀️ PV-Potenzial & CO2-Analyse")

# Beispiel-Daten erstellen (Diese Logik kannst du durch deine Berechnungen ersetzen)
data = {
    "Jahr": [2024, 2025, 2026, 2027, 2028, 2029, 2030],
    "CO2_Einsparung": [10, 25, 45, 70, 100, 135, 180],
    "Eigenverbrauch": [15, 30, 50, 65, 80, 95, 110]
}
df_co2 = pd.DataFrame(data)

st.subheader("Voraussichtliche CO2-Einsparung (in Tonnen)")

# --- HIER WAR DER FEHLER ---
# Wir nutzen jetzt Plotly statt st.line_chart(df_co2), um den Altair-Bug zu umgehen
try:
    fig = px.line(
        df_co2, 
        x="Jahr", 
        y="CO2_Einsparung",
        markers=True,
        title="CO2-Reduktion über die Jahre"
    )
    fig.update_layout(xaxis_title="Jahr", yaxis_title="Tonnen CO2")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Fehler bei der Grafik-Erstellung: {e}")

# Zusätzliche Info-Boxen
col1, col2 = st.columns(2)
with col1:
    st.metric("Ziel 2030", "180 t", "+12%")
with col2:
    st.info("Diese Analyse basiert auf den Standard-Einstrahlungswerten Ihrer Region.")
