import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Planung", layout="wide")

if not st.session_state.get("active_project"):
    st.warning("⚠️ Bitte wählen Sie auf der Home-Seite ein Projekt aus.")
    st.stop()

data = st.session_state["projekt_daten"]
st.title(f"🏗️ Projektplanung: {st.session_state['project_name']}")

# --- SEKTION 1: LASTGANG ---
st.header("📊 Lastganganalyse")
uploaded_csv = st.file_uploader("Lastgang hochladen (CSV, 15-min Werte)", type=["csv"])
if uploaded_csv:
    try:
        df_lg = pd.read_csv(uploaded_csv)
        st.session_state["projekt_daten"]["lastgang_vorhanden"] = True
        st.line_chart(df_lg.iloc[:96, 1]) # Zeige ersten Tag
        st.success("Lastgang erfolgreich importiert.")
    except Exception as e:
        st.error(f"Fehler beim Lesen der CSV: {e}")

# --- SEKTION 2: PV-FLÄCHEN ---
st.divider()
st.header("☀️ PV-Flächen & Hardware")
col_pv1, col_pv2 = st.columns([2, 1])

with col_pv1:
    if st.button("➕ Neue Fläche"):
        data['daecher'].append({'name': f"Dach {len(data['daecher'])+1}", 'kwp': 0.0, 'ausrichtung': 'Süd'})
    
    for i, dach in enumerate(data['daecher']):
        with st.expander(f"Konfiguration {dach['name']}", expanded=True):
            c1, c2, c3 = st.columns(3)
            dach['name'] = c1.text_input("Name", dach['name'], key=f"n_{i}")
            dach['kwp'] = c2.number_input("kWp", 0.0, 1000.0, dach['kwp'], key=f"k_{i}")
            dach['ausrichtung'] = c3.selectbox("Ausrichtung", ["Süd", "Ost/West", "Nord"], key=f"a_{i}")
            if st.button(f"Fläche {i+1} löschen", key=f"del_{i}"):
                data['daecher'].pop(i)
                st.rerun()

with col_pv2:
    st.subheader("Hardware & Datenblätter")
    st.text_input("Modultyp")
    st.file_uploader("Datenblatt Modul", type=["pdf"], key="pdf_mod")
    st.text_input("Wechselrichter")
    st.file_uploader("Datenblatt WR", type=["pdf"], key="pdf_wr")

# --- SEKTION 3: SPEICHER & LADEN ---
st.divider()
st.header("🔋 Sektorenkopplung")
cb1, cb2 = st.columns(2)
with cb1:
    data['total_kwh'] = st.number_input("Speicherkapazität (kWh)", 0.0, 5000.0, data.get('total_kwh', 0.0))
with cb2:
    anzahl_lp = st.number_input("Anzahl Ladepunkte", 0, 100, len(data['lade_punkte']))
    if anzahl_lp != len(data['lade_punkte']):
        data['lade_punkte'] = [{'id': i} for i in range(anzahl_lp)]

data['total_kwp'] = sum(d['kwp'] for d in data['daecher'])
