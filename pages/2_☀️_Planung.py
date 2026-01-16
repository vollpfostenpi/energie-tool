import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- KONFIGURATION ---
st.set_page_config(page_title="Technische Planung PRO", layout="wide", page_icon="🏗️")

# --- CSS FÜR PROFI-LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .stExpander { background-color: #ffffff; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- NAVIGATION ---
with st.sidebar:
    st.header("Navigation")
    if st.button("🏠 ZURÜCK ZU HOME", use_container_width=True):
        st.session_state["active_project"] = None
        st.switch_page("app.py")
    st.divider()
    st.info("Projekt: " + str(st.session_state.get('project_name', 'Kein Projekt')))

if not st.session_state.get("active_project"):
    st.warning("⚠️ Bitte wählen Sie auf der Home-Seite erst ein Projekt aus.")
    st.stop()

# Referenz auf Session State
data = st.session_state["projekt_daten"]

# --- FUNKTIONEN (CORE LOGIC) ---
def simulate_24h_yield(kwp, orientation):
    """Generiert ein realistisches 24h PV-Profil in 15-Min Intervallen"""
    intervals = 96
    t = np.linspace(0, 24, intervals)
    if orientation == "Süd": peak_time = 13.0
    elif orientation == "Ost/West": 
        # Doppel-Peak Simulation
        profile = 0.4 * np.exp(-((t-10)**2)/2) + 0.4 * np.exp(-((t-16)**2)/2)
        return profile * (kwp / 2)
    else: peak_time = 13.0 # Nord/Andere
    
    profile = np.exp(-((t-peak_time)**2)/4)
    profile = np.where(profile < 0.05, 0, profile)
    return profile * (kwp / 4) # Peak-Leistung Glättung

def get_slp_profile(annual_kwh):
    """Generiert ein H0 Standardlastprofil"""
    t = np.linspace(0, 24, 96)
    base_load = (annual_kwh / 365) / 24
    # Peaks am Morgen und Abend
    profile = base_load * (1 + 0.6 * np.exp(-((t-8)**2)/1.5) + 0.8 * np.exp(-((t-19)**2)/2))
    return profile

# --- HAUPTBEREICH ---
st.title("🏗️ Technische Anlagenplanung & Sektorenkopplung")

# TABS FÜR STRUKTURIERTE EINGABE
tab_load, tab_pv, tab_storage, tab_fin = st.tabs([
    "📊 Lastgang & CSV", "☀️ PV-Generatoren", "🔋 Speicher & Mobilität", "💰 Wirtschaftlichkeit"
])

# --- TAB 1: LASTGANG ---
with tab_load:
    st.header("Elektrischer Lastgang")
    col_l1, col_l2 = st.columns([1, 2])
    
    with col_l1:
        st.subheader("Verbrauchs-Parameter")
        annual_cons = st.number_input("Jahresstromverbrauch (kWh)", 0, 1000000, 15000, step=500)
        load_type = st.selectbox("Profil-Typ", ["Gewerbe (G0)", "Privat (H0)", "CSV-Datei"])
        
        if load_type == "CSV-Datei":
            up_csv = st.file_uploader("15-Min-Lastgang hochladen", type="csv")
            if up_csv:
                # Komplexe CSV-Logik (Zeilen-Simulation)
                df_raw = pd.read_csv(up_csv, sep=None, engine='python')
                st.success(f"Daten mit {len(df_raw)} Zeilen geladen.")
                data['csv_data'] = df_raw
        
    with col_l2:
        # Visualisierung Lastprofil
        t_axis = [f"{int(i//4):02d}:{int((i%4)*15):02d}" for i in range(96)]
        load_vals = get_slp_profile(annual_cons)
        fig_load = px.area(x=t_axis, y=load_vals, title="Tageslastgang (kW)", color_discrete_sequence=['#ff4b4b'])
        st.plotly_chart(fig_load, use_container_width=True)
        

# --- TAB 2: PV-GENERATOREN ---
with tab_pv:
    st.header("Photovoltaik-Flächen")
    
    if 'daecher' not in data: data['daecher'] = []
    
    c_pv1, c_pv2 = st.columns([2, 1])
    
    with c_pv1:
        if st.button("➕ Neue Dachfläche / Tracker hinzufügen"):
            data['daecher'].append({"id": len(data['daecher']), "name": f"Dach {len(data['daecher'])+1}", "kwp": 10.0, "orient": "Süd", "tilt": 35})
            st.rerun()

        for i, d in enumerate(data['daecher']):
            with st.expander(f"⚙️ {d['name']}", expanded=True):
                col_i1, col_i2, col_i3, col_i4 = st.columns([2,2,1,1])
                d['name'] = col_i1.text_input("Name", d['name'], key=f"n_{i}")
                d['kwp'] = col_i2.number_input("Leistung (kWp)", 0.0, 5000.0, float(d['kwp']), key=f"k_{i}")
                d['orient'] = col_i3.selectbox("Ausrichtung", ["Süd", "Süd-West", "Süd-Ost", "Ost/West", "Nord"], key=f"o_{i}")
                if col_i4.button("🗑️", key=f"d_{i}"):
                    data['daecher'].pop(i)
                    st.rerun()

    with c_pv2:
        st.subheader("Hardware & Specs")
        st.text_input("Modulhersteller", "z.B. Jinko Solar")
        st.text_input("Wechselrichter", "z.B. SMA / Fronius")
        st.number_input("Systemverluste (%)", 5, 25, 14)
        
    # Gesamtleistung berechnen
    total_kwp = sum(d['kwp'] for d in data['daecher'])
    data['total_kwp'] = total_kwp
    st.metric("Installierte Gesamtleistung", f"{total_kwp} kWp")

# --- TAB 3: SPEICHER & MOBILITÄT ---
with tab_storage:
    st.header("Sektorenkopplung & Speicher")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.subheader("🔋 Batterietechnik")
        data['bat_kwh'] = st.number_input("Brutto-Kapazität (kWh)", 0.0, 2000.0, 10.0)
        data['bat_dod'] = st.slider("Nutzbare Kapazität (DoD %)", 10, 100, 90)
        data['bat_p'] = st.number_input("Max. Lade-/Entladeleistung (kW)", 0.0, 500.0, 5.0)
        
    with col_s2:
        st.subheader("🚗 Ladeinfrastruktur (GEIG)")
        lp_anzahl = st.number_input("Anzahl Ladepunkte", 0, 100, 1)
        st.info("GEIG-Check: Ab 20 Stellplätzen bei Nicht-Wohngebäuden ist 1 LP Pflicht.")
        
        lp_list = []
        for j in range(lp_anzahl):
            lp_list.append(st.selectbox(f"Ladepunkt {j+1} Leistung", ["11 kW AC", "22 kW AC", "50 kW DC"], key=f"lp_{j}"))
        data['lade_punkte'] = lp_list

# --- TAB 4: WIRTSCHAFTLICHKEIT ---
with tab_fin:
    st.header("Wirtschaftlichkeitsanalyse (20 Jahre)")
    
    c_f1, c_f2 = st.columns(2)
    
    with c_f1:
        price_kwh = st.number_input("Strompreis aktuell (€/kWh)", 0.10, 0.80, 0.35)
        feed_in = st.number_input("Einspeisevergütung (€/kWh)", 0.01, 0.20, 0.08)
        invest_total = st.number_input("Gesamtinvestition (€ netto)", 0, 5000000, int(total_kwp * 1300 + data['bat_kwh'] * 600))
        
    with c_f2:
        st.subheader("Simulations-Parameter")
        degredation = st.slider("PV-Degradation / Jahr (%)", 0.1, 2.0, 0.5)
        elec_increase = st.slider("Strompreissteigerung / Jahr (%)", 0.0, 10.0, 3.0)

    # BERECHNUNG CASHFLOW
    years = np.arange(1, 21)
    # Vereinfachte Simulation
    yield_annual = total_kwp * 950 # kWh/kWp
    autarkie = 40 + (data['bat_kwh'] / (annual_cons/1000) * 5) if annual_cons > 0 else 0
    autarkie = min(autarkie, 85)
    
    savings = (annual_cons * (autarkie/100) * price_kwh) + (yield_annual - (annual_cons*autarkie/100)) * feed_in
    
    cashflow = [-invest_total]
    for y in years:
        annual_yield = yield_annual * (1 - (degredation/100))**y
        current_price = price_kwh * (1 + (elec_increase/100))**y
        # Jährlicher Vorteil
        benefit = (annual_cons * (autarkie/100) * current_price) + (annual_yield - (annual_cons*autarkie/100)) * feed_in
        cashflow.append(cashflow[-1] + benefit)
    
    fig_cf = go.Figure()
    fig_cf.add_trace(go.Bar(x=[0]+list(years), y=cashflow, name="Kumulierter Cashflow", marker_color=['red'] + ['green']*20))
    fig_cf.add_hline(y=0, line_dash="dash", line_color="black")
    fig_cf.update_layout(title="ROI / Amortisationsverlauf (20 Jahre)", xaxis_title="Jahre", yaxis_title="Cashflow (€)")
    st.plotly_chart(fig_cf, use_container_width=True)

# --- FINALES SPEICHERN ---
st.divider()
if st.button("💾 PLANUNGSDATEN FINALISIEREN & SPEICHERN", use_container_width=True):
    # Alle lokalen Variablen in den Session State schreiben
    data['simulation'] = {
        'annual_cons': annual_cons,
        'yield_total': yield_annual,
        'autarkie': autarkie,
        'invest': invest_total,
        'roi_years': invest_total / savings if savings > 0 else 0
    }
    st.session_state["projekt_daten"] = data
    st.success("Planung erfolgreich abgeschlossen! Gehen Sie nun zur Berichts-Seite.")
    st.balloons()
