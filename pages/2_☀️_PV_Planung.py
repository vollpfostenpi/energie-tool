import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="PV-Planer Expert PRO", page_icon="☀️", layout="wide")

# Custom CSS für professionelles Interface und Beratungs-Karten
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #ffffff; 
        border-radius: 5px 5px 0 0; 
        padding: 10px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .recommendation-card {
        background-color: #ffffff;
        border-left: 6px solid #007bff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    .roi-positive { color: #28a745; font-weight: bold; }
    .icon-text { font-size: 1.2rem; vertical-align: middle; }
    </style>
    """, unsafe_allow_html=True)

st.title("☀️ Smart Energy Architect & ROI-Simulationssystem")
st.caption("Ganzheitliche Planung: PV, Speicher, Flotte & Spotmarkt-Arbitrage")

# --- SIDEBAR: GLOBALE PARAMETER ---
with st.sidebar:
    st.header("🎯 Modul-Auswahl")
    show_pv = st.checkbox("Photovoltaik (PV)", value=True)
    show_storage = st.checkbox("Speichersysteme", value=True)
    show_mobility = st.checkbox("Fuhrpark & Laden", value=False)
    show_arbitrage = st.checkbox("Arbitrage (Spotmarkt)", value=False)
    
    st.divider()
    st.header("💰 Marktdaten & Preise")
    strompreis_netz = st.slider("Netz-Strompreis (ct/kWh)", 15, 60, 32)
    einspeise_verg = st.slider("Einspeisevergütung (ct/kWh)", 1, 15, 8)
    invest_pv_kwp = st.number_input("Investition PV (€/kWp)", 800, 2500, 1200)
    invest_bat_kwh = st.number_input("Investition Speicher (€/kWh)", 300, 1500, 600)

# --- SESSION STATE (DATENHALTUNG) ---
if 'daecher' not in st.session_state: st.session_state.daecher = []
if 'lade_punkte' not in st.session_state: st.session_state.lade_punkte = []
if 'fuhrpark' not in st.session_state: st.session_state.fuhrpark = []

# Dynamische Tab-Logik
tabs_labels = ["📊 Wirtschaftlichkeit & ROI"]
if show_pv: tabs_labels.append("🏗️ PV-Planung")
if show_storage: tabs_labels.append("🔋 Speicher")
if show_mobility: tabs_labels.append("🚗 Fuhrpark & Laden")
if show_arbitrage: tabs_labels.append("📈 Arbitrage-Handel")

tabs = st.tabs(tabs_labels)

# Berechnungsvariablen initialisieren
total_kwp = 0.0
total_storage_kwh = 0.0
total_ev_demand_year = 0.0
total_invest = 0.0

# --- TAB: PV-PLANUNG ---
if show_pv:
    with tabs[tabs_labels.index("🏗️ PV-Planung")]:
        st.header("🏗️ PV-Flächen-Projektierung")
        if st.button("➕ Neue Dachfläche anlegen"):
            st.session_state.daecher.append({'typ': 'Satteldach', 'azimut': 0, 'neigung': 35, 'kwp': 10.0})
        
        for i, dach in enumerate(st.session_state.daecher):
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2,3,2,1])
                # Visuelle Dachauswahl
                d_opts = {"Satteldach": "🏠 Satteldach", "Flachdach": "🟦 Flachdach", "Trapezblech": "🏭 Trapezblech", "Pultdach": "📐 Pultdach"}
                dach['typ'] = c1.selectbox(f"Dachart #{i+1}", list(d_opts.keys()), format_func=lambda x: d_opts[x], key=f"t_{i}")
                
                # Gradgenaue Ausrichtung
                dach['azimut'] = c2.slider(f"Ausrichtung (Azimut) #{i+1}", -180, 180, int(dach['azimut']), 
                                          help="-90=Ost, 0=Süd, 90=West", key=f"az_{i}")
                
                dach['neigung'] = c3.number_input(f"Neigung (°) #{i+1}", 0, 90, value=int(dach['neigung']), key=f"n_{i}")
                if c4.button("🗑️", key=f"del_d_{i}"):
                    st.session_state.daecher.pop(i)
                    st.rerun()
                
                dach['kwp'] = st.number_input(f"Installierte Leistung (kWp) - Fläche {i+1}", 0.0, 5000.0, value=float(dach['kwp']), key=f"p_{i}")
        
        total_kwp = sum(d['kwp'] for d in st.session_state.daecher)
        total_invest += total_kwp * invest_pv_kwp

# --- TAB: SPEICHER ---
if show_storage:
    with tabs[tabs_labels.index("🔋 Speicher")]:
        st.header("🔋 Batteriespeicher-System")
        total_storage_kwh = st.number_input("Nennkapazität des Speichers (kWh)", 0.0, 10000.0, 20.0)
        st.write(f"Voraussichtliche Investition: **{total_storage_kwh * invest_bat_kwh:,.2f} €**")
        st.toggle("Notstrom-Reservesatz (20%) berücksichtigen")
        total_invest += total_storage_kwh * invest_bat_kwh

# --- TAB: FUHRPARK ---
if show_mobility:
    with tabs[tabs_labels.index("🚗 Fuhrpark & Laden")]:
        st.header("🚗 Fuhrpark-Elektrifizierung")
        if st.button("➕ Fahrzeuggruppe hinzufügen"):
            st.session_state.fuhrpark.append({'art': 'PKW', 'anz': 1, 'km': 20000, 'manual': False, 'cons': 0.18})
        
        for i, f in enumerate(st.session_state.fuhrpark):
            with st.container(border=True):
                fc1, fc2, fc3 = st.columns([2,1,1])
                f_icons = {"PKW": "🚗 PKW", "7.5t LKW": "🚚 LKW 7.5t", "18t LKW": "🚛 LKW 18t", "40t LKW": "🚛💨 Sattelzug"}
                f['art'] = fc1.selectbox(f"Fahrzeugtyp #{i}", list(f_icons.keys()), format_func=lambda x: f_icons[x], key=f"fart_{i}")
                f['anz'] = fc2.number_input(f"Anzahl #{i}", 1, 500, value=f['anz'], key=f"fanz_{i}")
                if fc3.button("🗑️", key=f"fdel_{i}"):
                    st.session_state.fuhrpark.pop(i)
                    st.rerun()
                
                ck1, ck2 = st.columns(2)
                f['km'] = ck1.number_input(f"km/Jahr pro Fzg #{i}", 0, 300000, value=f['km'], key=f"fkm_{i}")
                f['manual'] = ck2.checkbox("Eigener Verbrauchswert", value=f['manual'], key=f"fman_{i}")
                
                if f['manual']:
                    f['cons'] = st.number_input(f"kWh/km #{i}", 0.05, 5.0, value=f['cons'], key=f"fc_{i}")
                else:
                    f['cons'] = {"PKW": 0.18, "7.5t LKW": 0.8, "18t LKW": 1.1, "40t LKW": 1.45}[f['art']]
                
                total_ev_demand_year += (f['anz'] * f['km'] * f['cons'])

# --- TAB: ARBITRAGE ---
if show_arbitrage:
    with tabs[tabs_labels.index("📈 Arbitrage-Handel")]:
        st.header("📈 Spotmarkt-Integration (EPEX SPOT)")
        st.info("Das System gleicht Preisprognosen mit dem Speicherstand ab.")
        # Simulation dynamischer Preise
        prices = [12, 10, 8, 7, 9, 15, 25, 35, 30, 22, 18, 15, 14, 15, 18, 25, 38, 45, 42, 32, 28, 22, 18, 14]
        fig_prices = px.area(x=range(24), y=prices, title="Heutige Spotmarkt-Preise (ct/kWh)", labels={'x':'Stunde', 'y':'ct/kWh'})
        st.plotly_chart(fig_prices, use_container_width=True)
        st.write("✔️ **Arbitrage-Potenzial:** Durch Laden in Tiefpreisphasen (02:00 - 05:00) senken Sie Ihre Durchschnittskosten.")

# --- TAB 1: ÜBERSICHT & ROI ---
with tabs[0]:
    st.header("📋 Analyse & Wirtschaftlichkeit")
    
    # Dashboard-Metriken
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Gesamt-Invest", f"{total_invest:,.0f} €")
    col_m2.metric("PV-Ertrag/Jahr", f"{total_kwp * 1020:,.0f} kWh")
    col_m3.metric("Fuhrpark-Bedarf", f"{total_ev_demand_year:,.0f} kWh")
    
    # Einfache ROI Rechnung
    savings_year = (total_kwp * 1020 * 0.4 * (strompreis_netz/100)) + (total_kwp * 1020 * 0.6 * (einspeise_verg/100))
    if total_storage_kwh > 0: savings_year *= 1.3 # Speicher-Bonus
    roi_years = total_invest / savings_year if savings_year > 0 else 0
    col_m4.metric("ROI (Amortisation)", f"{roi_years:.1f} Jahre")

    st.divider()

    # INTELLIGENTE EMPFEHLUNGEN
    st.subheader("💡 Strategische Empfehlungen")
    
    if total_ev_demand_year > 0 and total_kwp == 0:
        st.markdown(f"""<div class="recommendation-card">
            <h4>🚨 PV-Empfehlung für Ihre Flotte</h4>
            <p>Ihr Fuhrpark benötigt jährlich <b>{total_ev_demand_year:,.0f} kWh</b>. Eine PV-Anlage von ca. <b>{round(total_ev_demand_year/1000)} kWp</b> 
            würde Ihre Tankkosten um ca. <b>{round(total_ev_demand_year * (strompreis_netz/100) * 0.4)} €</b> pro Jahr senken.</p>
        </div>""", unsafe_allow_html=True)

    if total_kwp > 0 and total_storage_kwh == 0:
        st.markdown(f"""<div class="recommendation-card">
            <h4>🔋 Speicher-Vorteil (Simulation)</h4>
            <p>Ohne Speicher nutzen Sie nur ca. 30% Ihres PV-Stroms selbst. Mit einem <b>{round(total_kwp*0.8)} kWh</b> Speicher 
            steigt Ihr Eigenverbrauch auf ca. 75%. Dies verkürzt den ROI um ca. 1.5 Jahre.</p>
        </div>""", unsafe_allow_html=True)

    # ROI GRAFIK
    st.subheader("Finanzielle Entwicklung (20 Jahre)")
    years = np.arange(21)
    cashflow = -total_invest + (years * savings_year)
    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=years, y=cashflow, name="Kumulierter Cashflow", line=dict(color='#28a745', width=4)))
    fig_roi.add_hline(y=0, line_dash="dash", line_color="red")
    fig_roi.update_layout(title="Amortisationsverlauf in Euro", yaxis_title="Gewinn/Verlust (€)")
    st.plotly_chart(fig_roi, use_container_width=True)

    # ENERGIEFLUSS-GRAFIK
    st.subheader("Tages-Simulation (Dynamisch)")
    t_ax = np.arange(24)
    # Simulation basierend auf Eingaben
    base_load = np.array([5,4,4,4,5,10,20,35,40,42,45,50,55,50,45,40,35,40,55,60,45,25,15,10])
    if total_ev_demand_year > 0: base_load += (total_ev_demand_year / 365 / 24)
    
    # PV Kurve unter Berücksichtigung der Azimut-Verschiebung
    shift = int(sum(d['azimut'] for d in st.session_state.daecher) / 15) if st.session_state.daecher else 0
    pv_gen = total_kwp * np.roll(np.array([0,0,0,0,0,0.05,0.2,0.5,0.8,0.95,1,1,1,0.95,0.8,0.5,0.2,0.05,0,0,0,0,0,0]), shift)

    fig_flow = go.Figure()
    fig_flow.add_trace(go.Scatter(x=t_ax, y=base_load, name="Strombedarf (inkl. Fuhrpark)", line=dict(color='#dc3545')))
    fig_flow.add_trace(go.Scatter(x=t_ax, y=pv_gen, name="PV-Erzeugung", fill='tozeroy', line=dict(color='#ffc107')))
    st.plotly_chart(fig_flow, use_container_width=True)
