import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- KONFIGURATION ---
st.set_page_config(page_title="Energie-Strategie-Planer", page_icon="☀️", layout="wide")

# Custom Styling für die Beratungs-Boxen
st.markdown("""
    <style>
    .recommendation-card {
        background-color: #f0f7ff;
        border-left: 5px solid #007bff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("☀️ Smart Energy Architect & Strategie-Planer")

# --- 1. ZENTRALE THEMENSTEUERUNG (OPTIONALITÄT) ---
st.sidebar.header("🎯 Relevante Module")
show_pv = st.sidebar.checkbox("Photovoltaik (PV)", value=True)
show_storage = st.sidebar.checkbox("Speichersysteme", value=False)
show_mobility = st.sidebar.checkbox("Ladeinfrastruktur & E-Logistik", value=False)
show_arbitrage = st.sidebar.checkbox("Arbitrage-Handel (Spotmarkt)", value=False)

st.sidebar.divider()
strompreis_netz = st.sidebar.slider("Netz-Strompreis (ct/kWh)", 20, 60, 35)

# --- SESSION STATE INITIALISIERUNG ---
if 'daecher' not in st.session_state: st.session_state.daecher = []
if 'lade_punkte' not in st.session_state: st.session_state.lade_punkte = []
if 'lkw_flotte' not in st.session_state: st.session_state.lkw_flotte = []

# Dynamische Tabs basierend auf Auswahl
tabs_labels = ["📊 Übersicht & Empfehlung"]
if show_pv: tabs_labels.append("🏗️ PV-Planung")
if show_storage: tabs_labels.append("🔋 Speicher")
if show_mobility: tabs_labels.append("🚛 Mobilität & LKW")
if show_arbitrage: tabs_labels.append("📈 Arbitrage")

tabs = st.tabs(tabs_labels)

# Variablen für die Simulation
total_kwp = 0.0
total_storage_kwh = 0.0
total_ev_demand_year = 0.0

# --- TAB: PV-PLANUNG ---
if show_pv:
    with tabs[tabs_labels.index("🏗️ PV-Planung")]:
        st.header("Detailplanung Photovoltaik")
        projekttyp = st.radio("Sektor:", ["Wohnbau", "Industrie/Gewerbe"], horizontal=True)
        adresse = st.text_input("📍 Standort für Wetterdaten-Abgleich", placeholder="Straße, PLZ, Ort")
        
        st.subheader("Dachflächen")
        if st.button("➕ Dachfläche hinzufügen"):
            st.session_state.daecher.append({'id': len(st.session_state.daecher)+1, 'typ': 'Flachdach', 'ausrichtung': 'Süd', 'kwp': 0.0})
        
        for i, dach in enumerate(st.session_state.daecher):
            with st.expander(f"Dach #{i+1} ({dach['typ']})", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                dach['typ'] = c1.selectbox(f"Dachart #{i}", ["Flachdach", "Satteldach", "Trapezblech", "Pultdach"], key=f"typ_{i}")
                dach['ausrichtung'] = c2.selectbox(f"Richtung #{i}", ["Süd", "Ost/West", "Ost", "West", "Nord"], key=f"aus_{i}")
                mode = c3.radio(f"Eingabe #{i}", ["Direkt kWp", "Module"], key=f"mode_{i}")
                if mode == "Direkt kWp":
                    dach['kwp'] = c4.number_input(f"Leistung (kWp) #{i}", 0.0, 5000.0, key=f"p_{i}")
                else:
                    anz = c4.number_input(f"Anzahl Module #{i}", 0, 10000, key=f"anz_{i}")
                    dach['kwp'] = (anz * 430) / 1000
        
        total_kwp = sum(d['kwp'] for d in st.session_state.daecher)
        st.divider()
        st.subheader("Hardware-Komponenten")
        col_h1, col_h2 = st.columns(2)
        col_h1.text_input("PV-Modul Typ")
        col_h1.file_uploader("Datenblatt Modul", type=["pdf"])
        col_h2.text_input("Wechselrichter Typ")
        col_h2.file_uploader("Stringplan / Datenblatt WR", type=["pdf"])

# --- TAB: SPEICHER ---
if show_storage:
    with tabs[tabs_labels.index("🔋 Speicher")]:
        st.header("Energiespeicher-Konfiguration")
        sc1, sc2 = st.columns(2)
        total_storage_kwh = sc1.number_input("Speicherkapazität (kWh)", 0.0, 10000.0, 20.0)
        peak_shaving_active = sc2.toggle("Peak-Shaving zur Lastspitzenkappung")
        if peak_shaving_active:
            st.success("Peak-Shaving reduziert die Netzentgelte durch Glättung der Lastspitzen.")

# --- TAB: MOBILITÄT & LKW ---
if show_mobility:
    with tabs[tabs_labels.index("🚛 Mobilität & LKW")]:
        st.header("Ladeinfrastruktur & E-Logistik")
        
        st.subheader("🔌 Ladepunkte (AC & DC Schnelllader)")
        if st.button("➕ Ladepunkt hinzufügen"):
            st.session_state.lade_punkte.append({'typ': 'AC', 'p': 11.0, 'anz': 1})
        for i, lp in enumerate(st.session_state.lade_punkte):
            lc1, lc2, lc3, lc4 = st.columns([2,1,1,1])
            lp['typ'] = lc1.selectbox(f"Typ #{i}", ["AC Wallbox", "DC Schnelllader", "HPC"], key=f"lpt_{i}")
            lp['p'] = lc2.number_input(f"Leistung (kW) #{i}", 1, 400, key=f"lpp_{i}")
            lp['anz'] = lc3.number_input(f"Anzahl #{i}", 1, 100, key=f"lpa_{i}")
            if lc4.button("🗑️", key=f"lpd_{i}"):
                st.session_state.lade_punkte.pop(i)
                st.rerun()

        st.divider()
        st.subheader("🚛 E-LKW Flotten-Check")
        if st.button("➕ LKW Typ hinzufügen"):
            st.session_state.lkw_flotte.append({'typ': '40t', 'anz': 1, 'km': 80000})
        for i, lkw in enumerate(st.session_state.lkw_flotte):
            with st.container(border=True):
                col_l1, col_l2, col_l3 = st.columns(3)
                lkw['typ'] = col_l1.selectbox(f"Klasse #{i}", ["7.5t", "18t", "40t"], key=f"lkwt_{i}")
                lkw['anz'] = col_l2.number_input(f"Anzahl #{i}", 1, 500, key=f"lkwa_{i}")
                lkw['km'] = col_l3.number_input(f"km pro Jahr #{i}", 0, 300000, key=f"lkwk_{i}")
                
                # Berechnung Benchmarks
                cons_e = {"7.5t": 0.8, "18t": 1.1, "40t": 1.4}[lkw['typ']]
                cons_d = {"7.5t": 18, "18t": 25, "40t": 33}[lkw['typ']]
                bedarf_e = lkw['anz'] * lkw['km'] * cons_e
                bedarf_d = lkw['anz'] * (lkw['km']/100) * cons_d
                total_ev_demand_year += bedarf_e
                st.caption(f"⚡ Bedarf: {bedarf_e:,.0f} kWh/a | ⛽ Diesel: {bedarf_d:,.0f} L/a")

# --- TAB: ARBITRAGE ---
if show_arbitrage:
    with tabs[tabs_labels.index("📈 Arbitrage")]:
        st.header("Arbitrage-Handel & Spotmarkt")
        st.markdown("Nutzen Sie Preisdifferenzen an der Strombörse (EPEX SPOT), um den Speicher bei Tiefpreisen zu füllen.")
        # Simulierter Preisverlauf
        prices = [25, 20, 15, 12, 14, 18, 28, 35, 40, 32, 25, 22, 20, 18, 22, 30, 45, 55, 48, 35, 30, 25, 22, 18]
        fig_p = px.area(x=range(24), y=prices, title="Börsenstrompreis (Simuliert)", labels={'x':'Stunde', 'y':'ct/kWh'})
        st.plotly_chart(fig_p, use_container_width=True)

# --- TAB 1: ÜBERSICHT & BERATUNGS-LOGIK ---
with tabs[0]:
    st.header("📋 Strategische Zusammenfassung")
    
    # KPIs
    m1, m2, m3 = st.columns(3)
    m1.metric("PV-Gesamtleistung", f"{total_kwp:,.1f} kWp")
    m2.metric("E-Mobilitäts Bedarf", f"{total_ev_demand_year:,.0f} kWh/a")
    m3.metric("CO2-Potenzial", f"{(total_kwp * 1050 * 0.385)/1000:.1f} t/a")

    st.divider()
    
    # --- INTELLIGENTE BERATUNGS-LOGIK ---
    st.subheader("💡 Handlungsempfehlungen")
    
    if show_mobility and not show_pv:
        st.markdown(f"""<div class="recommendation-card">
            <h4>🚨 PV-Potenzial ungenutzt!</h4>
            <p>Ihr Mobilitätsbedarf liegt bei <b>{total_ev_demand_year:,.0f} kWh/a</b>. Ohne PV-Anlage beziehen Sie diesen Strom zum vollen Preis. 
            Eine Anlage mit ca. <b>{round(total_ev_demand_year/1000)} kWp</b> würde Ihre Energiekosten massiv senken.</p>
        </div>""", unsafe_allow_html=True)

    if show_pv and show_mobility and not show_storage:
        st.markdown(f"""<div class="recommendation-card">
            <h4>🔋 Speicher-Ergänzung empfohlen</h4>
            <p>Sie planen PV und E-Mobilität. Ohne Speicher können Sie nachts erzeugten Strom nicht nutzen. 
            Ein Speicher von ca. <b>{round(total_kwp * 0.7)} kWh</b> würde Ihre Autarkie deutlich steigern.</p>
        </div>""", unsafe_allow_html=True)

    if show_storage and not show_arbitrage:
        st.markdown(f"""<div class="recommendation-card">
            <h4>📈 Arbitrage-Option prüfen</h4>
            <p>Da Sie einen Speicher planen, könnten Sie zusätzlich vom <b>Arbitrage-Handel</b> profitieren. 
            Laden Sie den Speicher bei negativen oder niedrigen Börsenpreisen direkt aus dem Netz.</p>
        </div>""", unsafe_allow_html=True)

    # Simulation Grafik
    st.subheader("Energiefluss-Simulation (24h)")
    x_ax = np.arange(24)
    y_pv_sim = total_kwp * np.array([0,0,0,0,0,0.1,0.3,0.6,0.8,0.9,1,1,1,0.9,0.8,0.6,0.3,0.1,0,0,0,0,0,0])
    y_load_sim = np.array([10,8,7,7,8,12,25,40,45,42,40,45,50,48,45,40,35,40,50,55,45,30,20,15]) * (1 + total_ev_demand_year/100000)
    
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=x_ax, y=y_load_sim, name="Bedarf", line=dict(color='red', width=3)))
    if show_pv:
        fig_sim.add_trace(go.Scatter(x=x_ax, y=y_pv_sim, name="PV-Erzeugung", fill='tozeroy', line=dict(color='orange')))
    if show_storage:
        y_soc = np.linspace(20, 80, 24) # Dummy SOC
        fig_sim.add_trace(go.Scatter(x=x_ax, y=y_soc, name="Speicher SOC (%)", line=dict(color='green', dash='dot')))
    
    fig_sim.update_layout(hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig_sim, use_container_width=True)
