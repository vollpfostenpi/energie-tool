import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="PV-Planer Expert PRO", page_icon="☀️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .recommendation-card {
        background-color: #ffffff;
        border-left: 6px solid #007bff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("☀️ Smart Energy Architect & ROI-Simulator")

# --- SIDEBAR: GLOBALE PARAMETER ---
with st.sidebar:
    st.header("🎯 Modul-Auswahl")
    show_pv = st.checkbox("Photovoltaik (PV)", value=True)
    show_storage = st.checkbox("Speichersysteme (BESS)", value=True)
    show_mobility = st.checkbox("Ladeinfrastruktur & Fuhrpark", value=False)
    show_arbitrage = st.checkbox("Arbitrage & Peak-Shaving", value=False)
    
    st.divider()
    st.header("💰 Investitions-Modus")
    manual_invest = st.toggle("Investition manuell eingeben", value=False)
    
    if not manual_invest:
        invest_pv_kwp = st.number_input("Investition PV (€/kWp)", 800, 2500, 1100)
        invest_bat_kwh = st.number_input("Investition Speicher (€/kWh)", 200, 1500, 550)
    else:
        total_invest_manual = st.number_input("Gesamt-Projektkosten (€)", 0, 5000000, 50000)

    st.divider()
    strompreis_netz = st.slider("Arbeitspreis Netz (ct/kWh)", 15, 60, 32)
    leistungspreis = 0
    if show_arbitrage:
        leistungspreis = st.number_input("Leistungspreis (€/kW/Jahr) für RLM", 0, 200, 120, help="Relevant für Peak-Shaving Erlöse")
    einspeise_verg = st.slider("Einspeisevergütung (ct/kWh)", 0, 15, 7)

# --- SESSION STATE ---
if 'daecher' not in st.session_state: st.session_state.daecher = []
if 'lade_punkte' not in st.session_state: st.session_state.lade_punkte = []
if 'fuhrpark' not in st.session_state: st.session_state.fuhrpark = []

# Tabs
tabs_labels = ["📊 Wirtschaftlichkeit & ROI"]
if show_pv: tabs_labels.append("🏗️ PV-Planung")
if show_storage: tabs_labels.append("🔋 Speicher (BESS)")
if show_mobility: tabs_labels.append("🚗 Fuhrpark & Laden")
if show_arbitrage: tabs_labels.append("📈 Arbitrage & Markt")

tabs = st.tabs(tabs_labels)

# Globale Variablen initialisieren
total_kwp = 0.0
total_storage_kwh = 0.0
storage_power_kw = 0.0 # Entladeleistung
storage_cycles = 6000 # Standardwert
arbitrage_revenue = 0.0
peak_shaving_revenue = 0.0
total_ev_demand_year = 0.0

# --- TAB: PV-PLANUNG ---
if show_pv:
    with tabs[tabs_labels.index("🏗️ PV-Planung")]:
        st.header("🏗️ PV-Projektierung")
        if st.button("➕ Neues Dach hinzufügen"):
            st.session_state.daecher.append({'typ': 'Satteldach', 'azimut': 0, 'neigung': 35, 'kwp': 10.0})
        
        for i, dach in enumerate(st.session_state.daecher):
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2,3,2,1])
                opts = {"Satteldach": "🏠 Satteldach", "Flachdach": "🟦 Flachdach", "Trapezblech": "🏭 Trapezblech", "Pultdach": "📐 Pultdach"}
                dach['typ'] = c1.selectbox(f"Dachart #{i+1}", list(opts.keys()), format_func=lambda x: opts[x], key=f"t_{i}")
                dach['azimut'] = c2.slider(f"Azimut #{i+1}", -180, 180, int(dach['azimut']), key=f"az_{i}", help="-90=Ost, 0=Süd, 90=West")
                dach['kwp'] = c3.number_input(f"Leistung (kWp) #{i+1}", 0.0, 5000.0, value=float(dach['kwp']), key=f"p_{i}")
                if c4.button("🗑️", key=f"del_d_{i}"):
                    st.session_state.daecher.pop(i); st.rerun()
        
        # Hardware Upload
        with st.expander("📄 Hardware & Datenblätter (PV)"):
            h1, h2 = st.columns(2)
            h1.file_uploader("Datenblatt Module", key="pdf_mod")
            h2.file_uploader("Datenblatt Wechselrichter", key="pdf_wr")
        
        total_kwp = sum(d['kwp'] for d in st.session_state.daecher)

# --- TAB: SPEICHER (ERWEITERT) ---
if show_storage:
    with tabs[tabs_labels.index("🔋 Speicher (BESS)")]:
        st.header("🔋 Batteriespeicher-System (BESS)")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            total_storage_kwh = st.number_input("Nennkapazität (kWh)", 0.0, 10000.0, 50.0)
        
        with col_s2:
            st.info("Für Gewerbespeicher sind C-Rate und Zyklen entscheidend für die Wirtschaftlichkeit.")

        # --- ERWEITERTE TECHNISCHE DATEN ---
        with st.expander("⚙️ Erweiterte Technische Parameter (C-Rate, Leistung, Zyklen)", expanded=True):
            st.caption("Diese Werte beeinflussen die ROI-Berechnung, insbesondere bei Arbitrage und Peak-Shaving.")
            ec1, ec2, ec3 = st.columns(3)
            
            # C-Rate Logik
            c_rate = ec1.number_input("C-Rate (Lade-/Entladefaktor)", 0.1, 5.0, 1.0, step=0.1, help="1C = Speicher in 1h voll/leer. 0.5C = 2h.")
            
            # Automatische Berechnung der Leistung, aber überschreibbar
            calc_power = total_storage_kwh * c_rate
            storage_power_kw = ec2.number_input("Max. Entladeleistung (kW)", 0.0, 5000.0, float(calc_power), help="Wichtig für Lastspitzenkappung")
            
            storage_cycles = ec3.number_input("Zyklenlebensdauer (bei 80% DoD)", 1000, 20000, 6000)
            
            dod = st.slider("Depth of Discharge (DoD) - Nutzbare Kapazität", 50, 100, 90, format="%d%%") / 100
        
        # Hardware Upload
        st.file_uploader("Datenblatt Speicher / BMS", key="pdf_bat")

# --- TAB: FUHRPARK ---
if show_mobility:
    with tabs[tabs_labels.index("🚗 Fuhrpark & Laden")]:
        st.header("🚗 Ladeinfrastruktur")
        col_m1, col_m2 = st.columns(2)
        with col_m1: 
            if st.button("➕ Ladepunkt"): st.session_state.lade_punkte.append({'typ': 'Wallbox', 'p': 11})
            for i, lp in enumerate(st.session_state.lade_punkte):
                with st.container(border=True):
                    l1, l2, l3 = st.columns([2,1,1])
                    lp['typ'] = l1.selectbox(f"Typ #{i}", ["Wallbox (AC)", "DC Charger", "HPC"], key=f"lpt_{i}")
                    lp['p'] = l2.number_input(f"kW #{i}", 1, 400, value=int(lp['p']), key=f"lpp_{i}")
                    if l3.button("🗑️", key=f"lpdel_{i}"): st.session_state.lade_punkte.pop(i); st.rerun()
        
        with col_m2:
            if st.button("➕ Fahrzeug"): st.session_state.fuhrpark.append({'art': 'PKW', 'anz': 1, 'km': 20000, 'cons': 0.18})
            for i, f in enumerate(st.session_state.fuhrpark):
                with st.container(border=True):
                    f['art'] = st.selectbox(f"Art #{i}", ["PKW", "LKW 7.5t", "LKW 40t"], key=f"fa_{i}")
                    f['anz'] = st.number_input(f"Anz #{i}", 1, 100, value=f['anz'], key=f"fanz_{i}")
                    f['km'] = st.number_input(f"km/J #{i}", 1000, 200000, value=f['km'], key=f"fkm_{i}")
                    if st.button("🗑️ Fzg", key=f"fdel_{i}"): st.session_state.fuhrpark.pop(i); st.rerun()
                    f_cons_map = {"PKW": 0.18, "LKW 7.5t": 0.8, "LKW 40t": 1.4}
                    total_ev_demand_year += (f['anz'] * f['km'] * f_cons_map[f['art']])

# --- TAB: ARBITRAGE & MARKT ---
if show_arbitrage:
    with tabs[tabs_labels.index("📈 Arbitrage & Markt")]:
        st.header("📈 Erlösmodelle: Arbitrage & Peak-Shaving")
        
        col_arb1, col_arb2 = st.columns(2)
        
        with col_arb1:
            st.subheader("1. Spotmarkt-Arbitrage")
            st.markdown("Kauf bei Niedrigpreis, Nutzung/Verkauf bei Hochpreis.")
            
            spread = st.slider("Durchschnittl. Preis-Spread (ct/kWh)", 5, 40, 15, help="Preisdifferenz zwischen günstigster und teuerster Stunde am Tag")
            trading_cycles = st.slider("Handelszyklen pro Jahr", 0, 730, 250, help="Wie oft wird der Speicher rein für Handelszwecke voll ge- und entladen?")
            
            # Berechnung Arbitrage
            usable_kwh = total_storage_kwh * dod
            # Formel: Kapazität * Spread * Zyklen * Wirkungsgrad (ca 0.9)
            arbitrage_revenue = (usable_kwh * (spread/100) * trading_cycles * 0.9)
            
            st.metric("Prognostizierter Arbitrage-Erlös", f"{arbitrage_revenue:,.2f} €/Jahr", delta="Cashflow positiv")

        with col_arb2:
            st.subheader("2. Peak-Shaving (Lastspitzen)")
            st.markdown("Reduzierung der Netzentgelte durch Kappen der Jahreshöchstlast.")
            
            if storage_power_kw > 0:
                shaving_potential_kw = st.slider("Gekappte Lastspitze (kW)", 0.0, float(storage_power_kw), float(storage_power_kw*0.5))
                peak_shaving_revenue = shaving_potential_kw * leistungspreis
                st.metric("Ersparnis Netzentgelt", f"{peak_shaving_revenue:,.2f} €/Jahr")
            else:
                st.warning("Bitte definieren Sie zuerst die Entladeleistung im Tab 'Speicher'.")

# --- TAB 1: ÜBERSICHT & ROI ---
with tabs[0]:
    st.header("📊 Business Case & ROI Analyse")
    
    # 1. Investitionskosten ermitteln
    if manual_invest:
        invest_total = total_invest_manual
    else:
        invest_total = (total_kwp * invest_pv_kwp) + (total_storage_kwh * invest_bat_kwh)
        # Ladeinfrastruktur pauschal dazu (Dummy)
        invest_total += len(st.session_state.lade_punkte) * 2000 

    # 2. Ersparnisse / Einnahmen ermitteln
    # PV Ertrag
    pv_yield_kwh = total_kwp * 1000 # ca 1000 kWh/kWp
    
    # Eigenverbrauchsquote schätzen (Simuliert)
    if total_storage_kwh > 0:
        self_consumption_rate = 0.70 # Mit Speicher
        if total_ev_demand_year > 0: self_consumption_rate = 0.85 # Mit E-Auto + Speicher
    else:
        self_consumption_rate = 0.30 # Ohne Speicher
    
    self_used_kwh = min(pv_yield_kwh * self_consumption_rate, pv_yield_kwh) # Kann nicht mehr als Erzeugung sein
    # Wenn EV Bedarf höher als PV Erzeugung, wird alles verbraucht (vereinfacht)
    if total_ev_demand_year > pv_yield_kwh:
        self_used_kwh = pv_yield_kwh 

    fed_in_kwh = pv_yield_kwh - self_used_kwh
    
    # Cashflows
    savings_stromkosten = self_used_kwh * (strompreis_netz / 100)
    income_einspeisung = fed_in_kwh * (einspeise_verg / 100)
    
    total_annual_revenue = savings_stromkosten + income_einspeisung + arbitrage_revenue + peak_shaving_revenue
    
    # ROI
    roi_years = invest_total / total_annual_revenue if total_annual_revenue > 0 else 999

    # --- KPI DISPLAY ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gesamt-Investition (CAPEX)", f"{invest_total:,.0f} €")
    c2.metric("Jährlicher Cashflow", f"{total_annual_revenue:,.0f} €", help="Stromersparnis + Einspeisung + Arbitrage + Peak-Shaving")
    
    roi_color = "normal"
    if roi_years < 7: roi_color = "inverse"
    c3.metric("ROI (Amortisation)", f"{roi_years:.1f} Jahre")
    
    c4.metric("Speicher C-Rate", f"{storage_power_kw/total_storage_kwh if total_storage_kwh>0 else 0:.1f} C", 
              help="Zeigt an, wie leistungsfähig der Speicher im Verhältnis zur Kapazität ist.")

    st.divider()

    # --- ROI CHART ---
    st.subheader("Entwicklung des kumulierten Cashflows")
    
    years = 20
    x_axis = np.arange(years + 1)
    # Cashflow Array erstellen
    cf_cum = [-invest_total]
    for y in range(1, years + 1):
        # Degression PV Module und Speicherleistung berücksichtigen (vereinfacht -0.5% p.a.)
        factor = (1 - 0.005) ** y
        yearly_rev = total_annual_revenue * factor
        # Wartungskosten abziehen (ca 1% vom Invest)
        opex = invest_total * 0.01
        cf_cum.append(cf_cum[-1] + yearly_rev - opex)
    
    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=x_axis, y=cf_cum, fill='tozeroy', 
                                mode='lines+markers', name='Kumulierter Gewinn',
                                line=dict(color='#28a745' if roi_years < 10 else '#ffc107', width=3)))
    
    # Break-Even Linie
    fig_roi.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-Even")
    
    fig_roi.update_layout(
        xaxis_title="Jahre nach Inbetriebnahme",
        yaxis_title="Gewinn / Verlust (€)",
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    # --- TECHNISCHE ZUSAMMENFASSUNG ---
    with st.expander("📝 Detaillierte Zusammensetzung der Einnahmen", expanded=False):
        st.write(f"- **Stromkosteneinsparung (Eigenverbrauch):** {savings_stromkosten:,.2f} €")
        st.write(f"- **Einspeisevergütung:** {income_einspeisung:,.2f} €")
        if show_arbitrage:
            st.write(f"- **Arbitrage-Handel (Spotmarkt):** {arbitrage_revenue:,.2f} €")
            st.write(f"- **Peak-Shaving (Netzentgelt):** {peak_shaving_revenue:,.2f} €")
            if total_storage_kwh > 0 and (storage_power_kw / total_storage_kwh) < 0.5:
                st.warning("⚠️ Achtung: Die gewählte C-Rate ist niedrig (< 0.5). Arbitrage und Peak-Shaving sind physikalisch eventuell nur eingeschränkt möglich!")
