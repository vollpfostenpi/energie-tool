import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="PV-Expert PRO 2026", page_icon="☀️", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 12px; border: 1px solid #dee2e6; }
    .stAlert { border-radius: 12px; }
    .hardware-box { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("☀️ Smart Energy Architect & Business Simulator")

# --- SIDEBAR: GLOBALE PARAMETER ---
with st.sidebar:
    st.header("🔌 Netz & Infrastruktur")
    grid_limit_kva = st.number_input("Max. Netzanschlussleistung (kVA)", 10, 10000, 43)
    
    st.divider()
    st.header("🎯 Modul-Auswahl")
    show_pv = st.checkbox("Photovoltaik (PV)", value=True)
    show_storage = st.checkbox("Batteriespeicher (BESS)", value=True)
    show_mobility = st.checkbox("Fuhrpark & Laden", value=True)
    show_heat = st.checkbox("Sektorenkopplung Wärme", value=False)
    show_arbitrage = st.checkbox("Arbitrage Handel", value=False)
    
    st.divider()
    st.header("💰 Marktdaten")
    strompreis_netz = st.slider("Bezugspreis (ct/kWh)", 15, 60, 32)
    einspeise_verg_ueberschuss = st.number_input("EEG Überschuss (ct/kWh)", 0.0, 15.0, 8.2)
    einspeise_verg_voll = st.number_input("EEG Volleinspeisung (ct/kWh)", 0.0, 20.0, 13.0)

# --- SESSION STATE INITIALISIERUNG ---
for key in ['daecher', 'pv_module', 'pv_wr', 'bat_systeme', 'lade_punkte', 'fuhrpark']:
    if key not in st.session_state: st.session_state[key] = []

# Tabs Setup
tabs_labels = ["📊 ROI & Netz-Check"]
if show_pv: tabs_labels.append("🏗️ PV-Planung")
if show_storage: tabs_labels.append("🔋 Speicher")
if show_mobility: tabs_labels.append("🚗 Mobilität & Laden")
if show_arbitrage: tabs_labels.append("📈 Arbitrage")
tabs = st.tabs(tabs_labels)

# Globale Berechnungsvariablen
total_kwp = 0.0
total_storage_kwh = 0.0
total_lp_power_ac = 0.0
total_lp_power_dc = 0.0
thg_revenue = 0.0
pv_betriebsmodus = "Eigenverbrauch"

# --- TAB: PV-PLANUNG ---
if show_pv:
    with tabs[tabs_labels.index("🏗️ PV-Planung")]:
        st.header("🏗️ PV-Projektierung")
        
        # Logik: Nur PV Prüfung
        only_pv = show_pv and not (show_storage or show_mobility or show_heat or show_arbitrage)
        if only_pv:
            st.info("💡 Da nur PV geplant wird: Wählen Sie das Betriebskonzept.")
            pv_betriebsmodus = st.radio("Betriebsmodus", ["Eigenverbrauch (Überschusseinspeisung)", "Volleinspeisung"], horizontal=True)
        
        # Dachflächen
        st.subheader("🏠 Dachflächen")
        if st.button("➕ Dachfläche hinzufügen"):
            st.session_state.daecher.append({'kwp': 10.0})
        for i, d in enumerate(st.session_state.daecher):
            c1, c2 = st.columns([4,1])
            d['kwp'] = c1.number_input(f"Leistung Dach {i+1} (kWp)", 0.0, 5000.0, d['kwp'], key=f"dkwp_{i}")
            if c2.button("🗑️", key=f"ddel_{i}"): st.session_state.daecher.pop(i); st.rerun()
        total_kwp = sum(d['kwp'] for d in st.session_state.daecher)

        # Hardware: Module & Wechselrichter
        st.divider()
        st.subheader("🧩 PV-Komponenten")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Module**")
            if st.button("➕ Modul-Typ hinzufügen"): st.session_state.pv_module.append({'typ': '', 'anz': 0, 'w': 440})
            for i, m in enumerate(st.session_state.pv_module):
                with st.container(border=True):
                    m['typ'] = st.text_input(f"Modul-Hersteller/Typ #{i+1}", m['typ'], key=f"m_t_{i}")
                    m['anz'] = st.number_input(f"Anzahl #{i+1}", 0, 10000, m['anz'], key=f"m_a_{i}")
                    m['w'] = st.number_input(f"Leistung (Wp) #{i+1}", 0, 700, m['w'], key=f"m_w_{i}")
                    if st.button(f"🗑️ Modul {i+1}", key=f"m_d_{i}"): st.session_state.pv_module.pop(i); st.rerun()

        with col_m2:
            st.markdown("**Wechselrichter**")
            if st.button("➕ Wechselrichter hinzufügen"): st.session_state.pv_wr.append({'typ': '', 'anz': 0, 'p': 10.0})
            for i, w in enumerate(st.session_state.pv_wr):
                with st.container(border=True):
                    w['typ'] = st.text_input(f"WR-Hersteller/Typ #{i+1}", w['typ'], key=f"w_t_{i}")
                    w['anz'] = st.number_input(f"Anzahl #{i+1}", 0, 100, w['anz'], key=f"w_a_{i}")
                    w['p'] = st.number_input(f"AC-Leistung (kW) #{i+1}", 0.0, 500.0, w['p'], key=f"w_p_{i}")
                    if st.button(f"🗑️ WR {i+1}", key=f"w_d_{i}"): st.session_state.pv_wr.pop(i); st.rerun()

# --- TAB: SPEICHER ---
if show_storage:
    with tabs[tabs_labels.index("🔋 Speicher")]:
        st.header("🔋 Batteriespeicher (BESS)")
        if st.button("➕ Speichersystem hinzufügen"): 
            st.session_state.bat_systeme.append({'typ': '', 'kwh': 10.0, 'zyklen': 6000, 'dod': 90, 'c_rate': 1.0})
        
        for i, b in enumerate(st.session_state.bat_systeme):
            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                b['typ'] = c1.text_input(f"System-Name #{i+1}", b['typ'], key=f"b_t_{i}")
                b['kwh'] = c2.number_input(f"Kapazität (kWh) #{i+1}", 0.0, 5000.0, b['kwh'], key=f"b_k_{i}")
                b['c_rate'] = c3.number_input(f"C-Rate (Leistung)", 0.1, 2.0, b['c_rate'], key=f"b_c_{i}")
                
                c4, c5, c6 = st.columns(3)
                b['zyklen'] = c4.number_input("Garantierte Zyklen", 1000, 15000, b['zyklen'], key=f"b_z_{i}")
                b['dod'] = c5.slider("Entladetiefe (DoD %)", 50, 100, b['dod'], key=f"b_d_{i}")
                if c6.button("🗑️ Löschen", key=f"b_del_{i}"): st.session_state.bat_systeme.pop(i); st.rerun()
        
        total_storage_kwh = sum(b['kwh'] for b in st.session_state.bat_systeme)

# --- TAB: MOBILITÄT & LADEN ---
if show_mobility:
    with tabs[tabs_labels.index("🚗 Mobilität & Laden")]:
        st.header("🚗 Ladeinfrastruktur & Fuhrpark")
        
        # Ladeinfrastruktur AC/DC
        st.subheader("🔌 Ladepunkte (AC & DC)")
        if st.button("➕ Ladepunkt hinzufügen"):
            st.session_state.lade_punkte.append({'art': 'AC', 'p': 11.0, 'anz': 1})
        
        for i, lp in enumerate(st.session_state.lade_punkte):
            with st.container(border=True):
                l1, l2, l3, l4 = st.columns([2,1,1,1])
                lp['art'] = l1.selectbox(f"Typ #{i+1}", ["AC (Wallbox)", "DC (Schnelllader)"], key=f"lp_a_{i}")
                lp['p'] = l2.number_input(f"kW pro LP #{i+1}", 1.0, 400.0, lp['p'], key=f"lp_p_{i}")
                lp['anz'] = l3.number_input(f"Anzahl #{i+1}", 1, 100, lp['anz'], key=f"lp_n_{i}")
                if l4.button("🗑️", key=f"lp_d_{i}"): st.session_state.lade_punkte.pop(i); st.rerun()
                
                if "AC" in lp['art']: total_lp_power_ac += (lp['p'] * lp['anz'])
                else: total_lp_power_dc += (lp['p'] * lp['anz'])

        # THG Quote & Fuhrpark
        st.divider()
        st.subheader("🚐 Fuhrpark & THG-Quote")
        if st.button("➕ Fahrzeuggruppe hinzufügen"):
            st.session_state.fuhrpark.append({'art': 'PKW', 'anz': 1})
        for i, f in enumerate(st.session_state.fuhrpark):
            with st.container(border=True):
                f1, f2, f3 = st.columns([2,1,1])
                f['art'] = f1.selectbox(f"Typ #{i+1}", ["PKW", "LKW", "Busse"], key=f"f_a_{i}")
                f['anz'] = f2.number_input(f"Anzahl", 1, 500, f['anz'], key=f"f_n_{i}")
                if f3.button("🗑️", key=f"f_d_{i}"): st.session_state.fuhrpark.pop(i); st.rerun()
                
                # THG Erlös (Referenz 2026)
                thg_val = {"PKW": 115, "LKW": 420, "Busse": 480}
                thg_revenue += (f['anz'] * thg_val[f['art']])

# --- TAB 1: ROI & NETZ-CHECK ---
with tabs[0]:
    st.header("📊 Wirtschaftlichkeit & Netz-Check")
    
    # NETZANSCHLUSS CHECK
    total_peak = max(total_kwp, (total_lp_power_ac + total_lp_power_dc) * 0.6)
    with st.container(border=True):
        st.subheader("🔌 Netzanschluss-Kapazität")
        nc1, nc2, nc3 = st.columns(3)
        nc1.metric("Anschlussleistung", f"{grid_limit_kva} kVA")
        nc2.metric("Peak-Last (Sim.)", f"{total_peak:.1f} kW")
        auslastung = (total_peak / grid_limit_kva) * 100
        nc3.metric("Auslastung", f"{auslastung:.1f} %")
        if total_peak > grid_limit_kva: st.error("Netzanschluss kritisch!")
        else: st.success("Netzanschluss OK")

    # ROI RECHNUNG
    pv_ertrag = total_kwp * 1020
    if "Volleinspeisung" in pv_betriebsmodus and only_pv:
        benefit = pv_ertrag * (einspeise_verg_voll / 100)
    else:
        # Eigenverbrauch (Dynamisch)
        ev_rate = 0.3 + (0.35 if total_storage_kwh > 0 else 0) + (0.15 if show_mobility else 0)
        ev_rate = min(ev_rate, 0.95)
        benefit = (pv_ertrag * ev_rate * (strompreis_netz/100)) + \
                  (pv_ertrag * (1-ev_rate) * (einspeise_verg_ueberschuss/100)) + thg_revenue

    invest = (total_kwp * 1150) + (total_storage_kwh * 550) + (len(st.session_state.lade_punkte) * 2500)
    
    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Investment", f"{invest:,.0f} €")
    r2.metric("Erlös/Ersparnis p.a.", f"{benefit:,.2f} €")
    r3.metric("ROI", f"{invest/benefit if benefit > 0 else 0:.1f} Jahre")

    # Amortisationsgraph
    x = np.arange(21)
    y = [-invest + (benefit * i) for i in x]
    fig = px.line(x=x, y=y, title="Kumulierter Cashflow (20 Jahre)", labels={'x':'Jahre', 'y':'€'})
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
