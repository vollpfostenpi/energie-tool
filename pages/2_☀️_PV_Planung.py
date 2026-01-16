import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="Energy Architect Pro 2026", page_icon="☀️", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .stMetric { background-color: #f0f4f8; padding: 15px; border-radius: 12px; border: 1px solid #d1d9e6; }
    .stAlert { border-radius: 12px; }
    .section-box { padding: 20px; border-radius: 15px; background-color: #ffffff; border: 1px solid #e2e8f0; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("☀️ Energy Architect & ROI-Prozessor (Edition 2026)")

# --- SIDEBAR: NETZ & GLOBALE DATEN ---
with st.sidebar:
    st.header("🔌 Netz & Infrastruktur")
    grid_limit_kva = st.number_input("Max. Netzanschluss (kVA)", 10, 5000, 43, help="Technischer Flaschenhals des Standorts")
    has_imsys = st.toggle("iMSys vorhanden (Smart Meter)", value=True, help="Ohne iMSys greift die 60%-Regelung.")
    
    st.divider()
    st.header("🎯 Modulauswahl")
    show_pv = st.checkbox("Photovoltaik (PV)", value=True)
    show_storage = st.checkbox("Batteriespeicher (BESS)", value=True)
    show_heat = st.checkbox("Wärmepumpe (Sektorenkopplung)", value=True)
    show_mobility = st.checkbox("Fuhrpark & THG", value=True)
    show_arbitrage = st.checkbox("Arbitrage Handel", value=False)
    
    st.divider()
    st.header("💰 Marktdaten")
    strompreis_netz = st.slider("Bezugspreis (ct/kWh)", 15, 60, 32)
    einspeise_verg = st.number_input("EEG Vergütung (ct/kWh)", 5.0, 15.0, 8.1)

# --- SESSION STATE INITIALISIERUNG ---
for key in ['daecher', 'lade_punkte', 'fuhrpark']:
    if key not in st.session_state: st.session_state[key] = []

# Tabs Setup
tabs_labels = ["📊 ROI & Netz-Check"]
if show_pv: tabs_labels.append("🏗️ PV-Planung")
if show_storage: tabs_labels.append("🔋 Speicher")
if show_heat: tabs_labels.append("🔥 Wärme/Kälte")
if show_mobility: tabs_labels.append("🚗 Mobilität & THG")
if show_arbitrage: tabs_labels.append("📈 Arbitrage")
tabs = st.tabs(tabs_labels)

# Globale Variablen
total_kwp = 0.0
total_storage_kwh = 0.0
thg_revenue = 0.0
wp_strombedarf = 0.0
imsys_costs = 0.0
arb_rev = 0.0

# --- TAB: PV-PLANUNG ---
if show_pv:
    with tabs[tabs_labels.index("🏗️ PV-Planung")]:
        st.header("🏗️ PV-Projektierung")
        if st.button("➕ Neues Dach"):
            st.session_state.daecher.append({'typ': 'Satteldach', 'azimut': 0, 'kwp': 15.0})
        
        for i, dach in enumerate(st.session_state.daecher):
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2,2,2,1])
                dach['typ'] = c1.selectbox(f"Dachtyp #{i+1}", ["Satteldach", "Flachdach", "Ost-West"], key=f"dt_{i}")
                dach['azimut'] = c2.slider(f"Ausrichtung #{i+1} (Süd=0)", -180, 180, int(dach['azimut']), key=f"da_{i}")
                dach['kwp'] = c3.number_input(f"Leistung (kWp) #{i+1}", 0.0, 2000.0, value=float(dach['kwp']), key=f"dp_{i}")
                if c4.button("🗑️", key=f"dd_{i}"): st.session_state.daecher.pop(i); st.rerun()
        
        total_kwp = sum(d['kwp'] for d in st.session_state.daecher)
        
        # iMSys Kostenkalkulation 2026
        if total_kwp > 7:
            imsys_costs = 50.0 if total_kwp <= 15 else (110.0 if total_kwp <= 25 else 140.0)
            imsys_costs += 50.0 # Steuerbox-Zusatz
            st.info(f"📋 Messstellenbetrieb (iMSys + Steuerbox): **{imsys_costs} €/Jahr**")

        st.subheader("📄 Komponenten-Details")
        with st.container(border=True):
            st.text_input("Hersteller / Modultyp", placeholder="z.B. Jinko Tiger Neo")
            st.file_uploader("Datenblatt Module", type=["pdf"], key="pdf_pv")

# --- TAB: SPEICHER ---
if show_storage:
    with tabs[tabs_labels.index("🔋 Speicher")]:
        st.header("🔋 Batteriespeicher (BESS)")
        c_b1, c_b2 = st.columns(2)
        total_storage_kwh = c_b1.number_input("Nennkapazität (kWh)", 0.0, 5000.0, 20.0)
        c_rate = c_b2.number_input("C-Rate (Leistungsfaktor)", 0.1, 2.0, 1.0)
        storage_p = total_storage_kwh * c_rate
        st.metric("Verfügbare Leistung", f"{storage_p:.1f} kW")
        
        st.subheader("📄 Speicher-Hardware")
        with st.container(border=True):
            st.text_input("Hersteller / Modell Speicher", placeholder="z.B. BYD Battery-Box")
            st.file_uploader("Datenblatt Speicher", type=["pdf"], key="pdf_bat")

# --- TAB: WÄRME/KÄLTE ---
if show_heat:
    with tabs[tabs_labels.index("🔥 Wärme/Kälte")]:
        st.header("🔥 Sektorenkopplung Wärmepumpe")
        with st.container(border=True):
            h1, h2, h3 = st.columns(3)
            waermebedarf = h1.number_input("Heizbedarf (kWh/a)", 0, 500000, 20000)
            jaz_heiz = h2.number_input("Jahresarbeitszahl (Heizen)", 1.0, 6.0, 3.8)
            wp_strom_heiz = waermebedarf / jaz_heiz
            
            show_cooling = st.toggle("Kühlung im Sommer integrieren", value=True)
            wp_strom_kuehl = 0
            if show_cooling:
                kuehlbedarf = st.number_input("Kühlbedarf (kWh/a)", 0, 100000, 5000)
                jaz_kuehl = st.number_input("EER (Wirkungsgrad Kühlen)", 1.0, 6.0, 3.0)
                wp_strom_kuehl = kuehlbedarf / jaz_kuehl
            
            wp_strombedarf = wp_strom_heiz + wp_strom_kuehl
            st.metric("Gesamt-Strombedarf WP", f"{wp_strombedarf:,.0f} kWh/a")

# --- TAB: MOBILITÄT & THG ---
if show_mobility:
    with tabs[tabs_labels.index("🚗 Mobilität & THG")]:
        st.header("🚗 Fuhrpark & THG-Management")
        
        thg_mode = st.radio("THG-Quote", ["Referenz 2026", "Manuell"], horizontal=True)
        t1, t2, t3 = st.columns(3)
        v_pkw, v_lkw, v_bus = (115.0, 420.0, 480.0) if thg_mode == "Referenz 2026" else (
            t1.number_input("PKW (€)", 0, 500, 110),
            t2.number_input("LKW (€)", 0, 2000, 400),
            t3.number_input("Bus (€)", 0, 2000, 450)
        )

        if st.button("➕ Fahrzeuggruppe"):
            st.session_state.fuhrpark.append({'art': 'PKW (M1)', 'anz': 1})
            
        for i, f in enumerate(st.session_state.fuhrpark):
            with st.container(border=True):
                f1, f2, f3 = st.columns([2,1,1])
                f['art'] = f1.selectbox(f"Klasse #{i+1}", ["PKW (M1)", "LKW (N1-N3)", "Busse (M2/M3)"], key=f"fa_{i}")
                f['anz'] = f2.number_input(f"Anzahl #{i+1}", 1, 500, value=f['anz'], key=f"fan_{i}")
                if f3.button("🗑️", key=f"fdel_{i}"): st.session_state.fuhrpark.pop(i); st.rerun()
                
                thg_revenue += (f['anz'] * (v_pkw if "PKW" in f['art'] else (v_lkw if "LKW" in f['art'] else v_bus)))
        
        st.subheader("📄 Ladeinfrastruktur-Hardware")
        with st.container(border=True):
            st.text_input("Hersteller / Typ Wallbox")
            st.file_uploader("Datenblatt Ladestation", type=["pdf"], key="pdf_lp")

# --- TAB: ARBITRAGE ---
if show_arbitrage:
    with tabs[tabs_labels.index("📈 Arbitrage")]:
        st.header("📈 Markt-Handel")
        arb_spread = st.number_input("Ø Preis-Spread (ct/kWh)", 0.0, 40.0, 14.5)
        arb_rev = (total_storage_kwh * 0.9 * (arb_spread/100) * 250)
        st.metric("Arbitrage-Deckungsbeitrag", f"{arb_rev:,.2f} €/a")

# --- TAB 1: ROI & NETZ-CHECK ---
with tabs[0]:
    st.header("📋 Business Case")
    
    pv_ertrag = total_kwp * 1020
    if not has_imsys: pv_ertrag *= 0.93 # 60% Regelung Verlust
    
    # EV-Quote Simulation
    ev_rate = 0.3 + (0.35 if total_storage_kwh > 0 else 0) + (0.15 if show_heat else 0) + (0.10 if show_mobility else 0)
    ev_rate = min(ev_rate, 0.95)
    
    annual_saving = (pv_ertrag * ev_rate * (strompreis_netz/100)) + \
                    (pv_ertrag * (1-ev_rate) * (einspeise_verg/100)) + \
                    thg_revenue + arb_rev - imsys_costs

    invest = (total_kwp * 1150) + (total_storage_kwh * 550)
    
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Investment", f"{invest:,.0f} €")
    c2.metric("Einsparung/Erlös p.a.", f"{annual_saving:,.0f} €")
    c3.metric("ROI", f"{invest/annual_saving if annual_saving > 0 else 0:.1f} Jahre")

    st.divider()
    st.subheader("🔌 Netzanschluss-Prüfung")
    peak_in = total_kwp * 0.85
    if peak_in > grid_limit_kva:
        st.error(f"❌ Netzanschluss kritisch! Peak ({peak_in:.1f} kW) > Limit ({grid_limit_kva} kVA)")
    else:
        st.success(f"✅ Netzanschluss OK (Auslastung: {(peak_in/grid_limit_kva)*100:.1f}%)")
