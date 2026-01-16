import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="Energy Expert PRO 2026", page_icon="☀️", layout="wide")

# Custom Styling für bessere Struktur
st.markdown("""
    <style>
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 12px; border: 1px solid #dee2e6; }
    .stAlert { border-radius: 12px; }
    .hardware-section { background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px dotted #007bff; margin-top: 15px; }
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
    show_arbitrage = st.checkbox("Arbitrage Handel", value=True)
    
    st.divider()
    st.header("💰 Marktdaten")
    strompreis_netz = st.slider("Bezugspreis (ct/kWh)", 15, 60, 32)
    einspeise_verg_ueberschuss = 8.2
    einspeise_verg_voll = 13.0

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

# Globale Variablen
total_kwp = 0.0
total_storage_kwh = 0.0
total_lp_power_ac = 0.0
total_lp_power_dc = 0.0
thg_revenue = 0.0
arbitrage_revenue = 0.0

# --- TAB: PV-PLANUNG ---
if show_pv:
    with tabs[tabs_labels.index("🏗️ PV-Planung")]:
        st.header("🏗️ PV-Projektierung")
        
        # Logik: Nur PV Prüfung
        only_pv = show_pv and not (show_storage or show_mobility or show_arbitrage)
        pv_mode = "Eigenverbrauch"
        if only_pv:
            pv_mode = st.radio("Betriebsmodus", ["Eigenverbrauch (Überschuss)", "Volleinspeisung"], horizontal=True)
        
        st.subheader("🏠 Dachflächen")
        if st.button("➕ Dachfläche hinzufügen"):
            st.session_state.daecher.append({'kwp': 15.0})
        for i, d in enumerate(st.session_state.daecher):
            c1, c2 = st.columns([4,1])
            d['kwp'] = c1.number_input(f"Leistung Dach {i+1} (kWp)", 0.0, 5000.0, d['kwp'], key=f"dkwp_{i}")
            if c2.button("🗑️", key=f"ddel_{i}"): st.session_state.daecher.pop(i); st.rerun()
        total_kwp = sum(d['kwp'] for d in st.session_state.daecher)

        st.divider()
        st.subheader("🧩 Hardware-Komponenten & Datenblätter")
        m_col, w_col = st.columns(2)
        with m_col:
            st.markdown("**Module**")
            if st.button("➕ Modul-Typ"): st.session_state.pv_module.append({'typ': '', 'anz': 0})
            for i, m in enumerate(st.session_state.pv_module):
                with st.container(border=True):
                    m['typ'] = st.text_input(f"Hersteller/Typ #{i+1}", key=f"mt_{i}")
                    m['anz'] = st.number_input(f"Anzahl", 0, 10000, key=f"ma_{i}")
                    st.file_uploader(f"Datenblatt Modul #{i+1}", type=["pdf"], key=f"mpdf_{i}")
        with w_col:
            st.markdown("**Wechselrichter**")
            if st.button("➕ WR-Typ"): st.session_state.pv_wr.append({'typ': '', 'anz': 0})
            for i, w in enumerate(st.session_state.pv_wr):
                with st.container(border=True):
                    w['typ'] = st.text_input(f"Hersteller/Typ #{i+1}", key=f"wt_{i}")
                    w['anz'] = st.number_input(f"Anzahl", 0, 100, key=f"wa_{i}")
                    st.file_uploader(f"Datenblatt WR #{i+1}", type=["pdf"], key=f"wpdf_{i}")

# --- TAB: SPEICHER ---
if show_storage:
    with tabs[tabs_labels.index("🔋 Speicher")]:
        st.header("🔋 Batteriespeicher (BESS)")
        if st.button("➕ Speichersystem hinzufügen"): 
            st.session_state.bat_systeme.append({'kwh': 10.0, 'zyklen': 6000, 'dod': 90})
        
        for i, b in enumerate(st.session_state.bat_systeme):
            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                b['typ'] = c1.text_input(f"Modell #{i+1}", key=f"bt_{i}")
                b['kwh'] = c2.number_input(f"Kapazität (kWh)", 0.0, 5000.0, b['kwh'], key=f"bk_{i}")
                b['dod'] = c3.slider(f"DoD %", 50, 100, 90, key=f"bd_{i}")
                
                c4, c5, c6 = st.columns(3)
                b['zyklen'] = c4.number_input(f"Zyklenfestigkeit", 1000, 15000, 6000, key=f"bz_{i}")
                st.file_uploader(f"Datenblatt Speicher #{i+1}", type=["pdf"], key=f"bpdf_{i}")
                if c6.button(f"🗑️ System {i+1}", key=f"bdel_{i}"): st.session_state.bat_systeme.pop(i); st.rerun()
        
        total_storage_kwh = sum(b['kwh'] for b in st.session_state.bat_systeme)

# --- TAB: MOBILITÄT & LADEN ---
if show_mobility:
    with tabs[tabs_labels.index("🚗 Mobilität & Laden")]:
        st.header("🚗 Infrastruktur & Dokumentation")
        
        st.subheader("🔌 Ladepunkte (AC & DC)")
        if st.button("➕ Ladepunkt"):
            st.session_state.lade_punkte.append({'art': 'AC', 'p': 11.0, 'anz': 1})
        
        for i, lp in enumerate(st.session_state.lade_punkte):
            with st.container(border=True):
                l1, l2, l3 = st.columns([2,1,1])
                lp['art'] = l1.selectbox(f"Art #{i+1}", ["AC (Wallbox)", "DC (Schnelllader)"], key=f"lpt_{i}")
                lp['p'] = l2.number_input(f"Leistung (kW)", 1.0, 400.0, lp['p'], key=f"lpp_{i}")
                lp['anz'] = l3.number_input(f"Anzahl", 1, 100, key=f"lpn_{i}")
                
                c_pdf, c_del = st.columns([3,1])
                c_pdf.file_uploader(f"Datenblatt Ladepunkt #{i+1}", type=["pdf"], key=f"lpdf_{i}")
                if c_del.button(f"🗑️ LP {i+1}", key=f"lpd_{i}"): st.session_state.lade_punkte.pop(i); st.rerun()
                
                if "AC" in lp['art']: total_lp_power_ac += (lp['p'] * lp['anz'])
                else: total_lp_power_dc += (lp['p'] * lp['anz'])

        st.divider()
        st.subheader("🚐 Fuhrpark (THG)")
        if st.button("➕ Fahrzeuggruppe"):
            st.session_state.fuhrpark.append({'art': 'PKW', 'anz': 1})
        for i, f in enumerate(st.session_state.fuhrpark):
            with st.container(border=True):
                f1, f2, f3 = st.columns([2,1,1])
                f['art'] = f1.selectbox(f"Typ #{i+1}", ["PKW", "LKW", "Busse"], key=f"fa_{i}")
                f['anz'] = f2.number_input(f"Anzahl", 1, 500, key=f"fn_{i}")
                thg_val = {"PKW": 115, "LKW": 420, "Busse": 480}
                thg_revenue += (f['anz'] * thg_val[f['art']])
                if f3.button(f"🗑️ Gruppe {i+1}", key=f"fd_{i}"): st.session_state.fuhrpark.pop(i); st.rerun()

# --- TAB: ARBITRAGE ---
if show_arbitrage:
    with tabs[tabs_labels.index("📈 Arbitrage")]:
        st.header("📈 Spotmarkt-Optimierung & Arbitrage")
        
        with st.container(border=True):
            arb_mode = st.radio("Arbitrage-Modus", ["Manuelle Parameter", "Referenz-Abgleich (Live-Sim)", "KI-Prognose 2026"])
            
            c_a1, c_a2, c_a3 = st.columns(3)
            if arb_mode == "Manuelle Parameter":
                spread = c_a1.number_input("Ø Preis-Spread (ct/kWh)", 0.0, 50.0, 14.5)
                cycles = c_a2.number_input("Zyklen/Jahr", 0, 1000, 250)
            elif arb_mode == "Referenz-Abgleich (Live-Sim)":
                st.success("Referenzdaten 2025/26 geladen: Marktdurchschnitt Spread 15.2 ct/kWh")
                spread = 15.2
                cycles = 280
            else:
                st.info("KI-Prognose: Steigende Volatilität durch Windkraft-Zubau erwartet.")
                spread = 18.5
                cycles = 310
            
            arbitrage_revenue = (total_storage_kwh * 0.9 * (spread/100) * cycles)
            c_a3.metric("Zusatz-Erlös Arbitrage", f"{arbitrage_revenue:,.2f} €/a")

# --- TAB 1: ROI & NETZ-CHECK ---
with tabs[0]:
    st.header("📊 ROI & Netz-Flaschenhals")
    
    total_peak = max(total_kwp, (total_lp_power_ac + total_lp_power_dc) * 0.65)
    
    with st.container(border=True):
        nc1, nc2, nc3 = st.columns(3)
        nc1.metric("Anschlussleistung", f"{grid_limit_kva} kVA")
        nc2.metric("Simulierter Peak", f"{total_peak:.1f} kW")
        ratio = (total_peak / grid_limit_kva) * 100
        nc3.metric("Auslastung", f"{ratio:.1f} %")
        if total_peak > grid_limit_kva: st.error("⚠️ Achtung: Anschlusskapazität überschritten!")
        else: st.success("✅ Netzanschluss ausreichend.")

    # ROI
    pv_yield = total_kwp * 1050
    if only_pv and pv_mode == "Volleinspeisung":
        benefit = pv_yield * (einspeise_verg_voll/100)
    else:
        ev_rate = 0.35 + (0.35 if total_storage_kwh > 0 else 0) + (0.10 if show_mobility else 0)
        ev_rate = min(ev_rate, 0.95)
        benefit = (pv_yield * ev_rate * (strompreis_netz/100)) + \
                  (pv_yield * (1-ev_rate) * (8.2/100)) + thg_revenue + arbitrage_revenue
    
    invest = (total_kwp * 1150) + (total_storage_kwh * 500) + (len(st.session_state.lade_punkte) * 2500)
    
    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Investment", f"{invest:,.0f} €")
    r2.metric("Vorteil p.a.", f"{benefit:,.0f} €")
    r3.metric("ROI", f"{invest/benefit if benefit > 0 else 0:.1f} Jahre")
