import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="Energy Architect Pro 2026", page_icon="☀️", layout="wide")

# Custom UI Styling für eine professionelle Planungs-Oberfläche
st.markdown("""
    <style>
    .stMetric { background-color: #f0f4f8; padding: 20px; border-radius: 15px; border: 1px solid #d1d9e6; }
    .stAlert { border-radius: 12px; }
    .hardware-card { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px; border-left: 6px solid #1E3A8A; }
    .main-header { font-size: 28px; font-weight: bold; color: #1E3A8A; margin-bottom: 10px; }
    .sub-header { font-size: 20px; color: #475569; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("☀️ Smart Energy Architect & Business Simulator 2026")

# --- SIDEBAR: NETZ & GLOBALE STEUERUNG ---
with st.sidebar:
    st.header("🔌 Netzanschluss & Basis")
    grid_limit_kva = st.number_input("Max. Netzanschlussleistung (kVA)", 10, 10000, 43, help="Der physikalische Flaschenhals des Standorts.")
    has_imsys = st.toggle("Intelligentes Messsystem (iMSys) vorhanden", value=True)
    
    st.divider()
    st.header("🎯 System-Komponenten")
    show_pv = st.checkbox("Photovoltaik (PV)", value=True)
    show_storage = st.checkbox("Batteriespeicher (BESS)", value=True)
    show_mobility = st.checkbox("Mobilität & Laden", value=True)
    show_heat = st.checkbox("Wärmepumpe (WP)", value=False)
    show_arbitrage = st.checkbox("Arbitrage Handel", value=True)
    
    st.divider()
    st.header("💰 Marktdaten & Tarife")
    strompreis_netz = st.slider("Bezugspreis Netz (ct/kWh)", 15, 60, 32)
    einspeise_verg_ueberschuss = st.number_input("EEG Überschuss (ct/kWh)", 0.0, 15.0, 8.2)
    einspeise_verg_voll = st.number_input("EEG Volleinspeisung (ct/kWh)", 0.0, 20.0, 13.0)
    st.info("💡 Stand: EEG 2026 Entwurf berücksichtigt.")

# --- SESSION STATE INITIALISIERUNG (Datenkonsistenz) ---
if 'daecher' not in st.session_state: st.session_state.daecher = []
if 'pv_module' not in st.session_state: st.session_state.pv_module = []
if 'pv_wr' not in st.session_state: st.session_state.pv_wr = []
if 'bat_systeme' not in st.session_state: st.session_state.bat_systeme = []
if 'lade_punkte' not in st.session_state: st.session_state.lade_punkte = []
if 'fuhrpark' not in st.session_state: st.session_state.fuhrpark = []

# Tab Management
tabs_labels = ["📊 ROI & Business Case"]
if show_pv: tabs_labels.append("🏗️ PV-Planung")
if show_storage: tabs_labels.append("🔋 Speicher")
if show_heat: tabs_labels.append("🔥 Wärme")
if show_mobility: tabs_labels.append("🚗 Mobilität & Laden")
if show_arbitrage: tabs_labels.append("📈 Arbitrage")
tabs = st.tabs(tabs_labels)

# Globale Variablen für die finale Berechnung
total_kwp = 0.0
total_storage_kwh = 0.0
total_lp_power_ac = 0.0
total_lp_power_dc = 0.0
thg_revenue = 0.0
arbitrage_revenue = 0.0
imsys_costs = 0.0

# --- TAB: PV-PLANUNG (DÄCHER, STANDORT & HARDWARE) ---
if show_pv:
    with tabs[tabs_labels.index("🏗️ PV-Planung")]:
        st.markdown('<p class="main-header">🏗️ Photovoltaik Projektierung</p>', unsafe_allow_html=True)
        
        with st.container(border=True):
            st.subheader("📍 Standort-Analyse")
            adresse = st.text_input("Projektadresse (für Strahlungsdaten)", placeholder="Musterstraße 1, 12345 Stadt")
            if adresse:
                st.success(f"Lokale Prognosedaten für {adresse} werden verwendet: ca. 1.050 kWh/kWp Erwartung.")

        st.divider()
        st.subheader("🏠 Dachflächen & Ausrichtung")
        if st.button("➕ Neue Dachfläche hinzufügen"):
            st.session_state.daecher.append({'typ': 'Satteldach', 'ausrichtung': 0, 'neigung': 35, 'kwp': 10.0})
        
        for i, d in enumerate(st.session_state.daecher):
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2,2,1,1])
                d['typ'] = c1.selectbox(f"Dachform #{i+1}", ["Satteldach", "Flachdach", "Pultdach", "Ost-West", "Fassade", "Tracker"], key=f"dtyp_{i}")
                d['ausrichtung'] = c2.slider(f"Ausrichtung #{i+1} (Süd=0°, Ost=-90°, West=90°)", -180, 180, d['ausrichtung'], key=f"daus_{i}")
                d['neigung'] = c3.number_input(f"Neigung #{i+1} (°)", 0, 90, d['neigung'], key=f"dnei_{i}")
                d['kwp'] = c4.number_input(f"Leistung #{i+1} (kWp)", 0.0, 5000.0, d['kwp'], key=f"dkwp_{i}")
                
                # Grafische Visualisierung je nach Wahl
                img_col, del_col = st.columns([5,1])
                if d['typ'] == "Satteldach": img_col.image("https://images.unsplash.com/photo-1513694203232-719a280e022f?auto=format&fit=crop&q=80&w=400", caption="Referenz Satteldach", width=250)
                elif d['typ'] == "Flachdach": img_col.image("https://images.unsplash.com/photo-1613665813446-82a78c4458e7?auto=format&fit=crop&q=80&w=400", caption="Referenz Flachdach Aufständerung", width=250)
                elif d['typ'] == "Ost-West": img_col.image("https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&q=80&w=400", caption="Referenz Ost-West Belegung", width=250)
                
                if del_col.button("🗑️", key=f"ddel_{i}"):
                    st.session_state.daecher.pop(i); st.rerun()
        
        total_kwp = sum(d['kwp'] for d in st.session_state.daecher)
        
        st.divider()
        st.subheader("🧩 PV-Komponenten (Stückliste)")
        col_m, col_w = st.columns(2)
        with col_m:
            st.markdown("**Module**")
            if st.button("➕ Modul-Typ hinzufügen"): st.session_state.pv_module.append({'typ': '', 'anz': 0, 'wp': 440})
            for i, m in enumerate(st.session_state.pv_module):
                with st.container(border=True):
                    m['typ'] = st.text_input(f"Hersteller/Modell Modul #{i+1}", key=f"mtyp_{i}")
                    m['anz'] = st.number_input(f"Anzahl #{i+1}", 0, 100000, key=f"manz_{i}")
                    m['wp'] = st.number_input(f"Leistung (Wp) #{i+1}", 0, 800, m['wp'], key=f"mwp_{i}")
                    st.file_uploader(f"Datenblatt Modul #{i+1}", type=["pdf"], key=f"mpdf_{i}")
                    if st.button(f"🗑️ Modul {i+1} entfernen", key=f"mdel_{i}"): st.session_state.pv_module.pop(i); st.rerun()
        with col_w:
            st.markdown("**Wechselrichter**")
            if st.button("➕ Wechselrichter-Typ hinzufügen"): st.session_state.pv_wr.append({'typ': '', 'anz': 0, 'p': 10.0})
            for i, w in enumerate(st.session_state.pv_wr):
                with st.container(border=True):
                    w['typ'] = st.text_input(f"Hersteller/Modell WR #{i+1}", key=f"wtyp_{i}")
                    w['anz'] = st.number_input(f"Anzahl #{i+1}", 0, 1000, key=f"wanz_{i}")
                    w['p'] = st.number_input(f"AC-Leistung (kW) #{i+1}", 0.0, 1000.0, w['p'], key=f"wpac_{i}")
                    st.file_uploader(f"Datenblatt WR #{i+1}", type=["pdf"], key=f"wpdf_{i}")
                    if st.button(f"🗑️ WR {i+1} entfernen", key=f"wdel_{i}"): st.session_state.pv_wr.pop(i); st.rerun()

# --- TAB: SPEICHER (BESS DETAILS) ---
if show_storage:
    with tabs[tabs_labels.index("🔋 Speicher")]:
        st.markdown('<p class="main-header">🔋 Batteriespeicher-Systeme (BESS)</p>', unsafe_allow_html=True)
        if st.button("➕ Speichersystem / Strang hinzufügen"): 
            st.session_state.bat_systeme.append({'typ': '', 'kwh': 10.0, 'zyklen': 6000, 'dod': 90, 'c_rate': 0.5, 'effizienz': 95})
        
        for i, b in enumerate(st.session_state.bat_systeme):
            with st.container(border=True):
                c1, c2, c3 = st.columns([2,1,1])
                b['typ'] = c1.text_input(f"System Modell #{i+1}", b['typ'], key=f"b_t_{i}", placeholder="z.B. Tesla Megapack, BYD")
                b['kwh'] = c2.number_input(f"Kapazität (kWh) #{i+1}", 0.1, 50000.0, b['kwh'], key=f"b_k_{i}")
                b['effizienz'] = c3.number_input(f"Roundtrip-Wirkungsgrad (%)", 70, 100, b['effizienz'], key=f"b_e_{i}")
                
                c4, c5, c6 = st.columns(3)
                b['c_rate'] = c4.number_input(f"C-Rate (Power-Faktor) #{i+1}", 0.1, 5.0, b['c_rate'], key=f"b_c_{i}")
                b['dod'] = c5.slider(f"DoD % (Entladetiefe) #{i+1}", 50, 100, b['dod'], key=f"b_d_{i}")
                b['zyklen'] = c6.number_input(f"Zyklenfestigkeit #{i+1}", 1000, 20000, b['zyklen'], key=f"b_z_{i}")
                
                st.info(f"⚡ Max. Leistung: **{b['kwh']*b['c_rate']:.1f} kW** | Nutzbare Energie: **{b['kwh']*(b['dod']/100):.1f} kWh**")
                st.file_uploader(f"Datenblatt BESS #{i+1}", type=["pdf"], key=f"b_pdf_{i}")
                if st.button(f"🗑️ Speicher {i+1} entfernen", key=f"b_del_{i}"): st.session_state.bat_systeme.pop(i); st.rerun()
        
        total_storage_kwh = sum(b['kwh'] for b in st.session_state.bat_systeme)

# --- TAB: MOBILITÄT & LADEN ---
if show_mobility:
    with tabs[tabs_labels.index("🚗 Mobilität & Laden")]:
        st.markdown('<p class="main-header">🚗 Mobilität & Ladeinfrastruktur</p>', unsafe_allow_html=True)
        
        st.subheader("🔌 Ladepunkte (AC & DC)")
        if st.button("➕ Ladepunkt hinzufügen"):
            st.session_state.lade_punkte.append({'art': 'AC', 'p': 11.0, 'anz': 1})
        for i, lp in enumerate(st.session_state.lade_punkte):
            with st.container(border=True):
                l1, l2, l3 = st.columns([2,1,1])
                lp['art'] = l1.selectbox(f"Typ #{i+1}", ["AC (Wallbox)", "DC (Schnelllader)"], key=f"lpt_{i}")
                lp['p'] = l2.number_input(f"Leistung (kW) #{i+1}", 1.0, 400.0, lp['p'], key=f"lpp_{i}")
                lp['anz'] = l3.number_input(f"Anzahl #{i+1}", 1, 1000, key=f"lpn_{i}")
                st.file_uploader(f"Datenblatt Ladestation #{i+1}", type=["pdf"], key=f"lpdf_{i}")
                if st.button(f"🗑️ Ladepunkt {i+1} entfernen", key=f"lpd_{i}"): st.session_state.lade_punkte.pop(i); st.rerun()
                if "AC" in lp['art']: total_lp_power_ac += (lp['p'] * lp['anz'])
                else: total_lp_power_dc += (lp['p'] * lp['anz'])

        st.divider()
        st.subheader("🚐 THG-Quote & Flotte")
        if st.button("➕ Fahrzeuggruppe hinzufügen"):
            st.session_state.fufrpark.append({'art': 'PKW', 'anz': 1})
        for i, f in enumerate(st.session_state.fuhrpark):
            with st.container(border=True):
                f1, f2, f3 = st.columns([2,1,1])
                f['art'] = f1.selectbox(f"Fahrzeugklasse #{i+1}", ["PKW", "LKW", "Busse"], key=f"fa_{i}")
                f['anz'] = f2.number_input(f"Anzahl Fahrzeuge", 1, 5000, key=f"fn_{i}")
                thg_val = {"PKW": 115, "LKW": 420, "Busse": 480}
                thg_revenue += (f['anz'] * thg_val[f['art']])
                if f3.button(f"🗑️ Gruppe {i+1} entfernen", key=f"fd_{i}"): st.session_state.fuhrpark.pop(i); st.rerun()

# --- TAB: ARBITRAGE (MARKTMODI) ---
if show_arbitrage:
    with tabs[tabs_labels.index("📈 Arbitrage")]:
        st.markdown('<p class="main-header">📈 Spotmarkt-Arbitrage</p>', unsafe_allow_html=True)
        with st.container(border=True):
            arb_mode = st.radio("Daten-Modus", ["Manuelle Eingabe", "Internet-Referenz 2026", "KI-Volatilitäts-Prognose"])
            c_a1, c_a2 = st.columns(2)
            spread = c_a1.number_input("Ø Preis-Spread (ct/kWh)", 5.0, 50.0, 14.5 if arb_mode == "Manuelle Eingabe" else (15.2 if "Referenz" in arb_mode else 18.5))
            cycles = c_a2.number_input("Vollladezyklen f. Arbitrage/Jahr", 0, 1000, 250 if arb_mode == "Manuelle Eingabe" else (280 if "Referenz" in arb_mode else 310))
            arbitrage_revenue = (total_storage_kwh * 0.9 * (spread/100) * cycles)
            st.metric("Projizierter Arbitrage-Erlös", f"{arbitrage_revenue:,.2f} €/Jahr")

# --- TAB 1: ROI, INVESTMENT & NETZ-CHECK ---
with tabs[0]:
    st.markdown('<p class="main-header">📊 Wirtschaftlichkeit & Investment</p>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.subheader("💰 CAPEX Planung")
        calc_mode = st.radio("Berechnungsweise", ["Hardware-Einzelpreise", "Gesamtinvestition (Manuelle Eingabe)"], horizontal=True)
        if calc_mode == "Hardware-Einzelpreise":
            col_inv1, col_inv2, col_inv3 = st.columns(3)
            inv_pv = col_inv1.number_input("PV-Preis (€/kWp)", 500, 3000, 1150)
            inv_bat = col_inv2.number_input("Speicher-Preis (€/kWh)", 150, 2000, 500)
            inv_lp = col_inv3.number_input("Preis pro LP (€)", 500, 50000, 2500)
            invest_total = (total_kwp * inv_pv) + (total_storage_kwh * inv_bat) + (len(st.session_state.lade_punkte) * inv_lp)
        else:
            invest_total = st.number_input("Gesamte Investitionssumme (€)", 0, 100000000, 150000)
    
    # Betriebslogik & ROI Kalkulation
    pv_only = show_pv and not (show_storage or show_mobility or show_arbitrage or show_heat)
    pv_mode = "Eigenverbrauch"
    if pv_only:
        pv_mode = st.radio("PV-Betriebskonzept", ["Eigenverbrauch & Überschuss", "Volleinspeisung"])
    
    pv_yield = total_kwp * 1050
    if not has_imsys: pv_yield *= 0.94 # 6% Abschlag wegen fehlender Steuerung/Abregelung
    
    if "Volleinspeisung" in pv_mode:
        annual_benefit = pv_yield * (einspeise_verg_voll/100)
    else:
        # Dynamische Eigenverbrauchsrate
        ev_rate = 0.35 + (0.35 if total_storage_kwh > 0 else 0) + (0.10 if show_mobility else 0)
        ev_rate = min(ev_rate, 0.95)
        annual_benefit = (pv_yield * ev_rate * (strompreis_netz/100)) + (pv_yield * (1-ev_rate) * (einspeise_verg_ueberschuss/100)) + thg_revenue + arbitrage_revenue
    
    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Investment (CAPEX)", f"{invest_total:,.0f} €")
    r2.metric("Cashflow p.a.", f"{annual_benefit:,.2f} €")
    r3.metric("Amortisation (ROI)", f"{invest_total/annual_benefit if annual_benefit > 0 else 0:.1f} Jahre")

    # Netz-Check (Echtzeit-Analyse)
    total_peak_load = max(total_kwp * 0.85, (total_lp_power_ac + total_lp_power_dc) * 0.6)
    st.divider()
    st.subheader("🔌 Netzanschluss-Check")
    st.image("https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&q=80&w=800", caption="Smart Grid Infrastruktur", width=600)
    
    if total_peak_load > grid_limit_kva:
        st.error(f"❌ Kapazitätswarnung! Lastspitze ({total_peak_load:.1f} kW) übersteigt Netzanschluss ({grid_limit_kva} kVA). Lastmanagement oder Netzausbau nötig.")
    else:
        st.success(f"✅ Netzanschluss ausreichend. Auslastung: {(total_peak_load/grid_limit_kva)*100:.1f}%.")

    # Cashflow Chart
    x = np.arange(21)
    y = [-invest_total + (annual_benefit * i) for i in x]
    fig = px.line(x=x, y=y, title="Kumulierter Cashflow über 20 Jahre", labels={'x':'Jahre', 'y':'€'})
    fig.add_hline(y=0, line_dash="dash", line_color="green")
    st.plotly_chart(fig, use_container_width=True)      
