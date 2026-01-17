import streamlit as st
import json
import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime

# =================================================================
# 1. KONFIGURATION & DATEI-MANAGEMENT (DSGVO-KONFORM LOKAL)
# =================================================================
if "active_slug" not in st.session_state:
    st.error("⚠️ Kein aktives Projekt geladen. Bitte Projekt in der Übersicht wählen.")
    st.stop()

P_SLUG = st.session_state["active_slug"]
P_PATH = os.path.join("projects", P_SLUG)
STATE_FILE = os.path.join(P_PATH, "state.json")
DOC_PATH = os.path.join(P_PATH, "documents")
os.makedirs(DOC_PATH, exist_ok=True)

# Daten laden & Migration
with open(STATE_FILE, "r", encoding="utf-8") as f:
    d = json.load(f)

# Struktur-Check (Deep Initialization)
sections = {
    "pv": {"felder": [], "wr": [], "kabel_verlust": 2.0, "degradation": 0.5},
    "speicher": {"kap": 0.0, "p": 0.0, "zyklen": 6000, "dod": 90, "eff": 95, "temp_faktor": 1.0},
    "mobilität": {"lp": [], "fuhrpark": []},
    "finanzen": {"invest": 0.0, "strom_preis": 0.32, "einspeisung": 0.08, "diesel_preis": 1.78, "wartung_pa": 1.5}
}
for key, val in sections.items():
    if key not in d: d[key] = val

def save_state():
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4, ensure_ascii=False)

def upload_handler(file, label):
    if file:
        fname = f"{label}_{file.name.replace(' ', '_')}"
        with open(os.path.join(DOC_PATH, fname), "wb") as f:
            f.write(file.getbuffer())
        return fname
    return None

st.set_page_config(page_title="Professional Energy Engineering", layout="wide")

# =================================================================
# 2. KI-INTEGRATION & MARKT-DATABASE 2026
# =================================================================
# Diese Daten dienen als Basis für die automatisierten Rechner
MARKET_DATA = {
    "THG_2026": {"PKW": 165.0, "N1": 280.0, "N2": 3400.0, "N3": 5800.0, "Bus": 12500.0},
    "CONSUMPTION": {
        "PKW": {"e": 18.2, "d": 6.4, "service_e": 350, "service_d": 750, "tax_d": 250},
        "N1": {"e": 27.5, "d": 9.2, "service_e": 500, "service_d": 1100, "tax_d": 400},
        "N2": {"e": 72.0, "d": 17.5, "service_e": 1200, "service_d": 2800, "tax_d": 1200},
        "N3": {"e": 118.0, "d": 31.0, "service_e": 2500, "service_d": 6500, "tax_d": 3500},
        "Bus": {"e": 132.0, "d": 35.5, "service_e": 3000, "service_d": 8000, "tax_d": 1500}
    },
    "SUN_HOURS": {"Nord": 940, "Süd": 1180, "Ost": 1050, "West": 1020}
}

# =================================================================
# 3. MATHEMATISCHE CORE-FUNKTIONEN (THE RECKONER)
# =================================================================
def calc_lcoe(invest, yield_pa, years=20):
    """Berechnet die Stromgestehungskosten (Levelized Cost of Energy)."""
    # Vereinfachte Formel für LCOE
    total_yield = sum([yield_pa * ((1 - (d['pv']['degradation']/100))**y) for y in range(years)])
    return invest / total_yield if total_yield > 0 else 0

def calc_npv(invest, cashflows, rate=0.04):
    """Berechnet den Kapitalwert (Net Present Value)."""
    npv = -invest
    for t, cf in enumerate(cashflows):
        npv += cf / ((1 + rate) ** (t + 1))
    return npv

# =================================================================
# 4. BENUSEROBERFLÄCHE (UI)
# =================================================================
st.title("🛡️ Enterprise Fachplanung & Simulation")
st.subheader(f"Projekt: {d['metadata']['id']} | Kunde: {d['kunde']['name']}")

# SIDEBAR FÜR GLOBALE PARAMETER
with st.sidebar:
    st.header("⚙️ Globale Parameter")
    d['finanzen']['strom_preis'] = st.number_input("Strompreis (Bezug) €/kWh", 0.0, 1.0, d['finanzen']['strom_preis'])
    d['finanzen']['diesel_preis'] = st.number_input("Dieselpreis €/L", 0.0, 5.0, d['finanzen']['diesel_preis'])
    d['finanzen']['zins_satz'] = st.slider("Kalk. Zinssatz (%)", 0.0, 10.0, 4.0) / 100
    if st.button("💾 Stand jetzt sichern"):
        save_state()
        st.success("Lokal gespeichert.")

tabs = st.tabs(["☀️ PV-Engineering", "🔋 Speicher-Setup", "🔌 Mobilität & Fuhrpark", "📊 Ertrags-Analyse", "💰 ROI & TCO"])

# -----------------------------------------------------------------
# TAB 1: PV-ENGINEERING
# -----------------------------------------------------------------
with tabs[0]:
    st.header("Photovoltaik-Systemdesign")
    c_p1, c_p2 = st.columns([2, 1])
    
    with c_p1:
        st.markdown("### Modulfelder (Generatoren)")
        if st.button("➕ Neues Modulfeld"):
            d['pv']['felder'].append({"h": "Hersteller", "t": "Typ", "wp": 440, "n": 24, "az": 0, "ne": 35})
        
        for i, f in enumerate(d['pv']['felder']):
            with st.container(border=True):
                col1, col2, col3 = st.columns([2, 1, 1])
                f['h'] = col1.text_input("Hersteller", f['h'], key=f"pvh_{i}")
                f['t'] = col2.text_input("Typ", f['t'], key=f"pvt_{i}")
                f['wp'] = col3.number_input("Leistung (Wp)", 100, 800, f['wp'], key=f"pvw_{i}")
                
                col4, col5, col6, col7 = st.columns(4)
                f['n'] = col4.number_input("Anzahl", 1, 10000, f['n'], key=f"pvn_{i}")
                f['az'] = col5.number_input("Azimut (°)", -180, 180, f['az'], key=f"pvaz_{i}")
                f['ne'] = col6.number_input("Neigung (°)", 0, 90, f['ne'], key=f"pvne_{i}")
                if col7.button("🗑️", key=f"pvdel_{i}"):
                    d['pv']['felder'].pop(i); st.rerun()
                
                up_mod = st.file_uploader("Datenblatt Modul", type=["pdf"], key=f"pvup_{i}")
                if up_mod: f['db'] = upload_handler(up_mod, f"Modul_Feld_{i}")

    with c_p2:
        st.markdown("### Wechselrichter & Verluste")
        if st.button("➕ Wechselrichter"):
            d['pv']['wr'].append({"h": "Brand", "p": 20.0, "n": 1})
        for j, w in enumerate(d['pv']['wr']):
            with st.container(border=True):
                w['h'] = st.text_input("Hersteller/Modell", w['h'], key=f"wrh_{j}")
                c_w1, c_w2 = st.columns(2)
                w['p'] = c_w1.number_input("Leistung (kW)", 0.1, 2000.0, w['p'], key=f"wrp_{j}")
                w['n'] = c_w2.number_input("Stück", 1, 100, w['n'], key=f"wrn_{j}")
                if st.button("🗑️", key=f"wrdel_{j}"):
                    d['pv']['wr'].pop(j); st.rerun()
        
        st.divider()
        d['pv']['kabel_verlust'] = st.slider("Kabelverluste AC/DC (%)", 0.5, 5.0, d['pv']['kabel_verlust'])
        d['pv']['degradation'] = st.slider("Jährliche Degradation (%)", 0.1, 1.0, d['pv']['degradation'])

# -----------------------------------------------------------------
# TAB 2: SPEICHER-SETUP
# -----------------------------------------------------------------
with tabs[1]:
    st.header("Batteriespeicher-Spezifikation")
    
    s = d['speicher']
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        s['h'] = st.text_input("Hersteller", s.get('h', ''))
        s['kap'] = st.number_input("Nettokapazität (kWh)", 0.0, 10000.0, s['kap'])
        s['p'] = st.number_input("Max. Ladeleistung (kW)", 0.0, 5000.0, s['p'])
        s_up = st.file_uploader("Datenblatt Speicher", type=["pdf"])
        if s_up: s['db'] = upload_handler(s_up, "Speicher")
    with c_s2:
        s['zyklen'] = st.number_input("Garantierte Vollzyklen", 0, 15000, s['zyklen'])
        s['dod'] = st.slider("Entladetiefe (DoD) in %", 10, 100, s['dod'])
        s['eff'] = st.slider("Wirkungsgrad Roundtrip (%)", 70, 99, s['eff'])
        
    st.info(f"💡 Bei 280 Zyklen/Jahr beträgt die erwartete Lebensdauer ca. {s['zyklen']/280:.1f} Jahre.")

# -----------------------------------------------------------------
# TAB 3: MOBILITÄT & FUHRPARK (SUBSTITUTION)
# -----------------------------------------------------------------
with tabs[2]:
    st.header("Flottenmanagement & Diesel-Substitution")
    
    col_l, col_f = st.columns([1, 2])
    
    with col_l:
        st.subheader("Ladepunkte")
        if st.button("➕ Neuer Ladepunkt"):
            d['mobilität']['lp'].append({"t": "AC", "p": 11, "n": 1})
        for k, lp in enumerate(d['mobilität']['lp']):
            with st.container(border=True):
                lp['t'] = st.selectbox("Typ", ["AC", "DC"], key=f"lpt_{k}")
                c_lp1, c_lp2 = st.columns(2)
                lp['p'] = c_lp1.number_input("Leistung (kW)", 1, 400, lp['p'], key=f"lpp_{k}")
                lp['n'] = c_lp2.number_input("Stück", 1, 100, lp['n'], key=f"lpn_{k}")
                if st.button("🗑️", key=f"lpdel_{k}"):
                    d['mobilität']['lp'].pop(k); st.rerun()

    with col_f:
        st.subheader("Fuhrpark-Rechner")
        if st.button("➕ Fahrzeuggruppe"):
            d['mobilität']['fuhrpark'].append({"k": "PKW", "n": 1, "km": 20000, "e_c": 18.0, "d_c": 6.0})
        
        for f_idx, fz in enumerate(d['mobilität']['fuhrpark']):
            with st.container(border=True):
                cf1, cf2, cf3 = st.columns([2, 1, 1])
                fz['k'] = cf1.selectbox("Klasse", list(MARKET_DATA["CONSUMPTION"].keys()), key=f"fzk_{f_idx}")
                fz['n'] = cf2.number_input("Anzahl", 1, 1000, fz['n'], key=f"fzn_{f_idx}")
                fz['km'] = cf3.number_input("km / Jahr", 1, 500000, fz['km'], key=f"fzm_{f_idx}")
                
                # KI RECHNER INTEGRATION
                if st.button(f"🤖 KI-Werte für {fz['k']} laden", key=f"fzki_{f_idx}"):
                    base = MARKET_DATA["CONSUMPTION"][fz['k']]
                    fz['e_c'], fz['d_c'] = base['e'], base['d']
                    st.rerun()
                
                ce1, ce2 = st.columns(2)
                fz['e_c'] = ce1.number_input("Verbrauch Elektro (kWh/100km)", 0.0, 300.0, fz['e_c'], key=f"fzec_{f_idx}")
                fz['d_c'] = ce2.number_input("Verbrauch Diesel (L/100km)", 0.0, 100.0, fz['d_c'], key=f"fzdc_{f_idx}")
                
                # Interne Live-Kalkulation
                d_cost = (fz['km']/100 * fz['d_c'] * d['finanzen']['diesel_preis']) + MARKET_DATA["CONSUMPTION"][fz['k']]['service_d'] + MARKET_DATA["CONSUMPTION"][fz['k']]['tax_d']
                e_cost = (fz['km']/100 * fz['e_c'] * d['finanzen']['strom_preis']) + MARKET_DATA["CONSUMPTION"][fz['k']]['service_e']
                st.write(f"**TCO Einsparung:** { (d_cost - e_cost) * fz['n'] :,.2f} € / Jahr")

# -----------------------------------------------------------------
# TAB 4: ERTRAGS-ANALYSE (SIMULATION)
# -----------------------------------------------------------------
with tabs[3]:
    st.header("Standort-Simulation & Arbitrage")
    c_e1, c_e2 = st.columns(2)
    
    with c_e1:
        region = st.selectbox("Wetter-Region", list(MARKET_DATA["SUN_HOURS"].keys()))
        sun = MARKET_DATA["SUN_HOURS"][region]
        st.metric("Spez. Ertrag (Simuliert)", f"{sun} kWh/kWp")
        
        # PV-Gen Berechnung
        total_kwp = sum([(f['wp'] * f['n']) / 1000 for f in d['pv']['felder']])
        raw_yield = total_kwp * sun * (1 - (d['pv']['kabel_verlust']/100))
        st.write(f"Voraussichtlicher Jahresertrag: **{raw_yield:,.0f} kWh**")

    with c_e2:
        st.subheader("Börsenhandel (Arbitrage)")
        arb_on = st.toggle("Arbitrage-Modus (EPEX Spot)", False)
        spread = st.slider("Ø Preis-Spread (ct/kWh)", 5.0, 40.0, 12.0)
        potential = (d['speicher']['kap'] * 280 * (spread/100)) if arb_on else 0
        st.metric("Arbitrage-Potenzial", f"{potential:,.2f} € / Jahr")

# -----------------------------------------------------------------
# TAB 5: ROI & TCO (DIE RECHNUNG)
# -----------------------------------------------------------------
with tabs[4]:
    st.header("Wirtschaftlichkeit (Enterprise Reporting)")
    
    invest = st.number_input("Gesamtinvestition Netto (€)", 0.0, 10000000.0, d['finanzen']['invest'])
    d['finanzen']['invest'] = invest
    
    # 20-Jahre Cashflow Simulation
    cf_list = []
    for y in range(20):
        # Degradierten Ertrag rechnen
        y_yield = raw_yield * ((1 - (d['pv']['degradation']/100))**y)
        # Einsparung aus Fuhrpark
        f_save = 0
        for fz in d['mobilität']['fuhrpark']:
            d_tco = (fz['km']/100 * fz['d_c'] * d['finanzen']['diesel_preis']) + MARKET_DATA["CONSUMPTION"][fz['k']]['service_d'] + MARKET_DATA["CONSUMPTION"][fz['k']]['tax_d']
            e_tco = (fz['km']/100 * fz['e_c'] * d['finanzen']['strom_preis']) + MARKET_DATA["CONSUMPTION"][fz['k']]['service_e']
            f_save += (d_tco - e_tco) * fz['n']
        
        # THG Quoten
        thg_save = sum([MARKET_DATA["THG_2026"].get(fz['k'], 0) * fz['n'] for fz in d['mobilität']['fuhrpark']])
        
        yearly_cf = f_save + thg_save + potential + (y_yield * 0.08) - (invest * (d['finanzen']['wartung_pa']/100))
        cf_list.append(yearly_cf)
    
    # KPIs
    st.divider()
    res1, res2, res3 = st.columns(3)
    
    lcoe = calc_lcoe(invest, raw_yield)
    res1.metric("LCOE (Stromkosten)", f"{lcoe:.4f} €/kWh")
    
    npv_val = calc_npv(invest, cf_list, rate=d['finanzen']['zins_satz'])
    res2.metric("Net Present Value (NPV)", f"{npv_val:,.2f} €", delta="Positiv" if npv_val > 0 else "Negativ")
    
    if sum(cf_list) > 0:
        payback = invest / (sum(cf_list)/20)
        res3.metric("Amortisation (Statisch)", f"{payback:.1f} Jahre")

    st.subheader("Kumulierter Cashflow (20 Jahre)")
    st.line_chart(np.cumsum([-invest] + cf_list))

# FINALER SAVE
st.divider()
if st.button("💾 PROJEKT-DATEN FINALISIEREN", use_container_width=True, type="primary"):
    save_state()
    st.balloons()
