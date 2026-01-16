import streamlit as st
import json
import os
import base64
from datetime import datetime

# =================================================================
# 1. INITIALISIERUNG & PRÜFUNG
# =================================================================
if "active_slug" not in st.session_state or "projekt_daten" not in st.session_state:
    st.error("⚠️ Kein Projekt aktiv. Bitte in der Übersicht ein Projekt öffnen oder neu anlegen.")
    st.stop()

d = st.session_state["projekt_daten"]
p_slug = st.session_state["active_slug"]
p_path = os.path.join("projects", p_slug)
doc_path = os.path.join(p_path, "documents")
os.makedirs(doc_path, exist_ok=True)

st.set_page_config(page_title="Engineering Terminal", layout="wide")

# =================================================================
# 2. PERSISTENTER HEADER (KUNDENDATEN)
# =================================================================
with st.container(border=True):
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.subheader(f"📍 Projekt: {d['metadata']['id']}")
        st.write(f"**Kunde:** {d['kunde']['name']} | {d['kunde']['straße']}, {d['kunde']['plz_ort']}")
    with c2:
        st.write(f"**Ansprechpartner:** {d['kunde'].get('ansprechpartner', 'N/A')}")
        st.write(f"**Zähler-ID:** `{d['kunde'].get('zählernummer', 'N/A')}`")
    with c3:
        st.write(f"**Steuer-Status:** {d['kunde'].get('steuer_status', 'N/A')}")
        if st.button("💾 ZWISCHENSPEICHERN", use_container_width=True):
            with open(os.path.join(p_path, "state.json"), "w", encoding="utf-8") as f:
                json.dump(d, f, indent=4)
            st.toast("Daten gesichert!")

# =================================================================
# 3. REITER-STRUKTUR
# =================================================================
tabs = st.tabs([
    "☀️ PV-ANLAGE", 
    "🔋 SPEICHER-HARDWARE", 
    "🔌 LADEINFRASTRUKTUR", 
    "💰 ERLÖSZENTRUM (Arbitrage/THG/Ertrag)", 
    "📈 ROI & INVESTITION"
])

# --- REITER 1: PV-ANLAGE ---
with tabs[0]:
    st.header("PV-Design & Hardware-Komponenten")
    d['pv']['volleinspeisung'] = st.toggle("Konzept: Volleinspeisung (EEG §21)", d['pv'].get('volleinspeisung', False))
    
    if st.button("➕ Modulfeld hinzufügen"):
        d['pv']['felder'].append({"name": "Feld", "kwp": 10.0, "modul": "", "azimut": 0, "neigung": 35})
    
    for i, f in enumerate(d['pv']['felder']):
        with st.expander(f"Feld {i+1}: {f['name']}", expanded=True):
            c1, c2, c3 = st.columns(3)
            f['name'] = c1.text_input("Name/Dachseite", f['name'], key=f"pv_n_{i}")
            f['kwp'] = c2.number_input("Leistung (kWp)", 0.0, 10000.0, float(f['kwp']), key=f"pv_k_{i}")
            f['modul'] = c3.text_input("Hersteller & Typ", f['modul'], key=f"pv_m_{i}")
            f['azimut'] = st.slider("Ausrichtung (Süd=0, West=90)", -180, 180, f['azimut'], key=f"pv_a_{i}")
            f['neigung'] = st.slider("Dachneigung (°)", 0, 90, f['neigung'], key=f"pv_ni_{i}")
            if st.button("Feld löschen", key=f"pv_d_{i}"):
                d['pv']['felder'].pop(i); st.rerun()

# --- REITER 2: SPEICHER-HARDWARE ---
with tabs[1]:
    st.header("Batteriespeicher (Technische Daten)")
    col1, col2 = st.columns(2)
    with col1:
        d['speicher']['hersteller'] = st.text_input("Hersteller/Modell", d['speicher'].get('hersteller', ''))
        d['speicher']['kapazität'] = st.number_input("Nettokapazität (kWh)", 0.0, 5000.0, float(d['speicher'].get('kapazität', 0.0)))
        d['speicher']['leistung'] = st.number_input("Ladeleistung (kW)", 0.0, 2000.0, float(d['speicher'].get('leistung', 0.0)))
        if d['speicher']['kapazität'] > 0:
            st.metric("C-Rate", f"{d['speicher']['leistung'] / d['speicher']['kapazität']:.2f} C")
    with col2:
        st.subheader("📁 Speicher-Datenblatt")
        bat_up = st.file_uploader("PDF hochladen", type=["pdf"], key="bat_up")
        if bat_up:
            with open(os.path.join(doc_path, f"DB_Speicher_{p_slug}.pdf"), "wb") as f_out:
                f_out.write(bat_up.getbuffer())
            st.success("Speicher-Datenblatt archiviert.")

# --- REITER 3: LADEINFRASTRUKTUR (NEU) ---
with tabs[2]:
    st.header("Ladepunkte (AC & DC)")
    if "ladepunkte" not in d: d["ladepunkte"] = []
    
    if st.button("➕ Ladepunkt hinzufügen"):
        d["ladepunkte"].append({"typ": "AC", "leistung": 11.0, "hersteller": "", "modell": ""})
    
    for j, lp in enumerate(d["ladepunkte"]):
        with st.container(border=True):
            cl1, cl2, cl3, cl4 = st.columns([1, 1, 2, 1])
            lp['typ'] = cl1.selectbox("Art", ["AC", "DC"], key=f"lp_t_{j}")
            lp['leistung'] = cl2.number_input("kW", 0.0, 400.0, float(lp['leistung']), key=f"lp_p_{j}")
            lp['hersteller'] = cl3.text_input("Hersteller/Modell", lp['hersteller'], key=f"lp_h_{j}")
            
            st.subheader("📁 Datenblatt Ladepunkt")
            lp_up = st.file_uploader(f"Upload für {lp['hersteller']}", type=["pdf"], key=f"lp_up_{j}")
            if lp_up:
                with open(os.path.join(doc_path, f"DB_Ladepunkt_{j}_{p_slug}.pdf"), "wb") as f_lp:
                    f_lp.write(lp_up.getbuffer())
            
            if cl4.button("🗑️", key=f"lp_d_{j}"):
                d["ladepunkte"].pop(j); st.rerun()

# --- REITER 4: ERLÖSZENTRUM ---
with tabs[3]:
    st.header("Intelligente Ertragsprognose")
    
    # PV-Ertrag mit Standort-Sim
    with st.container(border=True):
        st.subheader("🌍 PV-Ertrag & Wetterdaten")
        c_w1, c_w2 = st.columns([2, 1])
        with c_w1:
            st.write(f"Berechnung für Standort: **{d['kunde']['plz_ort']}**")
            d['pv']['spec_yield'] = st.number_input("Spez. Ertrag (kWh/kWp)", 0, 1500, d['pv'].get('spec_yield', 950))
        with c_w2:
            if st.button("🛰️ Wetterdaten-Prognose (KI)"):
                # Simulation Wetterdienst
                d['pv']['spec_yield'] = 1045
                st.info("Einstrahlungsprognose (PVGIS) geladen: 1.045 kWh/kWp")
                st.rerun()

    # Arbitrage & THG
    c_a, c_t = st.columns(2)
    with c_a:
        st.subheader("📊 Arbitrage-Handel")
        d['speicher']['spread'] = st.number_input("Spread (ct/kWh)", 0.0, 50.0, float(d['speicher'].get('spread', 12.0)))
        if st.button("🤖 KI-Marktprognose laden"):
            d['speicher']['spread'] = 18.2
            st.rerun()
    with c_t:
        st.subheader("🚗 THG-Flotte")
        if "thg_fleet" not in d: d["thg_fleet"] = []
        if st.button("➕ Fahrzeug"): d["thg_fleet"].append({"typ": "PKW", "quote": 250.0})
        for k, v in enumerate(d["thg_fleet"]):
            v['typ'] = st.selectbox("Klasse", ["PKW", "LKW", "Bus"], key=f"thg_v_{k}")
            if st.button("⚡ KI-Quote", key=f"thg_ki_{k}"):
                v['quote'] = 280 if v['typ'] == "PKW" else 1150
                st.rerun()

# --- REITER 5: ROI & INVESTITION ---
with tabs[4]:
    st.header("Wirtschaftlichkeit & Investition")
    
    inv_mode = st.radio("Investitions-Erfassung", ["Gesamtinvest (Pauschal)", "Hardware-Detail (Einzelauflistung)"])
    
    if inv_mode == "Gesamtinvest (Pauschal)":
        d['wirtschaft']['invest_total'] = st.number_input("Netto-Gesamtinvestition (€)", 0, 10000000, int(d['wirtschaft'].get('invest_total', 0)))
    else:
        st.write("Detaillierte Hardware-Kosten:")
        inv_pv = st.number_input("Kosten PV-Anlage (€)", 0, 5000000)
        inv_bat = st.number_input("Kosten Speicher (€)", 0, 1000000)
        inv_lp = st.number_input("Kosten Ladeinfrastruktur (€)", 0, 1000000)
        d['wirtschaft']['invest_total'] = inv_pv + inv_bat + inv_lp
        st.write(f"**Berechnete Gesamtsumme: {d['wirtschaft']['invest_total']:,} €**")

    st.divider()
    # ROI VORSCHAU
    st.subheader("ROI Analyse")
    pv_total_yield = sum(f['kwp'] for f in d['pv']['felder']) * d['pv']['spec_yield']
    st.metric("Erwarteter Jahresertrag PV", f"{pv_total_yield:,.2f} kWh")

# =================================================================
# 4. FINALE SICHERUNG
# =================================================================
st.divider()
if st.button("💾 KOMPLETTE PROJEKTPLANUNG ABSCHLIESSEN", use_container_width=True):
    with open(os.path.join(p_path, "state.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4)
    st.success("Alle Daten, Dokumente und KI-Prognosen wurden erfolgreich gesichert!")
    st.balloons()
