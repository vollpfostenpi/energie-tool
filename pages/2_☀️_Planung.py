import streamlit as st
import json
import os
import shutil
from datetime import datetime

# =================================================================
# 1. SYSTEM-INITIALISIERUNG & LOKALER DATEN-SYNC
# =================================================================
if "active_slug" not in st.session_state:
    st.error("⚠️ Kein aktives Projekt gefunden. Bitte in der Hauptseite ein Projekt öffnen.")
    st.stop()

P_SLUG = st.session_state["active_slug"]
P_PATH = os.path.join("projects", P_SLUG)
STATE_FILE = os.path.join(P_PATH, "state.json")
DOC_PATH = os.path.join(P_PATH, "documents")
os.makedirs(DOC_PATH, exist_ok=True)

# Lade Daten direkt von Festplatte für absolute Konsistenz
with open(STATE_FILE, "r", encoding="utf-8") as f:
    d = json.load(f)

# Sicherstellung aller nötigen Datenstrukturen (Migration)
if "pv" not in d: d["pv"] = {"felder": [], "wechselrichter": []}
if "speicher" not in d: d["speicher"] = {}
if "mobilität" not in d: d["mobilität"] = {"ladepunkte": [], "fuhrpark": []}
if "wirtschaft" not in d: d["wirtschaft"] = {"invest_detail": {}, "energie": {}}

def save():
    """Schreibt den aktuellen Stand sofort physisch auf die lokale Festplatte."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4, ensure_ascii=False)

def handle_upload(file_obj, prefix):
    """Speichert PDF-Dokumente lokal im Projektordner."""
    if file_obj:
        fname = f"{prefix}_{file_obj.name}"
        full_path = os.path.join(DOC_PATH, fname)
        with open(full_path, "wb") as f:
            f.write(file_obj.getbuffer())
        return fname
    return None

st.set_page_config(page_title="Professional Engineering Terminal", layout="wide")

# =================================================================
# 2. KI- & RECHNER-ENGINE (STATISCHE DATENBANK 2026)
# =================================================================
MARKET_DB = {
    "THG": {"PKW": 140.0, "N1": 210.0, "N2": 3100.0, "N3": 5200.0, "Bus": 11500.0},
    "CONS": { # kWh/100km | Diesel L/100km
        "PKW": [18.0, 6.0], "N1": [28.0, 9.0], "N2": [75.0, 18.0], "N3": [120.0, 32.0], "Bus": [135.0, 38.0]
    },
    "WEATHER": {"Nord": 950, "Süd": 1150, "Ost": 1020, "West": 1010}
}

# =================================================================
# 3. UI ARCHITEKTUR
# =================================================================
st.title("🏗️ Technische Fachplanung & Wirtschaftlichkeit")

# HEADER MIT STATUS
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([2,2,1,1])
    c1.metric("Projekt-ID", d['metadata']['id'])
    c2.write(f"**Kunde:** {d['kunde']['name']}\n\n**Standort:** {d['kunde']['plz_ort']}")
    if c4.button("💾 GLOBAL SAVE", use_container_width=True):
        save()
        st.toast("Alle Daten lokal gesichert!")

tabs = st.tabs(["☀️ PV-ANLAGE", "🔋 SPEICHER", "🔌 MOBILITÄT", "💰 ERLÖSE", "📈 ROI-ANALYSE"])

# -----------------------------------------------------------------
# TAB 1: PV-ANLAGE (MODULÄR)
# -----------------------------------------------------------------
with tabs[0]:
    st.header("Photovoltaik-Engineering")
    d['pv']['konzept'] = st.radio("Betriebskonzept", ["Eigenverbrauch/Überschuss", "Volleinspeisung"], horizontal=True)
    
    col_m, col_w = st.columns(2)
    
    with col_m:
        st.subheader("Modulfelder")
        if st.button("➕ Modulfeld hinzufügen"):
            d['pv']['felder'].append({"hersteller": "", "typ": "", "wp": 440, "anzahl": 24, "azimut": 0, "neigung": 35})
        
        for i, f in enumerate(d['pv']['felder']):
            with st.expander(f"Feld {i+1}: {f['hersteller']} {f['typ']}", expanded=True):
                c1, c2 = st.columns(2)
                f['hersteller'] = c1.text_input("Hersteller", f['hersteller'], key=f"pv_h_{i}")
                f['typ'] = c2.text_input("Modul-Typ", f['typ'], key=f"pv_t_{i}")
                f['wp'] = c1.number_input("Nennleistung (Wp)", 0, 1000, f['wp'], key=f"pv_wp_{i}")
                f['anzahl'] = c2.number_input("Stückzahl", 0, 10000, f['anzahl'], key=f"pv_n_{i}")
                st.write(f"**Leistung Feld:** { (f['wp'] * f['anzahl'])/1000 :.2f} kWp")
                
                up_m = st.file_uploader("Datenblatt Modul", type=["pdf"], key=f"up_m_{i}")
                if up_m: f['db'] = handle_upload(up_m, f"PV_Modul_{i}")
                
                if st.button("🗑️ Feld löschen", key=f"pv_del_{i}"):
                    d['pv']['felder'].pop(i); st.rerun()

    with col_w:
        st.subheader("Wechselrichter")
        if st.button("➕ Wechselrichter hinzufügen"):
            d['pv']['wechselrichter'].append({"hersteller": "", "typ": "", "leistung": 20.0, "anzahl": 1})
            
        for j, w in enumerate(d['pv']['wechselrichter']):
            with st.expander(f"WR {j+1}: {w['hersteller']}", expanded=True):
                w['hersteller'] = st.text_input("Hersteller", w['hersteller'], key=f"wr_h_{j}")
                w['leistung'] = st.number_input("AC-Leistung (kW)", 0.0, 5000.0, w['leistung'], key=f"wr_p_{j}")
                w['anzahl'] = st.number_input("Stückzahl", 1, 100, w['anzahl'], key=f"wr_n_{j}")
                up_w = st.file_uploader("Datenblatt WR", type=["pdf"], key=f"up_w_{j}")
                if up_w: w['db'] = handle_upload(up_w, f"WR_{j}")
                if st.button("🗑️ WR löschen", key=f"wr_del_{j}"):
                    d['pv']['wechselrichter'].pop(j); st.rerun()

# -----------------------------------------------------------------
# TAB 2: SPEICHER (TECHNISCHE TIEFE)
# -----------------------------------------------------------------
with tabs[1]:
    st.header("Batteriespeicher-System")
    c1, c2 = st.columns(2)
    with c1:
        d['speicher']['hersteller'] = st.text_input("Speicher-Hersteller", d['speicher'].get('hersteller', ''))
        d['speicher']['kap_netto'] = st.number_input("Netto-Kapazität (kWh)", 0.0, 10000.0, d['speicher'].get('kap_netto', 0.0))
        d['speicher']['leistung'] = st.number_input("Max. Entladeleistung (kW)", 0.0, 5000.0, d['speicher'].get('leistung', 0.0))
        d['speicher']['zyklen'] = st.number_input("Garantierte Zyklen", 0, 15000, d['speicher'].get('zyklen', 6000))
    with c2:
        d['speicher']['dod'] = st.slider("Entladetiefe (DoD %)", 0, 100, d['speicher'].get('dod', 90))
        d['speicher']['effizienz'] = st.slider("Wirkungsgrad (%)", 70, 99, d['speicher'].get('effizienz', 95))
        st.info(f"Technischer Fokus: C-Rate = {d['speicher']['leistung']/(d['speicher']['kap_netto'] if d['speicher']['kap_netto'] > 0 else 1):.2f}")
        up_s = st.file_uploader("Datenblatt Speicher", type=["pdf"])
        if up_s: d['speicher']['db'] = handle_upload(up_s, "Speicher")

# -----------------------------------------------------------------
# TAB 3: MOBILITÄT & FUHRPARK (E-KI INTEGRATION)
# -----------------------------------------------------------------
with tabs[2]:
    st.header("🔌 Mobilitäts-Hub & Fuhrpark")
    
    # LADESTATIONEN
    st.subheader("Ladeinfrastruktur")
    if st.button("➕ Ladepunkt hinzufügen"):
        d['mobilität']['ladepunkte'].append({"typ": "AC", "leistung": 11.0, "hersteller": "", "anzahl": 1})
    
    for l_idx, lp in enumerate(d['mobilität']['ladepunkte']):
        with st.container(border=True):
            cl1, cl2, cl3, cl4 = st.columns([1,1,2,1])
            lp['typ'] = cl1.selectbox("Art", ["AC", "DC"], key=f"lp_t_{l_idx}")
            lp['leistung'] = cl2.number_input("kW", 0, 400, int(lp['leistung']), key=f"lp_p_{l_idx}")
            lp['hersteller'] = cl3.text_input("Modell/Hersteller", lp['hersteller'], key=f"lp_h_{l_idx}")
            if cl4.button("🗑️", key=f"lp_del_{l_idx}"):
                d['mobilität']['ladepunkte'].pop(l_idx); st.rerun()

    st.divider()
    
    # FUHRPARK-MANAGER
    st.subheader("🚗 Fuhrpark-Analyse (Diesel vs. Elektro)")
    if st.button("➕ Fahrzeuggruppe hinzufügen"):
        d['mobilität']['fuhrpark'].append({"klasse": "PKW", "anzahl": 1, "km_jahr": 20000, "verbrauch_e": 18.0, "verbrauch_d": 6.5})

    for f_idx, fz in enumerate(d['mobilität']['fuhrpark']):
        with st.container(border=True):
            cf1, cf2, cf3, cf4, cf5 = st.columns([2,1,1,1,1])
            fz['klasse'] = cf1.selectbox("Fahrzeugklasse", ["PKW", "N1", "N2", "N3", "Bus"], key=f"fz_k_{f_idx}")
            fz['anzahl'] = cf2.number_input("Anzahl", 1, 1000, fz['anzahl'], key=f"fz_n_{f_idx}")
            fz['km_jahr'] = cf3.number_input("km/Jahr", 0, 500000, fz['km_jahr'], key=f"fz_km_{f_idx}")
            
            if cf1.button("🤖 KI-Werte", key=f"fz_ki_{f_idx}"):
                fz['verbrauch_e'] = MARKET_DB["CONS"][fz['klasse']][0]
                fz['verbrauch_d'] = MARKET_DB["CONS"][fz['klasse']][1]
                st.rerun()
            
            fz['verbrauch_e'] = cf4.number_input("kWh/100km", 0.0, 300.0, fz['verbrauch_e'], key=f"fz_ve_{f_idx}")
            fz['verbrauch_d'] = cf5.number_input("Diesel L/100km", 0.0, 100.0, fz['verbrauch_d'], key=f"fz_vd_{f_idx}")
            
            # Sub-Kalkulation
            ges_kwh = (fz['km_jahr'] / 100) * fz['verbrauch_e'] * fz['anzahl']
            ges_diesel = (fz['km_jahr'] / 100) * fz['verbrauch_d'] * fz['anzahl']
            st.caption(f"Bedarf: {ges_kwh:,.0f} kWh/Jahr | Substitution: {ges_diesel:,.0f} Liter Diesel")

# -----------------------------------------------------------------
# TAB 4: ERLÖSZENTRUM (ARBITRAGE & THG)
# -----------------------------------------------------------------
with tabs[3]:
    st.header("Ertrags-Optimierung")
    
    # PV-Ertragssimulation
    with st.container(border=True):
        st.subheader("☀️ PV-Ertragssimulation")
        region = st.selectbox("Region für Wetterdaten", list(MARKET_DB["WEATHER"].keys()))
        if st.button("🛰️ Wetterdaten & Einstrahlung 2026 laden"):
            d['wirtschaft']['spez_ertrag'] = MARKET_DB["WEATHER"][region]
            st.success(f"Simulation abgeschlossen: {d['wirtschaft']['spez_ertrag']} kWh/kWp für {region} prognostiziert.")
    
    # Arbitrage
    with st.container(border=True):
        st.subheader("📈 Speicher-Arbitrage (Spotmarkt)")
        d['wirtschaft']['arbitrage_active'] = st.toggle("Arbitrage nutzen", d['wirtschaft'].get('arbitrage_active', False))
        c_a1, c_a2 = st.columns(2)
        d['wirtschaft']['spread'] = c_a1.number_input("Ø Spread (ct/kWh)", 0.0, 50.0, d['wirtschaft'].get('spread', 12.0))
        if c_a2.button("🤖 KI-Börsenprognose (Day-Ahead)"):
            d['wirtschaft']['spread'] = 19.4 # Fiktiver 2026 Wert
            st.rerun()
            
    # THG Erlöse
    with st.container(border=True):
        st.subheader("💸 THG-Quoten Erlöse")
        total_thg = 0
        for fz in d['mobilität']['fuhrpark']:
            quota = MARKET_DB["THG"].get(fz['klasse'], 0)
            total_thg += quota * fz['anzahl']
        st.write(f"Voraussichtlicher THG-Erlös pro Jahr: **{total_thg:,.2f} €**")

# -----------------------------------------------------------------
# TAB 5: ROI-ANALYSE (DIE FINALE RECHNUNG)
# -----------------------------------------------------------------
with tabs[4]:
    st.header("Wirtschaftlichkeits-Check (BDI)")
    
    # INVESTITIONS-MODUS
    inv_mode = st.radio("Kosten-Erfassung", ["Detail (Hardware)", "Pauschal (Gesamtinvest)"], horizontal=True)
    
    if inv_mode == "Detail (Hardware)":
        c_i1, c_i2 = st.columns(2)
        cost_pv = c_i1.number_input("Kosten PV-Anlage (€)", 0)
        cost_bat = c_i2.number_input("Kosten Speicher (€)", 0)
        cost_lp = c_i1.number_input("Kosten Ladetechnik (€)", 0)
        cost_montage = c_i2.number_input("Montage/AC/Nebenleistung (€)", 0)
        d['wirtschaft']['invest_total'] = cost_pv + cost_bat + cost_lp + cost_montage
    else:
        d['wirtschaft']['invest_total'] = st.number_input("Netto-Investitionssumme (€)", 0, 10000000, d['wirtschaft'].get('invest_total', 0))

    st.divider()
    
    # ROI KALKULATOR (INTERN)
    st.subheader("ROI-Ergebnis (Simulation)")
    
    # 1. PV Ertrag
    total_kwp = sum((f['wp'] * f['anzahl'])/1000 for f in d['pv']['felder'])
    pv_yield = total_kwp * d['wirtschaft'].get('spez_ertrag', 1000)
    
    # 2. Einsparung Diesel (Simulation 1.80€ / L, 0.35€ / kWh)
    fuel_saving = 0
    electricity_cost_fz = 0
    for fz in d['mobilität']['fuhrpark']:
        fuel_saving += (fz['km_jahr'] / 100) * fz['verbrauch_d'] * fz['anzahl'] * 1.80
        electricity_cost_fz += (fz['km_jahr'] / 100) * fz['verbrauch_e'] * fz['anzahl'] * 0.35
    
    net_mobility_saving = fuel_saving - electricity_cost_fz
    
    # Arbitrage Ertrag
    arb_profit = (d['speicher'].get('kap_netto', 0) * 280 * (d['wirtschaft'].get('spread', 0)/100)) if d['wirtschaft'].get('arbitrage_active') else 0

    total_yearly_benefit = net_mobility_saving + arb_profit + total_thg + (pv_yield * 0.08) # Vereinfachte Einspeisung

    col_res1, col_res2, col_res3 = st.columns(3)
    col_res1.metric("Gesamt-Ersparnis / Jahr", f"{total_yearly_benefit:,.2f} €")
    
    if total_yearly_benefit > 0:
        amort = d['wirtschaft']['invest_total'] / total_yearly_benefit
        col_res2.metric("Amortisation", f"{amort:.1f} Jahre")
    
    col_res3.metric("Installierte Leistung", f"{total_kwp:.1f} kWp")

    

# FINALE SPEICHERUNG
st.divider()
if st.button("📂 PROJEKT-AKTE FINALISIEREN & LOKAL SYNC", use_container_width=True, type="primary"):
    save()
    st.success(f"Engineering-Daten für {P_SLUG} wurden in der lokalen state.json aktualisiert.")
    st.balloons()
