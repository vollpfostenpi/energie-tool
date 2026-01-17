import streamlit as st
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

# =================================================================
# 1. INITIALISIERUNG & PERSISTENZ-LAYER
# =================================================================
if "active_slug" not in st.session_state:
    st.error("⚠️ Kein aktives Projekt gefunden. Bitte im Dashboard starten.")
    st.stop()

P_SLUG = st.session_state["active_slug"]
P_PATH = os.path.join("projects", P_SLUG)
STATE_FILE = os.path.join(P_PATH, "state.json")
DOC_PATH = os.path.join(P_PATH, "documents")
os.makedirs(DOC_PATH, exist_ok=True)

# Lade Daten aus der JSON in den Session State
if "project_data" not in st.session_state:
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        st.session_state["project_data"] = json.load(f)

d = st.session_state["project_data"]

# Migration / Datenstruktur-Check (Verhindert KeyError bei Upgrades)
def ensure_keys(data, defaults):
    for k, v in defaults.items():
        if k not in data:
            data[k] = v
        elif isinstance(v, dict):
            ensure_keys(data[k], v)

defaults = {
    "scope": ["PV-System", "Speicher", "Ladeinfrastruktur"],
    "tech": {"hak_ampere": 63, "hak_kva": 43.6, "ems_needed": False, "mode": "Detail"},
    "dach": {"form": "Satteldach", "breite": 10.0, "tiefe": 6.0, "flaeche": 60.0, "neigung": 35, "azimut": 0},
    "pv": {"felder": [], "wr": [], "konzept": "Überschusseinspeisung", "total_kwp": 0.0},
    "speicher": {"kap": 0.0, "p": 0.0, "hersteller": "", "typ": "", "db_file": None},
    "mobilität": {"wallboxen": [], "fuhrpark": []},
    "wirtschaft": {"strompreis": 0.35, "dieselpreis": 1.75, "einspeise_v": 0.08, "last_jahr": 5000}
}
ensure_keys(d, defaults)

def save():
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4, ensure_ascii=False)

def upload_handler(file, prefix):
    if file:
        fname = f"{prefix}_{file.name.replace(' ', '_')}"
        with open(os.path.join(DOC_PATH, fname), "wb") as f:
            f.write(file.getbuffer())
        return fname
    return None

st.set_page_config(page_title="Technische Fachplanung", layout="wide", page_icon="🏗️")

# =================================================================
# 2. KI-SIMULATIONS-ENGINE (PHYSIKALISCHES MODELL)
# =================================================================
def run_simulation(pv_kwp, last_jahr, bat_kwh, ev_kwh, region="Mitte"):
    # Standort-Faktoren (Globalstrahlung 2026 Prognose)
    yield_map = {"Nord": 920, "Mitte": 1040, "Süd": 1160}
    spec_yield = yield_map.get(region, 1000)
    
    # Zeitreihen (365 Tage)
    days = 365
    t = np.linspace(0, 1, days)
    
    # PV-Erzeugung (Saisonale Glockenkurve + Wetter-Rauschen)
    pv_gen = (np.sin(np.pi * t - np.pi/10)**2 + 0.1) * (pv_kwp * spec_yield / 180)
    pv_gen = np.maximum(pv_gen * np.random.normal(1, 0.2, days), 0)
    
    # Lastgang (Winter-Peak für Gebäude)
    load_base = (np.cos(2 * np.pi * t) * 0.2 + 1.0) * (last_jahr / days)
    load_ev = (ev_kwh / days) * np.random.normal(1, 0.4, days)
    load_total = load_base + load_ev
    
    # Batteriesimulation (Ladestands-Algorithmus)
    soc = 0
    self_supply = []
    feed_in = []
    
    for p, l in zip(pv_gen, load_total):
        diff = p - l
        if diff > 0: # Überschuss
            charge = min(diff, bat_kwh - soc)
            soc += charge
            feed_in.append(diff - charge)
            self_supply.append(l)
        else: # Defizit
            discharge = min(abs(diff), soc)
            soc -= discharge
            self_supply.append(p + discharge)
            feed_in.append(0)
            
    return pd.DataFrame({
        "PV": pv_gen, "Last": load_total, "Eigen": self_supply, "Einspeisung": feed_in
    }, index=pd.date_range("2026-01-01", periods=days))

# =================================================================
# 3. SIDEBAR: SCOPE & HAK
# =================================================================
with st.sidebar:
    st.header("⚙️ Projekt-Umfang")
    d['scope'] = st.multiselect("Module aktivieren:", ["PV-System", "Speicher", "Ladeinfrastruktur"], default=d['scope'])
    
    st.divider()
    st.header("🔌 Gebäudeanschluss")
    hak_vals = [35, 50, 63, 80, 100, 125, 160, 250, 400]
    d['tech']['hak_ampere'] = st.selectbox("HAK Sicherung (A)", hak_vals, index=hak_vals.index(d['tech']['hak_ampere']))
    d['tech']['hak_kva'] = (d['tech']['hak_ampere'] * 400 * 1.732) / 1000
    st.metric("Verfügbare Leistung", f"{d['tech']['hak_kva']:.1f} kVA")
    
    if st.button("💾 Stand speichern", use_container_width=True):
        save()
        st.success("Gespeichert!")

# =================================================================
# 4. TAB-LOGIK (DIE HAUPTMASCHINE)
# =================================================================
st.title(f"Planungstool: {d['kunde']['name']}")

# Dynamische Tab-Erstellung
active_tabs = []
if "PV-System" in d['scope']: active_tabs.append("☀️ PV & Dach")
if "Speicher" in d['scope']: active_tabs.append("🔋 Speicher")
if "Ladeinfrastruktur" in d['scope']: active_tabs.append("🔌 Mobilität")
active_tabs.append("📊 KI-Analyse & ROI")

tabs = st.tabs(active_tabs)
t_idx = 0

# -----------------------------------------------------------------
# TAB: PV & DACH
# -----------------------------------------------------------------
if "PV-System" in d['scope']:
    with tabs[t_idx]:
        # DACHKONFIGURATOR (GRAFISCH)
        st.subheader("🏠 Dachgeometrie & Fläche")
        c1, c2, c3 = st.columns([1, 1, 1])
        
        with c1:
            st.write("**Dachform wählen**")
            # Visuelle Auswahl über Radio mit Emojis
            d['dach']['form'] = st.radio("Typ", ["Satteldach 🏠", "Flachdach 🏢", "Pultdach 📉", "Walmdach ⛺"], label_visibility="collapsed")
        
        with c2:
            d['dach']['breite'] = st.number_input("Breite (m)", 0.0, 500.0, d['dach']['breite'])
            d['dach']['tiefe'] = st.number_input("Tiefe/Höhe (m)", 0.0, 200.0, d['dach']['tiefe'])
            d['dach']['flaeche'] = d['dach']['breite'] * d['dach']['tiefe']
            st.caption(f"Gesamtfläche: {d['dach']['flaeche']:.1f} m²")

        with c3:
            d['dach']['neigung'] = st.slider("Dachneigung (°)", 0, 90, d['dach']['neigung'])
            d['dach']['azimut'] = st.slider("Ausrichtung (Süd=0, West=90)", -180, 180, d['dach']['azimut'])

        

        st.divider()
        
        # HARDWARE DETAILPLANUNG
        st.subheader("⚡ PV-Komponenten & Verschaltung")
        
        col_pv1, col_pv2 = st.columns(2)
        
        with col_pv1:
            st.markdown("#### Modulfelder (Strings)")
            if st.button("➕ Neues Feld"):
                d['pv']['felder'].append({"h": "", "t": "", "w": 440, "n": 10, "db": None})
            
            total_kwp = 0
            for i, f in enumerate(d['pv']['felder']):
                with st.expander(f"Feld #{i+1}: {f['n']} Module", expanded=True):
                    f['h'] = st.text_input("Hersteller", f['h'], key=f"f_h_{i}")
                    f['t'] = st.text_input("Modultyp", f['t'], key=f"f_t_{i}")
                    cf1, cf2 = st.columns(2)
                    f['w'] = cf1.number_input("Leistung (Wp)", 100, 700, f['w'], key=f"f_w_{i}")
                    f['n'] = cf2.number_input("Anzahl", 1, 5000, f['n'], key=f"f_n_{i}")
                    
                    up_m = st.file_uploader("Datenblatt PDF", type="pdf", key=f"f_up_{i}")
                    if up_m: f['db'] = upload_handler(up_m, f"M{i}")
                    
                    total_kwp += (f['w'] * f['n']) / 1000
                    if st.button("🗑️ Feld entfernen", key=f"f_del_{i}"):
                        d['pv']['felder'].pop(i); st.rerun()
            
            d['pv']['total_kwp'] = total_kwp
            st.metric("Gesamtleistung PV", f"{total_kwp:.2f} kWp")

        with col_pv2:
            st.markdown("#### Wechselrichter & Technik")
            if st.button("➕ Wechselrichter"):
                d['pv']['wr'].append({"h": "", "p": 10.0, "db": None})
            
            for j, w in enumerate(d['pv']['wr']):
                with st.container(border=True):
                    w['h'] = st.text_input("Modell", w['h'], key=f"w_h_{j}")
                    w['p'] = st.number_input("AC-Leistung (kW)", 0.0, 500.0, w['p'], key=f"w_p_{j}")
                    
                    up_w = st.file_uploader("Datenblatt WR", type="pdf", key=f"w_up_{j}")
                    if up_w: w['db'] = upload_handler(up_w, f"WR{j}")
                    
                    if st.button("🗑️ WR entfernen", key=f"w_del_{j}"):
                        d['pv']['wr'].pop(j); st.rerun()

    t_idx += 1

# -----------------------------------------------------------------
# TAB: SPEICHER
# -----------------------------------------------------------------
if "Speicher" in d['scope']:
    with tabs[t_idx]:
        st.header("🔋 Speichersystem")
        cs1, cs2 = st.columns(2)
        
        with cs1:
            s = d['speicher']
            s['hersteller'] = st.text_input("Hersteller", s['hersteller'])
            s['typ'] = st.text_input("Modell/Typ", s['typ'])
            s['kap'] = st.number_input("Nettokapazität (kWh)", 0.0, 1000.0, s['kap'])
            s['p'] = st.number_input("Ladeleistung (kW)", 0.0, 500.0, s['p'])
            
            up_s = st.file_uploader("Datenblatt Speicher", type="pdf")
            if up_s: s['db_file'] = upload_handler(up_s, "BAT")
            
        with cs2:
            st.subheader("🤖 KI-Empfehlung")
            if d['pv']['total_kwp'] > 0:
                rec = d['pv']['total_kwp'] * 1.1
                st.info(f"Basierend auf {d['pv']['total_kwp']} kWp PV-Leistung empfehlen wir einen Speicher von ca. **{rec:.1f} kWh**.")
                if s['kap'] > 0:
                    delta = (s['kap'] / d['pv']['total_kwp'])
                    if 0.8 <= delta <= 1.5: st.success("Dimensionierung ist optimal.")
                    else: st.warning("Dimensionierung weicht vom Standard ab.")

    t_idx += 1

# -----------------------------------------------------------------
# TAB: MOBILITÄT & FUHRPARK (VOLLSTÄNDIG)
# -----------------------------------------------------------------
if "Ladeinfrastruktur" in d['scope']:
    with tabs[t_idx]:
        st.header("🔌 Ladeinfrastruktur & E-Fuhrpark")
        
        # A. WALLBOXEN
        st.subheader("1. Ladepunkte")
        if st.button("➕ Ladepunkt"):
            d['mobilität']['wallboxen'].append({"n": "Wallbox", "p": 11})
        
        total_wb_p = 0
        for k, wb in enumerate(d['mobilität']['wallboxen']):
            clp1, clp2, clp3 = st.columns([2, 1, 1])
            wb['n'] = clp1.text_input("Hersteller/Modell", wb['n'], key=f"wb_n_{k}")
            wb['p'] = clp2.selectbox("Leistung (kW)", [11, 22, 44, 150], index=0 if wb['p']==11 else 1, key=f"wb_p_{k}")
            total_wb_p += wb['p']
            if clp3.button("🗑️", key=f"wb_d_{k}"):
                d['mobilität']['wallboxen'].pop(k); st.rerun()
        
        # HAK CHECK
        if (total_wb_p + d['pv']['total_kwp']) > d['tech']['hak_kva']:
            st.error(f"🚨 HAK Überlastung möglich! Gesamtleistung ({total_wb_p + d['pv']['total_kwp']:.1f} kW) > HAK ({d['tech']['hak_kva']:.1f} kVA). EMS zwingend!")
        
        st.divider()
        
        # B. FUHRPARK-VERGLEICH (DER DIESEL-RECHNER)
        st.subheader("2. TCO Fuhrpark-Vergleich (Diesel vs. Elektro)")
        if st.button("➕ Fahrzeug-Gruppe"):
            d['mobilität']['fuhrpark'].append({"t": "PKW", "n": 1, "km": 15000, "ec": 18.0, "dc": 6.5})
            
        total_ev_kwh = 0
        for idx, fz in enumerate(d['mobilität']['fuhrpark']):
            with st.container(border=True):
                cf1, cf2, cf3, cf4, cf5, cf6 = st.columns([1,1,1,1,1,1])
                fz['t'] = cf1.selectbox("Klasse", ["PKW", "Transporter (N1)", "LKW (N3)"], key=f"fz_t_{idx}")
                fz['n'] = cf2.number_input("Anzahl", 1, 100, fz['n'], key=f"fz_n_{idx}")
                fz['km'] = cf3.number_input("km/Jahr", 1000, 200000, fz['km'], key=f"fz_km_{idx}")
                fz['ec'] = cf4.number_input("kWh/100km", 5.0, 150.0, fz['ec'], key=f"fz_ec_{idx}")
                fz['dc'] = cf5.number_input("L/100km (Diesel)", 2.0, 50.0, fz['dc'], key=f"fz_dc_{idx}")
                
                # Kalkulation
                c_diesel = (fz['km']/100) * fz['dc'] * d['wirtschaft']['dieselpreis'] * fz['n']
                c_elektro = (fz['km']/100) * fz['ec'] * d['wirtschaft']['strompreis'] * fz['n']
                total_ev_kwh += (fz['km']/100) * fz['ec'] * fz['n']
                
                cf6.metric("Ersparnis/J", f"{c_diesel - c_elektro:,.0f} €")
                if cf6.button("🗑️", key=f"fz_del_{idx}"):
                    d['mobilität']['fuhrpark'].pop(idx); st.rerun()
        
        d['mobilität']['total_ev_kwh'] = total_ev_kwh

    t_idx += 1

# -----------------------------------------------------------------
# TAB: KI-ANALYSE & ROI
# -----------------------------------------------------------------
with tabs[t_idx]:
    st.header("📊 Wirtschaftlichkeit & Simulation")
    
    with st.expander("Globale Parameter"):
        cp1, cp2, cp3 = st.columns(3)
        d['wirtschaft']['strompreis'] = cp1.number_input("Strompreis Netz (€/kWh)", 0.10, 0.80, d['wirtschaft']['strompreis'])
        d['wirtschaft']['last_jahr'] = cp2.number_input("Gebäudeverbrauch (kWh/a)", 500, 1000000, d['wirtschaft']['last_jahr'])
        region = cp3.selectbox("Region", ["Nord", "Mitte", "Süd"])

    if d['pv']['total_kwp'] > 0:
        # SIMULATION
        sim_data = run_simulation(
            d['pv']['total_kwp'], 
            d['wirtschaft']['last_jahr'], 
            d['speicher']['kap'], 
            d.get('mobilität', {}).get('total_ev_kwh', 0),
            region
        )
        
        st.subheader("Jahresverlauf 2026 (Synthetisches Profil)")
        st.line_chart(sim_data[["PV", "Last", "Eigen"]])
        
        # KPI BOXEN
        k1, k2, k3, k4 = st.columns(4)
        total_p = sim_data["PV"].sum()
        total_e = sim_data["Eigen"].sum()
        autarkie = (total_e / sim_data["Last"].sum()) * 100
        
        k1.metric("PV-Ertrag", f"{total_p:,.0f} kWh")
        k2.metric("Eigenverbrauch", f"{total_e:,.0f} kWh")
        k3.metric("Autarkie", f"{autarkie:.1f} %")
        k4.metric("E-Mobilität Last", f"{d.get('mobilität', {}).get('total_ev_kwh', 0):,.0f} kWh")
        
        # KI OPTIMIERER
        st.subheader("🤖 KI-Berater")
        recs = []
        if autarkie < 35 and d['speicher']['kap'] == 0:
            recs.append("Der Autarkiegrad ist kritisch niedrig. Ein Speicher von mind. 10 kWh würde die Stromkosten um ca. 40% senken.")
        if d['pv']['total_kwp'] < (d['wirtschaft']['last_jahr'] / 1000):
            recs.append("Die PV-Anlage ist im Verhältnis zum Verbrauch klein dimensioniert. Dachflächen-Maximierung prüfen.")
        
        for r in recs: st.info(r)

    else:
        st.warning("Bitte erst PV-Konfiguration vornehmen.")

# FINALER SAVE
save()
