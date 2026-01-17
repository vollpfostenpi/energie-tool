import streamlit as st
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

# =================================================================
# 1. KERN-SYSTEM & DATEN-SICHERUNG
# =================================================================
if "active_slug" not in st.session_state:
    st.error("⚠️ Kein aktives Projekt. Bitte über das Dashboard starten.")
    st.stop()

P_SLUG = st.session_state["active_slug"]
P_PATH = os.path.join("projects", P_SLUG)
STATE_FILE = os.path.join(P_PATH, "state.json")
DOC_PATH = os.path.join(P_PATH, "documents")
os.makedirs(DOC_PATH, exist_ok=True)

# Daten laden
if "project_data" not in st.session_state:
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        st.session_state["project_data"] = json.load(f)

d = st.session_state["project_data"]

# TIEFE DATEN-STRUKTUR SICHERSTELLEN (Nichts löschen!)
def migrate_data(target, source_defaults):
    for key, value in source_defaults.items():
        if key not in target:
            target[key] = value
        elif isinstance(value, dict):
            migrate_data(target[key], value)

defaults = {
    "scope": ["PV-System", "Speicher", "Ladeinfrastruktur"],
    "tech": {"hak_ampere": 63, "ems_required": False, "berechnungs_modus": "Detail"},
    "dach": {"form": "Satteldach", "breite": 10.0, "tiefe": 6.0, "flaeche": 60.0, "neigung": 35, "azimut": 0},
    "pv": {"felder": [], "wr": [], "konzept": "Überschusseinspeisung", "total_kwp": 0.0},
    "speicher": {"kap": 0.0, "p": 0.0, "hersteller": "", "db_file": None, "spannung": "Hochvolt"},
    "mobilität": {"ladepunkte": [], "fuhrpark": []},
    "wirtschaft": {"strompreis": 0.35, "dieselpreis": 1.70, "einspeise_v": 0.08, "lastgang_jahr": 5000}
}
migrate_data(d, defaults)

def save():
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4, ensure_ascii=False)

st.set_page_config(page_title="Profi-Planung OS", layout="wide", page_icon="🏗️")

# =================================================================
# 2. KI-ERTRAGS-MODEL (STANDORT-BASIERT)
# =================================================================
def get_ki_simulation(pv_kwp, last_jahr, bat_kwh, ev_kwh, region="Mitte"):
    # Einstrahlungs-KI-Daten 2026
    regions = {"Nord": 940, "Mitte": 1060, "Süd": 1180}
    spec_yield = regions.get(region, 1000)
    
    days = 365
    t = np.linspace(0, 1, days)
    
    # PV-Kurve: Sinus-Saisonalität + zufällige Wolken-Simulation
    pv_profile = (np.sin(np.pi * t - 0.2)**2 + 0.05) * (pv_kwp * spec_yield / 150)
    pv_profile = np.maximum(pv_profile * np.random.normal(1, 0.25, days), 0)
    
    # Lastgang: H0-Profil (Gebäude) + E-Auto Last
    load_base = (np.cos(2 * np.pi * t) * 0.2 + 1.2) * (last_jahr / days)
    load_ev = (ev_kwh / days) * np.random.normal(1, 0.5, days)
    total_load = load_base + load_ev
    
    # Batterie-Zyklus-Simulation
    soc = 0
    eigen_results = []
    for p, l in zip(pv_profile, total_load):
        diff = p - l
        if diff > 0: # Laden
            charge = min(diff, bat_kwh - soc)
            soc += charge
            eigen_results.append(l + charge)
        else: # Entladen
            discharge = min(abs(diff), soc)
            soc -= discharge
            eigen_results.append(p + discharge)
            
    return pd.DataFrame({"PV": pv_profile, "Last": total_load, "Eigen": eigen_results})

# =================================================================
# 3. SIDEBAR: SEKTOR-AUSWAHL & HAK-CHECK
# =================================================================
with st.sidebar:
    st.header("🎯 Projekt-Sektoren")
    d['scope'] = st.multiselect("Themen zur Planung:", 
                                ["PV-System", "Speicher", "Ladeinfrastruktur"], 
                                default=d['scope'])
    
    st.divider()
    st.header("⚡ Gebäudeanschluss")
    d['tech']['hak_ampere'] = st.selectbox("HAK Größe (Ampere)", [35, 50, 63, 80, 100, 125, 160, 250], 
                                           index=[35, 50, 63, 80, 100, 125, 160, 250].index(d['tech']['hak_ampere']))
    
    hak_p_max = (d['tech']['hak_ampere'] * 400 * 1.73) / 1000
    st.metric("Max. Belastbarkeit", f"{hak_p_max:.1f} kW")
    
    if st.button("💾 Projektstand speichern", use_container_width=True):
        save()
        st.toast("Daten lokal gesichert!")

# =================================================================
# 4. HAUPTBEREICH: MULTI-TAB-PLANUNG
# =================================================================
st.title(f"Fachplanung: {d['kunde']['name']}")

# Dynamische Tabs generieren
tab_list = []
if "PV-System" in d['scope']: tab_list.append("☀️ PV & Dach")
if "Speicher" in d['scope']: tab_list.append("🔋 Speicher")
if "Ladeinfrastruktur" in d['scope']: tab_list.append("🔌 E-Mobilität")
tab_list.append("📊 KI-Analyse & ROI")

tabs = st.tabs(tab_list)
t_idx = 0

# -----------------------------------------------------------------
# TAB 1: PV & DACH (Grafik + Hardware + QM)
# -----------------------------------------------------------------
if "PV-System" in d['scope']:
    with tabs[t_idx]:
        st.header("PV-Anlagendesign")
        
        # A. DACHFORM & GEOMETRIE
        with st.container(border=True):
            st.subheader("Dach-Konfigurator")
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                d['dach']['form'] = st.radio("Dachform", ["Satteldach 🏠", "Flachdach 🏢", "Pultdach 📐", "Walmdach ⛺"], horizontal=False)
            with c2:
                d['dach']['breite'] = st.number_input("Dachbreite (m)", 0.0, 500.0, d['dach']['breite'])
                d['dach']['tiefe'] = st.number_input("Dachtiefe/Höhe (m)", 0.0, 200.0, d['dach']['tiefe'])
            with c3:
                area = d['dach']['breite'] * d['dach']['tiefe']
                d['dach']['flaeche'] = st.number_input("Nutzbare Fläche (m²)", 0.0, 10000.0, area)
                st.info(f"Platz für ca. {int(d['dach']['flaeche']/2)} Module")
        
        

        # B. DETAILLIERTE HARDWARE-LISTEN
        st.subheader("Komponenten-Eingabe")
        
        col_pv_left, col_pv_right = st.columns(2)
        
        with col_pv_left:
            st.markdown("#### 🟦 PV-Modulfelder (Strings)")
            if st.button("➕ Modulfeld hinzufügen"):
                d['pv']['felder'].append({"h": "", "t": "440W Standard", "w": 440, "n": 12})
            
            total_kwp = 0
            for i, f in enumerate(d['pv']['felder']):
                with st.expander(f"Feld {i+1}: {f['t']}", expanded=True):
                    f['h'] = st.text_input("Hersteller", f['h'], key=f"h_{i}")
                    f['t'] = st.text_input("Typ", f['t'], key=f"t_{i}")
                    sub1, sub2 = st.columns(2)
                    f['w'] = sub1.number_input("Watt pro Modul", 0, 800, f['w'], key=f"w_{i}")
                    f['n'] = sub2.number_input("Anzahl Module", 1, 5000, f['n'], key=f"n_{i}")
                    total_kwp += (f['w'] * f['n']) / 1000
                    if st.button("🗑️ Löschen", key=f"del_f_{i}"):
                        d['pv']['felder'].pop(i); st.rerun()
            
            d['pv']['total_kwp'] = total_kwp
            st.metric("Gesamtleistung PV", f"{total_kwp:.2f} kWp")

        with col_pv_right:
            st.markdown("#### 🚥 Wechselrichter")
            if st.button("➕ Wechselrichter hinzufügen"):
                d['pv']['wr'].append({"h": "", "p": 10.0})
            
            for j, w in enumerate(d['pv']['wr']):
                with st.container(border=True):
                    w['h'] = st.text_input("Modellbezeichnung", w['h'], key=f"wr_h_{j}")
                    w['p'] = st.number_input("AC-Nennleistung (kW)", 0.0, 500.0, w['p'], key=f"wr_p_{j}")
                    if st.button("🗑️ Entfernen", key=f"del_w_{j}"):
                        d['pv']['wr'].pop(j); st.rerun()

        st.divider()
        d['pv']['konzept'] = st.radio("Einspeisekonzept", ["Überschusseinspeisung", "Volleinspeisung"], horizontal=True)
        if d['pv']['konzept'] == "Volleinspeisung":
            st.warning("⚠️ Hinweis: Volleinspeisung benötigt einen separaten Zählerplatz.")

    t_idx += 1

# -----------------------------------------------------------------
# TAB 2: SPEICHER (Setup + Empfehlung)
# -----------------------------------------------------------------
if "Speicher" in d['scope']:
    with tabs[t_idx]:
        st.header("Batteriespeicher-Konfiguration")
        cs1, cs2 = st.columns(2)
        with cs1:
            s = d['speicher']
            s['hersteller'] = st.text_input("Hersteller / System", s['hersteller'])
            s['kap'] = st.number_input("Kapazität (kWh)", 0.0, 1000.0, s['kap'])
            s['p'] = st.number_input("Max. Lade/Entladeleistung (kW)", 0.0, 500.0, s['p'])
            s['spannung'] = st.selectbox("Typ", ["Hochvolt (HV)", "Niedervolt (48V)"])
        
        with cs2:
            st.subheader("🤖 KI-Speicher-Check")
            if d['pv']['total_kwp'] > 0:
                rec_bat = d['pv']['total_kwp'] * 1.0
                st.info(f"Empfohlene Speichergröße: **{rec_bat:.1f} kWh**")
                if s['kap'] > 0:
                    ratio = s['kap'] / d['pv']['total_kwp']
                    if 0.7 < ratio < 1.3: st.success("Dimensionierung ist ideal.")
                    else: st.warning("Speichergröße weicht deutlich von der PV-Leistung ab.")

    t_idx += 1

# -----------------------------------------------------------------
# TAB 3: MOBILITÄT (Wallboxen + Diesel-Killer)
# -----------------------------------------------------------------
if "Ladeinfrastruktur" in d['scope']:
    with tabs[t_idx]:
        st.header("Ladeinfrastruktur & E-Fuhrpark")
        
        # WALLBOXEN
        st.subheader("1. Ladestationen")
        if st.button("➕ Wallbox hinzufügen"):
            d['mobilität']['ladepunkte'].append({"n": "Ladepunkt", "p": 11})
        
        sum_wb_p = 0
        for k, lp in enumerate(d['mobilität']['ladepunkte']):
            cl1, cl2, cl3 = st.columns([2,1,1])
            lp['n'] = cl1.text_input("Modell", lp['n'], key=f"wb_n_{k}")
            lp['p'] = cl2.selectbox("Leistung (kW)", [11, 22, 50, 150], key=f"wb_p_{k}")
            sum_wb_p += lp['p']
            if cl3.button("🗑️", key=f"wb_del_{k}"):
                d['mobilität']['ladepunkte'].pop(k); st.rerun()
        
        # HAK CHECK (Wichtig!)
        if (sum_wb_p + d['pv']['total_kwp']) > hak_p_max:
            st.error(f"🚨 KRITISCH: Summe PV ({d['pv']['total_kwp']}kW) & Wallboxen ({sum_wb_p}kW) übersteigt HAK ({hak_p_max:.1f}kW)!")
            st.info("💡 Empfehlung: Lastmanagement (EMS) zwingend erforderlich.")

        st.divider()
        
        # FUHRPARK / DIESEL-VERGLEICH
        st.subheader("2. TCO-Vergleich: Diesel vs. Elektro")
        if st.button("➕ Fahrzeuggruppe"):
            d['mobilität']['fuhrpark'].append({"typ": "PKW", "anz": 1, "km": 20000, "e_cons": 18, "d_cons": 7.0})
        
        total_ev_demand = 0
        for idx, fz in enumerate(d['mobilität']['fuhrpark']):
            with st.container(border=True):
                cf1, cf2, cf3, cf4, cf5 = st.columns([1,1,1,1,1])
                fz['typ'] = cf1.selectbox("Klasse", ["PKW", "Transporter", "LKW"], key=f"f_t_{idx}")
                fz['anz'] = cf2.number_input("Stück", 1, 100, fz['anz'], key=f"f_a_{idx}")
                fz['km'] = cf3.number_input("km/Jahr", 1000, 200000, fz['km'], key=f"f_k_{idx}")
                fz['e_cons'] = cf4.number_input("kWh/100km", 0.0, 150.0, fz['e_cons'], key=f"f_ec_{idx}")
                fz['d_cons'] = cf5.number_input("L/100km Diesel", 0.0, 50.0, fz['d_cons'], key=f"f_dc_{idx}")
                
                # Ersparnis-Vorschau
                save_yr = ((fz['km']/100 * fz['d_cons'] * d['wirtschaft']['dieselpreis']) - 
                           (fz['km']/100 * fz['e_cons'] * d['wirtschaft']['strompreis'])) * fz['anz']
                st.caption(f"💰 Ersparnis dieser Gruppe: **{save_yr:,.0f} € / Jahr**")
                total_ev_demand += (fz['km']/100 * fz['e_cons'] * fz['anz'])
                if st.button("🗑️ Gruppe löschen", key=f"f_del_{idx}"):
                    d['mobilität']['fuhrpark'].pop(idx); st.rerun()
        
        d['mobilität']['total_ev_kwh'] = total_ev_demand

    t_idx += 1

# -----------------------------------------------------------------
# TAB 4: KI-ANALYSE & ROI (Das Gehirn)
# -----------------------------------------------------------------
with tabs[t_idx]:
    st.header("📊 Wirtschaftlichkeit & KI-Simulation")
    
    with st.expander("Globale Parameter"):
        cw1, cw2, cw3 = st.columns(3)
        d['wirtschaft']['strompreis'] = cw1.number_input("Strompreis Netz (€/kWh)", 0.1, 1.0, d['wirtschaft']['strompreis'])
        d['wirtschaft']['dieselpreis'] = cw2.number_input("Dieselpreis (€/L)", 0.1, 3.0, d['wirtschaft']['dieselpreis'])
        d['wirtschaft']['lastgang_jahr'] = cw3.number_input("Verbrauch Gebäude (kWh/a)", 500, 1000000, d['wirtschaft']['lastgang_jahr'])
        region = st.selectbox("KI-Standort-Modell", ["Nord", "Mitte", "Süd"])

    if d['pv']['total_kwp'] > 0:
        # Simulation starten
        sim = get_ki_simulation(d['pv']['total_kwp'], d['wirtschaft']['lastgang_jahr'], d['speicher']['kap'], d['mobilität'].get('total_ev_kwh', 0), region)
        
        st.subheader("Ertrag vs. Last (Jahresverlauf 2026)")
        st.line_chart(sim[["PV", "Last", "Eigen"]])
        
        # KPIs
        total_gen = sim["PV"].sum()
        total_self = sim["Eigen"].sum()
        autarkie = (total_self / sim["Last"].sum()) * 100
        
        k1, k2, k3 = st.columns(3)
        k1.metric("PV-Ertrag", f"{total_gen:,.0f} kWh")
        k2.metric("Autarkiegrad", f"{autarkie:.1f} %")
        k3.metric("Zusatzbedarf E-Flotte", f"{d['mobilität'].get('total_ev_kwh', 0):,.0f} kWh")
        
        # KI-BERATER
        st.subheader("🤖 KI-Optimierungshinweise")
        if "Speicher" not in d['scope'] and autarkie < 40:
            st.info("💡 KI-Tipp: Ohne Speicher verschenken Sie viel Energie. Ein 10kWh System würde Ihre Autarkie auf ca. 65% heben.")
        if d['pv']['total_kwp'] < (d['wirtschaft']['lastgang_jahr']/1000):
            st.warning("⚠️ PV-Leistung ist geringer als der Jahresverbrauch. Dach-Vollausbau empfohlen!")
    else:
        st.info("Bitte konfigurieren Sie zuerst eine PV-Anlage.")

# FINALER SAVE BEI JEDER ÄNDERUNG
save()
