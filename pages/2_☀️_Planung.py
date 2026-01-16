import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time

# =================================================================
# 1. SEITEN-KONFIGURATION & SICHERHEITS-CHECK
# =================================================================
st.set_page_config(page_title="Technische Detailplanung PRO", layout="wide", page_icon="🏗️")

# NAVIGATION ZURÜCK ZU HOME
with st.sidebar:
    st.header("Navigation")
    if st.button("🏠 ZURÜCK ZUR ÜBERSICHT", use_container_width=True):
        st.switch_page("app.py")
    st.divider()
    
    if "active_slug" in st.session_state and st.session_state["active_slug"]:
        st.success(f"Projekt: {st.session_state['active_slug']}")
    else:
        st.warning("Kein Projekt geladen!")
        st.info("Bitte auf der Startseite ein Projekt auswählen.")
        st.stop()

# DATEN-VALIDIERUNG (VERHINDERT DEN KEYERROR)
if "projekt_daten" not in st.session_state:
    st.error("Systemfehler: Sitzungsdaten verloren. Bitte neu starten.")
    st.stop()

# Lokale Referenz für kürzeren Code
d = st.session_state["projekt_daten"]

# Sicherstellen, dass alle Sektionen existieren
for key in ['pv_system', 'energy_storage', 'mobility', 'analysis', 'economics']:
    if key not in d: d[key] = {}

# =================================================================
# 2. PHYSIKALISCHE HILFSFUNKTIONEN
# =================================================================
def simulate_pv_yield(kwp, orientation, tilt):
    """
    Simuliert den spezifischen Ertrag basierend auf Geometrie.
    Basis: 1000 kWh/kWp für optimale Südausrichtung (35°).
    """
    # Vereinfachtes Modell der Einstrahlungsverluste
    factors = {
        "Süd": 1.0, "Süd-West": 0.95, "Süd-Ost": 0.95, 
        "West": 0.80, "Ost": 0.80, "Nord": 0.60
    }
    tilt_factor = 1.0 - (abs(tilt - 35) / 100) # Maximum bei 35 Grad
    base_yield = 1000 # kWh pro kWp
    return kwp * base_yield * factors.get(orientation, 0.8) * tilt_factor

def generate_load_profile(annual_kwh):
    """Erzeugt ein Standardlastprofil (H0) für 24 Stunden."""
    hours = np.arange(24)
    # Typisches Haushaltsprofil: Peak morgens (7-9) und abends (17-21)
    profile = (np.exp(-((hours-8)**2)/4) * 0.8 + 
               np.exp(-((hours-19)**2)/6) * 1.2 + 0.3)
    # Skalierung auf den Tagesverbrauch
    daily_kwh = annual_kwh / 365
    scaled_profile = (profile / profile.sum()) * daily_kwh
    return scaled_profile

# =================================================================
# 3. UI: PLANUNGS-TABS
# =================================================================
st.title("🏗️ Technische Anlagen-Konfiguration")
st.markdown("---")

tab_pv, tab_bat, tab_mob, tab_sim = st.tabs([
    "☀️ PHOTOVOLTAIK", "🔋 SPEICHERSYSTEM", "🚗 MOBILITÄT & GEIG", "📊 SIMULATION & ROI"
])

# -----------------------------------------------------------------
# TAB 1: PHOTOVOLTAIK
# -----------------------------------------------------------------
with tab_pv:
    st.header("PV-Generatoren & Dachflächen")
    
    col_add, col_info = st.columns([1, 3])
    if col_add.button("➕ NEUE FLÄCHE HINZUFÜGEN"):
        if 'daecher' not in d['pv_system']: d['pv_system']['daecher'] = []
        d['pv_system']['daecher'].append({
            "id": len(d['pv_system']['daecher']),
            "name": f"Fläche {len(d['pv_system']['daecher'])+1}",
            "kwp": 10.0, "orient": "Süd", "tilt": 35
        })

    # Dynamische Listenbearbeitung
    total_yield_forecast = 0
    total_kwp = 0
    
    if 'daecher' in d['pv_system'] and d['pv_system']['daecher']:
        for i, dach in enumerate(d['pv_system']['daecher']):
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 0.5])
                dach['name'] = c1.text_input("Bezeichnung", dach['name'], key=f"n_{i}")
                dach['kwp'] = c2.number_input("Leistung (kWp)", 0.1, 1000.0, float(dach['kwp']), key=f"k_{i}")
                dach['orient'] = c3.selectbox("Ausrichtung", ["Süd", "Süd-West", "West", "Ost", "Nord"], 
                                             index=["Süd", "Süd-West", "West", "Ost", "Nord"].index(dach.get('orient', 'Süd')), 
                                             key=f"o_{i}")
                dach['tilt'] = c4.number_input("Neigung (°)", 0, 90, int(dach.get('tilt', 35)), key=f"t_{i}")
                
                if c5.button("🗑️", key=f"del_{i}"):
                    d['pv_system']['daecher'].pop(i)
                    st.rerun()
                
                # Ertrag für diese Fläche berechnen
                y = simulate_pv_yield(dach['kwp'], dach['orient'], dach['tilt'])
                total_yield_forecast += y
                total_kwp += dach['kwp']
    
    d['pv_system']['total_kwp'] = total_kwp
    d['analysis']['pv_forecast_annual'] = total_yield_forecast

    st.divider()
    st.subheader("Wechselrichter & Technik")
    tc1, tc2 = st.columns(2)
    d['pv_system']['module_type'] = tc1.text_input("Modultyp", d['pv_system'].get('module_type', ''))
    d['pv_system']['wr_type'] = tc2.text_input("WR-Modell", d['pv_system'].get('wr_type', ''))

# -----------------------------------------------------------------
# TAB 2: SPEICHERSYSTEM
# -----------------------------------------------------------------
with tab_bat:
    st.header("Batteriespeicher-Konfiguration")
    bc1, bc2 = st.columns(2)
    
    with bc1:
        d['energy_storage']['capacity_kwh'] = st.number_input("Brutto-Kapazität (kWh)", 0.0, 500.0, 
                                                              float(d['energy_storage'].get('capacity_kwh', 10.0)))
        d['energy_storage']['dod'] = st.slider("Nutzungsgrad (DoD %)", 10, 100, 
                                               int(d['energy_storage'].get('dod', 90)))
        d['energy_storage']['model'] = st.text_input("Speicherhersteller/-modell", 
                                                    d['energy_storage'].get('model', ''))
    
    with bc2:
        st.info("💡 Dimensionierungshilfe: Empfohlen sind ca. 1-1.5 kWh Speicher pro kWp PV-Leistung.")
        netto = d['energy_storage']['capacity_kwh'] * (d['energy_storage']['dod'] / 100)
        st.metric("Nutzbare Energie", f"{netto:.2f} kWh")
        
        d['energy_storage']['backup_power'] = st.checkbox("Notstrom/Inselbetrieb fähig", 
                                                         value=d['energy_storage'].get('backup_power', False))

# -----------------------------------------------------------------
# TAB 3: MOBILITÄT & GEIG
# -----------------------------------------------------------------
with tab_mob:
    st.header("Ladeinfrastruktur (Mobilitätswende)")
    st.markdown("""
    Das **GEIG** (Gebäude-Elektromobilitätsinfrastruktur-Gesetz) schreibt vor:
    * Wohngebäude > 10 Stellplätze: Leitungsinfrastruktur für alle Plätze.
    * Nicht-Wohngebäude > 10 Stellplätze: Mindestens 1 Ladepunkt + Infrastruktur für jeden 5. Platz.
    """)
    
    geig_check = st.toggle("GEIG-Relevanz prüfen", value=d['mobility'].get('geig_compliant', False))
    d['mobility']['geig_compliant'] = geig_check
    
    if st.button("➕ LADEPUNKT HINZUFÜGEN"):
        if 'charging_points' not in d['mobility']: d['mobility']['charging_points'] = []
        d['mobility']['charging_points'].append({"power": 11, "type": "Typ 2 AC"})
    
    for j, lp in enumerate(d['mobility'].get('charging_points', [])):
        with st.container(border=True):
            lc1, lc2, lc3 = st.columns([2, 2, 1])
            lp['power'] = lc1.selectbox(f"Leistung LP {j+1}", [3.7, 11, 22, 50, 150], 
                                       index=[3.7, 11, 22, 50, 150].index(lp['power']), key=f"lp_p_{j}")
            lp['type'] = lc2.text_input("Anschlusstyp", lp['type'], key=f"lp_t_{j}")
            if lc3.button("Entfernen", key=f"lp_del_{j}"):
                d['mobility']['charging_points'].pop(j)
                st.rerun()

# -----------------------------------------------------------------
# TAB 4: SIMULATION & ROI
# -----------------------------------------------------------------
with tab_sim:
    st.header("Energetische & Finanzielle Simulation")
    
    sc1, sc2 = st.columns([1, 2])
    
    with sc1:
        st.subheader("Eingangsdaten")
        annual_cons = st.number_input("Jahresstromverbrauch (kWh)", 500, 1000000, 
                                      int(d['analysis'].get('annual_consumption', 5000)))
        d['analysis']['annual_consumption'] = annual_cons
        
        buy_price = st.number_input("Strompreis Bezug (€/kWh)", 0.1, 0.8, 
                                     float(d['economics'].get('electricity_price_buy', 0.35)))
        d['economics']['electricity_price_buy'] = buy_price
        
        sell_price = st.number_input("Einspeisevergütung (€/kWh)", 0.01, 0.3, 
                                      float(d['economics'].get('feed_in_tariff', 0.082)))
        d['economics']['feed_in_tariff'] = sell_price
        
        invest = st.number_input("Investitionskosten Netto (€)", 0, 500000, 
                                  int(d['economics'].get('investment_net', total_kwp*1400 + 8000)))
        d['economics']['investment_net'] = invest

    with sc2:
        # BERECHNUNG DER AUTARKIE (Physikalische Näherung)
        # Ratio PV/Consumption und Battery/Daily_Consumption
        ratio_pv = total_yield_forecast / annual_cons if annual_cons > 0 else 0
        ratio_bat = (d['energy_storage']['capacity_kwh'] * 0.9) / (annual_cons / 365) if annual_cons > 0 else 0
        
        # Heuristik für Autarkiegrad
        autarkie = min(95.0, (35 * np.log1p(ratio_pv)) + (25 * np.log1p(ratio_bat)))
        d['analysis']['autarkie_rate'] = autarkie
        
        st.subheader("Ergebnis der 24h-Simulation")
        load_curve = generate_load_profile(annual_cons)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(24)), y=load_curve, fill='tozeroy', name="Lastgang (kWh)"))
        # Simulierter PV-Peak
        pv_curve = np.zeros(24)
        for h in range(6, 20):
            pv_curve[h] = (total_yield_forecast / 365 / 8) * np.sin(np.pi * (h-6) / 13)
        fig.add_trace(go.Scatter(x=list(range(24)), y=pv_curve, fill='tozeroy', name="PV-Erzeugung (kWh)"))
        
        fig.update_layout(title="Tagesverlauf (Beispieltag)", xaxis_title="Stunde", yaxis_title="Energie (kWh)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    
    # ROI BERECHNUNG ÜBER 20 JAHRE
    st.subheader("Wirtschaftlichkeitsprognose (ROI)")
    years = np.arange(21)
    savings_per_year = (annual_cons * (autarkie/100) * buy_price) + \
                       (max(0, total_yield_forecast - (annual_cons * autarkie/100)) * sell_price)
    
    cashflow = [-
