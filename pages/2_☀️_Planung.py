import streamlit as st
import json
import os

if "projekt_daten" not in st.session_state:
    st.error("Kein aktives Projekt!")
    st.stop()

d = st.session_state["projekt_daten"]

st.title(f"🏗️ Engineering: {d['metadata']['name']}")

# --- TOP BAR: SCHNELLÜBERSICHT ---
m1, m2, m3, m4 = st.columns(4)
total_kwp = sum(f['kwp'] for f in d['pv']['fields'])
m1.metric("PV-Leistung", f"{total_kwp:.2f} kWp")
m2.metric("AC-Leistung", f"{d['pv'].get('ac_power', 0)} kVA")
m3.metric("Speicher", f"{d['storage'].get('capacity', 0)} kWh")
m4.metric("Typ", d['metadata']['type'])

# --- TABS: DETAIL-ENGINEERING ---
t_pv, t_bat, t_mob, t_eco = st.tabs(["☀️ PV & Dach", "🔋 Speicher & Arbitrage", "🚗 Mobilität & THG", "📉 ROI & BDI"])

with t_pv:
    st.header("Photovoltaik-Konfiguration")
    
    # Checkbox für Volleinspeisung (Wichtig für Wirtschaftlichkeit)
    d['pv']['full_feed'] = st.toggle("Volleinspeisung (gem. EEG § 21)", d['pv'].get('full_feed', False))
    if d['pv']['full_feed']:
        st.info("System ist auf Volleinspeisung optimiert. Eigenverbrauch wird ignoriert.")

    if st.button("➕ Modulfeld hinzufügen"):
        d['pv']['fields'].append({"name": "Feld", "kwp": 10.0, "modul": "Datenblatt...", "azimut": 0, "neigung": 35})
    
    for i, field in enumerate(d['pv']['fields']):
        with st.expander(f"Feld {i+1}: {field['name']}", expanded=True):
            c1, c2, c3 = st.columns(3)
            field['name'] = c1.text_input("Bezeichnung", field['name'], key=f"fn_{i}")
            field['kwp'] = c2.number_input("Leistung (kWp)", 0.1, 5000.0, float(field['kwp']), key=f"fk_{i}")
            field['modul'] = c3.text_input("Hersteller / Typenbezeichnung", field['modul'], key=f"fm_{i}")
            
            # Ausrichtung (Grafische Hilfswerte)
            field['azimut'] = st.select_slider("Ausrichtung", options=[-90, -45, 0, 45, 90], format_func=lambda x: {0:"Süd", -90:"Ost", 90:"West"}.get(x, f"{x}°"), key=f"fa_{i}")
            field['neigung'] = st.slider("Dachneigung (°)", 0, 90, field['neigung'], key=f"fi_{i}")
            if st.button("Löschen", key=f"fd_{i}"):
                d['pv']['fields'].pop(i)
                st.rerun()

    st.subheader("Wechselrichter & Infrastruktur")
    d['pv']['ac_power'] = st.number_input("Anschlussleistung AC (kVA)", 0.0, 2000.0, float(d['pv'].get('ac_power', 0.0)))

with t_bat:
    st.header("Speichersystem & Markt-Arbitrage")
    colb1, colb2 = st.columns(2)
    
    with colb1:
        d['storage']['capacity'] = st.number_input("Nutzbare Kapazität (kWh)", 0.0, 5000.0, float(d['storage'].get('capacity', 0.0)))
        d['storage']['brand'] = st.text_input("Speicher-Hersteller (Datenblatt)", d['storage'].get('brand', ''))
    
    with colb2:
        st.subheader("Intelligenter Arbitrage-Handel")
        d['storage']['arbitrage'] = st.toggle("Netzdienlicher Arbitrage-Betrieb", d['storage'].get('arbitrage', False))
        if d['storage']['arbitrage']:
            d['storage']['spread'] = st.slider("Geplanter Preis-Spread (ct/kWh)", 0.0, 0.50, float(d['storage'].get('spread', 0.15)))
            st.caption("Differenz zwischen Kauf- und Verkaufspreis am Spotmarkt.")

with t_mob:
    st.header("Flottenmanagement & THG-Management")
    if st.button("➕ Fahrzeuggruppe hinzufügen"):
        d['mobility']['fleets'].append({"typ": "PKW (M1)", "anzahl": 1, "verbrauch": 18.0})
    
    for j, fleet in enumerate(d['mobility']['fleets']):
        with st.container(border=True):
            cf1, cf2, cf3, cf4 = st.columns([2, 1, 1, 1])
            fleet['typ'] = cf1.selectbox("Klasse", ["PKW", "Transporter", "LKW", "Bus"], key=f"ft_{j}")
            fleet['anzahl'] = cf2.number_input("Menge", 1, 100, fleet['anzahl'], key=f"fa_{j}")
            fleet['verbrauch'] = cf3.number_input("kWh/100km", 10.0, 150.0, float(fleet['verbrauch']), key=f"fv_{j}")
            
            # THG Logik
            thg_e = fleet['anzahl'] * (280 if fleet['typ'] == "PKW" else 1200)
            cf4.metric("THG Erlös", f"{thg_e} €/a")

with t_eco:
    st.header("Wirtschaftlichkeit & Energiepreise")
    c_e1, c_e2 = st.columns(2)
    d['economics']['invest'] = c_e1.number_input("Netto-Investition (€)", 0, 5000000, d['economics'].get('invest', 25000))
    d['economics']['price_buy'] = c_e2.number_input("Strompreis Bezug (€/kWh)", 0.1, 0.8, d['economics'].get('price_buy', 0.35))
    d['economics']['price_sell'] = c_e1.number_input("Einspeisevergütung (€/kWh)", 0.0, 0.3, d['economics'].get('price_sell', 0.082))

# SPEICHERN
st.divider()
if st.button("💾 Alle Planungsdaten sicher speichern", use_container_width=True):
    p_path = os.path.join("projects", st.session_state["active_slug"], "state.json")
    with open(p_path, "w") as f:
        json.dump(d, f, indent=4)
    st.success("Datenbank aktualisiert!")
    st.balloons()
