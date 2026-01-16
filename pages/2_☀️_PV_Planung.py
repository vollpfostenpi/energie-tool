import streamlit as st
import pandas as pd
import plotly.express as px
from core.core import (
    get_thg_quote_params, 
    get_co2_price_prognosis, 
    get_official_sources, 
    simulate_self_consumption,
    pv_compute_multi,
    PVInput
)

st.set_page_config(page_title="PV & Flottenplanung", layout="wide")

st.title("☀️ PV-Planung & Sektorenkopplung")

# THG & CO2 Dashboard
tab1, tab2 = st.tabs(["💰 THG & CO2 (Flotte)", "🏗️ PV-Simulation"])

with tab1:
    st.subheader("Treibhausgasminderungsquote (THG)")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown(f"""
        Nutzen Sie die Elektromobilität als Einnahmequelle. Durch den Verkauf von CO2-Zertifikaten (THG-Quote) 
        erhalten Sie jährliche Prämien für Ihre E-Flotte.
        
        **Nützlicher Link:** [Infos der NOW GmbH]({get_official_sources()['THG-Quote']})
        """)
        
        thg_params = get_thg_quote_params()
        lkw_count = st.number_input("Anzahl schwere E-LKW (>12t)", value=0, step=1)
        pkw_count = st.number_input("Anzahl E-PKW / Transporter", value=0, step=1)
        
        total_thg = (lkw_count * thg_params["LKW_schwer"]) + (pkw_count * thg_params["PKW"])
        st.metric("Erwarteter THG-Erlös pro Jahr", f"{total_thg:,.2f} €".replace(",", "."))

    with c2:
        st.info("**CO2-Preispfad (BEHG)**")
        co2_path = get_co2_price_prognosis()
        df_co2 = pd.DataFrame(list(co2_path.items()), columns=["Jahr", "€/t CO2"])
        st.line_chart(df_co2.set_index("Jahr"))
        st.caption("Quelle: Umweltbundesamt")

with tab2:
    st.subheader("Photovoltaik & Mehrflächen-Berechnung")
    
    with st.expander("Dachflächen definieren", expanded=True):
        # Beispiel für ein Standard-Dach Setup
        area = st.number_input("Gesamtfläche (m²)", value=500, step=50)
        roof_type = st.selectbox("Dachtyp", ["Trapezblech (Industrie)", "Flachdach (aufgeständert)", "Satteldach (Wohnbau)"])
        
        if st.button("Berechnung starten"):
            # Simulation mit PVInput Dataclass aus core.py
            roofs = [PVInput(roof_type=roof_type, area_m2=area, azimuth=180, tilt=15, spec_yield=1050)]
            _, res = pv_compute_multi(roofs, module_w=440, margin=5, spec_yield=1050, inv_eff=0.97)
            
            st.write(f"### Ergebnis: {res['total_kwp']:.2f} kWp installierte Leistung")
            st.write(f"Erwarteter Jahresertrag: {res['total_annual_ac']:,.0f} kWh".replace(",", "."))



    if 'lastgang_data' not in st.session_state:
        st.info("💡 Tipp: Laden Sie auf der Seite 'Lastgang Analyse' Daten hoch, um den Eigenverbrauch exakt zu simulieren.")
