import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="PV-Planer Expert PRO", page_icon="☀️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .recommendation-card {
        background-color: #ffffff;
        border-left: 6px solid #007bff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
    }
    .stAlert { padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("☀️ Smart Energy Architect & Netz-Check")

# --- SIDEBAR: GLOBALE PARAMETER ---
with st.sidebar:
    st.header("🔌 Netz & Infrastruktur")
    grid_limit_kva = st.number_input("Vorh. Netzanschlussleistung (kVA)", 10, 5000, 40)
    
    st.divider()
    st.header("🎯 Modul-Auswahl")
    show_pv = st.checkbox("Photovoltaik (PV)", value=True)
    show_storage = st.checkbox("Speichersysteme (BESS)", value=False)
    show_mobility = st.checkbox("Ladeinfrastruktur & Fuhrpark", value=False)
    show_arbitrage = st.checkbox("Arbitrage & Peak-Shaving", value=False)
    
    # Logik: Nur PV Modus erkennen
    only_pv_mode = show_pv and not (show_storage or show_mobility or show_arbitrage)
    
    st.divider()
    st.header("💰 Vergütung & Preise")
    strompreis_netz = st.slider("Arbeitspreis Bezug (ct/kWh)", 15, 60, 32)
    
    # Differenzierte Vergütungssätze
    with st.expander("Einspeisevergütungen (EEG)", expanded=True):
        verg_ueberschuss = st.number_input("Vergütung Überschuss (ct/kWh)", 0.0, 15.0, 8.2, help="Satz für Eigenverbrauchsanlagen")
        verg_voll = st.number_input("Vergütung Volleinspeisung (ct/kWh)", 0.0, 20.0, 13.0, help="Höherer Satz für Volleinspeiser")
        marktwert_solar = st.number_input("Marktwert Solar (ct/kWh)", 0.0, 50.0, 6.0, help="Für Direktvermarktung (>100 kWp)")
        dv_kosten = st.number_input("Kosten Direktvermarkter (ct/kWh)", 0.0, 2.0, 0.2, help="Gebühr pro kWh")

    st.divider()
    st.header("💰 Investitions-Modus")
    manual_invest = st.toggle("Investition manuell eingeben", value=False)
    if not manual_invest:
        invest_pv_kwp = st.number_input("Investition PV (€/kWp)", 600, 2500, 1100)
        invest_bat_kwh = st.number_input("Investition Speicher (€/kWh)", 200, 1500, 550)
    else:
        total_invest_manual = st.number_input("Gesamt-Projektkosten (€)", 0, 5000000, 50000)
    
    leistungspreis = 0
    if show_arbitrage:
        leistungspreis = st.number_input("Leistungspreis (€/kW/Jahr) RLM", 0, 200, 120)

# --- SESSION STATE ---
if 'daecher' not in st.session_state: st.session_state.daecher = []
if 'lade_punkte' not in st.session_state: st.session_state.lade_punkte = []
if 'fuhrpark' not in st.session_state: st.session_state.fuhrpark = []

tabs_labels = ["📊 Wirtschaftlichkeit & Netz"]
if show_pv: tabs_labels.append("🏗️ PV-Planung")
if show_storage: tabs_labels.append("🔋 Speicher")
if show_mobility: tabs_labels.append("🚗 Fuhrpark")
if show_arbitrage: tabs_labels.append("📈 Arbitrage")

tabs = st.tabs(tabs_labels)

total_kwp = 0.0
total_storage_kwh = 0.0
pv_operation_mode = "Eigenverbrauch (Überschuss)" # Default

# --- TAB: PV-PLANUNG ---
if show_pv:
    with tabs[tabs_labels.index("🏗️ PV-Planung")]:
        st.header("🏗️ PV-Projektierung")
        
        # 1. ABFRAGE BETRIEBSMODUS (Nur wenn nur PV aktiv ist)
        if only_pv_mode:
            st.info("💡 Da Sie nur PV ohne weitere Verbraucher/Speicher planen, wählen Sie das Betriebskonzept:")
            pv_operation_mode = st.radio("Betriebskonzept wählen:", 
                                         ["Eigenverbrauch (Überschuss)", "Volleinspeisung"], 
                                         horizontal=True)
        
        # Dach Planung
        if st.button("➕ Neues Dach hinzufügen"):
            st.session_state.daecher.append({'typ': 'Satteldach', 'azimut': 0, 'neigung': 35, 'kwp': 10.0})
        
        for i, dach in enumerate(st.session_state.daecher):
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2,3,2,1])
                opts = {"Satteldach": "🏠 Satteldach", "Flachdach": "🟦 Flachdach", "Trapezblech": "🏭 Trapezblech", "Pultdach": "📐 Pultdach"}
                dach['typ'] = c1.selectbox(f"Dachart #{i+1}", list(opts.keys()), format_func=lambda x: opts[x], key=f"t_{i}")
                dach['azimut'] = c2.slider(f"Azimut #{i+1}", -180, 180, int(dach['azimut']), key=f"az_{i}", help="-90=Ost, 0=Süd, 90=West")
                dach['kwp'] = c3.number_input(f"Leistung (kWp) #{i+1}", 0.0, 5000.0, value=float(dach['kwp']), key=f"p_{i}")
                if c4.button("🗑️", key=f"del_d_{i}"):
                    st.session_state.daecher.pop(i); st.rerun()
        
        total_kwp = sum(d['kwp'] for d in st.session_state.daecher)

        # HINWEISE ZUR GRÖSSE & DIREKTVERMARKTUNG
        if total_kwp > 0:
            st.divider()
            st.subheader("⚖️ Rechtliche Einordnung")
            if total_kwp >= 100:
                st.error(f"⚠️ **Direktvermarktungspflicht:** Mit {total_kwp:.1f} kWp liegen Sie über 100 kWp. Die feste Einspeisevergütung entfällt. Sie müssen den Strom über einen Direktvermarkter an der Börse verkaufen (Marktwert abzüglich Vermarktungskosten).")
                pv_operation_mode = "Direktvermarktung (>100 kWp)" # Override
            elif total_kwp >= 25 and pv_operation_mode == "Eigenverbrauch (Überschuss)":
                st.warning("ℹ️ **Tipp:** Zwischen 25 kWp und 100 kWp kann eine Aufteilung in 'Überschuss' (für Eigenbedarf) und 'Volleinspeisung' (Restdach) sinnvoll sein, oder eine Volleinspeisung, falls der Eigenverbrauch gering ist.")
            elif pv_operation_mode == "Volleinspeisung":
                st.success(f"✅ Sie nutzen die höhere Volleinspeisevergütung ({verg_voll} ct/kWh). Dies erfordert einen separaten Zähler.")

        # Hardware Upload
        st.subheader("📄 Komponenten")
        with st.expander("Hardware-Details erfassen", expanded=False):
            col_hw1, col_hw2 = st.columns(2)
            with col_hw1:
                st.text_input("Modul Hersteller/Typ", key="mod_typ")
                st.file_uploader("Datenblatt Module", key="pdf_mod")
            with col_hw2:
                st.text_input("Wechselrichter Hersteller/Typ", key="wr_typ")
                st.file_uploader("Datenblatt WR", key="pdf_wr")

# --- TAB: SPEICHER ---
if show_storage:
    with tabs[tabs_labels.index("🔋 Speicher")]:
        st.header("🔋 Speicher")
        total_storage_kwh = st.number_input("Kapazität (kWh)", 0.0, 5000.0, 0.0)
        # Hardware Upload
        with st.expander("Speicher Hardware"):
            st.text_input("Speicher Typ", key="bat_typ")
            st.file_uploader("Datenblatt", key="pdf_bat")

# --- TAB: MOBILITY & ARBITRAGE (Platzhalter für Logik) ---
total_ev_demand_year = 0
arbitrage_revenue = 0
peak_shaving_revenue = 0
max_load_peak_kw = 0

if show_mobility:
    with tabs[tabs_labels.index("🚗 Fuhrpark")]:
        st.header("Fuhrpark")
        if st.button("➕ Fzg"): st.session_state.fuhrpark.append({'art': 'PKW', 'km': 20000})
        for i, f in enumerate(st.session_state.fuhrpark):
            f['art'] = st.selectbox(f"Art #{i}", ["PKW", "LKW"], key=f"f_{i}")
            f['km'] = st.number_input(f"km #{i}", value=f['km'], key=f"k_{i}")
            total_ev_demand_year += (f['km'] * 0.2) # Simple calc
if show_arbitrage:
    with tabs[tabs_labels.index("📈 Arbitrage")]:
        st.header("Arbitrage")
        st.info("Erlöse fließen in ROI ein.")
        arbitrage_revenue = st.number_input("Geschätzter Arbitrage Erlös (€/Jahr)", 0.0, 50000.0, 0.0)

# --- TAB 1: ÜBERSICHT & ROI ---
with tabs[0]:
    st.header("📊 Business Case & Netz-Check")
    
    # NETZ CHECK
    with st.container(border=True):
        col_n1, col_n2 = st.columns(2)
        ratio = (total_kwp / grid_limit_kva) * 100
        col_n1.metric("Anschlussauslastung (PV)", f"{ratio:.1f} %")
        if total_kwp > grid_limit_kva:
            col_n1.error("Netzanschluss überschritten!")
        else:
            col_n1.success("Netzanschluss OK")
    
    st.divider()

    # ROI BERECHNUNG NACH MODUS
    pv_yield_kwh = total_kwp * 1000 # Ertrag
    
    revenue_strom = 0.0
    revenue_einspeisung = 0.0
    
    st.subheader(f"Wirtschaftlichkeit: {pv_operation_mode}")

    if pv_operation_mode == "Direktvermarktung (>100 kWp)":
        # Direktvermarktung: Alles wird zum Marktwert verkauft (minus Kosten)
        # Ggf. Eigenverbrauch zum Opportunitätskostensatz, hier vereinfacht:
        # Erlös = Erzeugung * (Marktwert - Kosten)
        erloes_pro_kwh = marktwert_solar - dv_kosten
        revenue_einspeisung = pv_yield_kwh * (erloes_pro_kwh / 100)
        st.info(f"Kalkulation basierend auf Marktwert ({marktwert_solar} ct) abzgl. Vermarktungskosten ({dv_kosten} ct).")

    elif pv_operation_mode == "Volleinspeisung":
        # Volleinspeisung: Alles wird zur hohen Vergütung eingespeist
        revenue_einspeisung = pv_yield_kwh * (verg_voll / 100)
        st.info(f"Kalkulation basierend auf Volleinspeisevergütung ({verg_voll} ct).")

    else: 
        # Eigenverbrauch (Überschuss)
        # 1. Eigenverbrauch bestimmen
        base_self_use = 0.3 if total_storage_kwh == 0 else 0.6
        if total_ev_demand_year > 0: base_self_use += 0.1
        
        self_used_kwh = min(pv_yield_kwh * base_self_use, pv_yield_kwh)
        fed_in_kwh = pv_yield_kwh - self_used_kwh
        
        revenue_strom = self_used_kwh * (strompreis_netz / 100) # Eingesparte Stromkosten
        revenue_einspeisung = fed_in_kwh * (verg_ueberschuss / 100) # EEG Vergütung
        
        st.info(f"Kalkulation: {base_self_use*100:.0f}% Eigenverbrauch ({strompreis_netz} ct gespart) + Rest Überschussvergütung ({verg_ueberschuss} ct).")

    # Gesamtergebnis
    total_annual_revenue = revenue_strom + revenue_einspeisung + arbitrage_revenue + peak_shaving_revenue
    
    if manual_invest:
        invest_total = total_invest_manual
    else:
        invest_total = (total_kwp * invest_pv_kwp) + (total_storage_kwh * invest_bat_kwh) + (len(st.session_state.lade_punkte) * 2000)

    roi_years = invest_total / total_annual_revenue if total_annual_revenue > 0 else 999

    # METRIKEN
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Investition", f"{invest_total:,.0f} €")
    m2.metric("Cashflow / Jahr", f"{total_annual_revenue:,.0f} €")
    m3.metric("ROI", f"{roi_years:.1f} Jahre", delta_color="inverse")
    
    avg_erloes_kwh = (total_annual_revenue / pv_yield_kwh * 100) if pv_yield_kwh > 0 else 0
    m4.metric("Ø Erlös pro kWh", f"{avg_erloes_kwh:.2f} ct")

    # Chart
    years = 20
    cf_cum = [-invest_total]
    for y in range(1, years+1):
        cf_cum.append(cf_cum[-1] + total_annual_revenue - (invest_total*0.01))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(years+1), y=cf_cum, fill='tozeroy', line=dict(color='green')))
    fig.add_hline(y=0, line_color="red", line_dash="dash")
    fig.update_layout(title="Amortisationsverlauf", height=350, yaxis_title="€")
    st.plotly_chart(fig, use_container_width=True)
