import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- KONFIGURATION ---
st.set_page_config(page_title="Energie-Strategie-Planer", page_icon="☀️", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .recommendation-card {
        background-color: #f0f7ff;
        border-left: 5px solid #007bff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("☀️ Smart Energy Architect")

# --- 1. ZENTRALE THEMENSTEUERUNG ---
st.sidebar.header("🎯 Relevante Module")
show_pv = st.sidebar.checkbox("Photovoltaik (PV)", value=True)
show_storage = st.sidebar.checkbox("Speichersysteme", value=False)
show_mobility = st.sidebar.checkbox("Ladeinfrastruktur & Fuhrpark", value=False)
show_arbitrage = st.sidebar.checkbox("Arbitrage-Handel (Spotmarkt)", value=False)

st.sidebar.divider()
strompreis_netz = st.sidebar.slider("Netz-Strompreis (ct/kWh)", 20, 60, 35)

# --- SESSION STATE INITIALISIERUNG ---
if 'daecher' not in st.session_state: st.session_state.daecher = []
if 'lade_punkte' not in st.session_state: st.session_state.lade_punkte = []
if 'fuhrpark' not in st.session_state: st.session_state.fuhrpark = []

# Dynamische Tabs
tabs_labels = ["📊 Übersicht & Empfehlung"]
if show_pv: tabs_labels.append("🏗️ PV-Planung")
if show_storage: tabs_labels.append("🔋 Speicher")
if show_mobility: tabs_labels.append("🚗 Fuhrpark & Laden")
if show_arbitrage: tabs_labels.append("📈 Arbitrage")

tabs = st.tabs(tabs_labels)

# Variablen für die Simulation initialisieren
total_kwp = 0.0
total_storage_kwh = 0.0
total_ev_demand_year = 0.0

# --- TAB: PV-PLANUNG ---
if show_pv:
    with tabs[tabs_labels.index("🏗️ PV-Planung")]:
        st.header("Detailplanung Photovoltaik")
        if st.button("➕ Dachfläche hinzufügen"):
            st.session_state.daecher.append({'typ': 'Flachdach', 'ausrichtung': 'Süd', 'kwp': 0.0})
        
        for i, dach in enumerate(st.session_state.daecher):
            with st.expander(f"Dach #{i+1}", expanded=True):
                c1, c2, c3, c4 = st.columns([2,2,2,1])
                dach['typ'] = c1.selectbox(f"Dachart #{i}", ["Flachdach", "Satteldach", "Trapezblech"], key=f"t_{i}")
                dach['ausrichtung'] = c2.selectbox(f"Richtung #{i}", ["Süd", "Ost/West", "Ost", "West"], key=f"a_{i}")
                dach['kwp'] = c3.number_input(f"Leistung (kWp) #{i}", 0.0, 5000.0, key=f"p_{i}")
                if c4.button("🗑️", key=f"del_d_{i}"):
                    st.session_state.daecher.pop(i)
                    st.rerun()
        total_kwp = sum(d['kwp'] for d in st.session_state.daecher)

# --- TAB: SPEICHER ---
if show_storage:
    with tabs[tabs_labels.index("🔋 Speicher")]:
        st.header("Speicherkonfiguration")
        total_storage_kwh = st.number_input("Speicherkapazität (kWh)", 0.0, 10000.0, 20.0)
        st.toggle("Peak-Shaving aktivieren")

# --- TAB: FUHRPARK & LADEN ---
if show_mobility:
    with tabs[tabs_labels.index("🚗 Fuhrpark & Laden")]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔌 Ladepunkte")
            if st.button("➕ Ladepunkt hinzufügen"):
                st
