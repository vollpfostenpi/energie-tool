from __future__ import annotations
import io
import os
import json
import shutil
import zipfile
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

# --- PDF ENGINE (ReportLab für Profi-Berichte) ---
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
except ImportError:
    st.error("Bitte installiere ReportLab: pip install reportlab")

# --- PFADE & SYSTEM ---
APP_DIR = os.getcwd()
CONFIG_DIR = os.path.join(APP_DIR, "config")
ASSETS_DIR = os.path.join(APP_DIR, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
INSTALLER_JSON = os.path.join(CONFIG_DIR, "installer.json")
PROJECTS_DIR = os.path.join(APP_DIR, "projects")

def ensure_dirs():
    for d in [CONFIG_DIR, ASSETS_DIR, PROJECTS_DIR]:
        os.makedirs(d, exist_ok=True)

def load_json(path: str, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    return default or {}

def save_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

ensure_dirs()
st.set_page_config(page_title="Energy Expert Pro", page_icon="⚙️", layout="wide")

# --- SESSION STATE INITIALISIERUNG (Das Gedächtnis) ---
if "projekt_daten" not in st.session_state:
    st.session_state["projekt_daten"] = {
        'metadata': {'project_name': '', 'customer_name': '', 'address': '', 'email': '', 'phone': ''},
        'pv_planung': {'daecher': [], 'total_kwp': 0.0},
        'speicher': {'kapazitaet': 0.0, 'modell': ''},
        'mobility': {'ladepunkte': []},
        'lastgang': None
    }
if "active_project" not in st.session_state:
    st.session_state["active_project"] = None

# --- NAVIGATION HELPER ---
def go_home():
    st.session_state["active_project"] = None
    st.rerun()

# --- PDF GENERATOR FUNKTION (Die "Profi"-Logik) ---
def build_pdf():
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Header
    inst = load_json(INSTALLER_JSON, {})
    story.append(Paragraph(f"Projektbericht: {st.session_state['projekt_daten']['metadata']['project_name']}", styles['Title']))
    story.append(Paragraph(f"Erstellt durch: {inst.get('name', 'Fachbetrieb')}", styles['Normal']))
    story.append(Spacer(1, 1*cm))
    
    # Tabelle mit Daten
    data = [
        ["Kunde", st.session_state['projekt_daten']['metadata']['customer_name']],
        ["PV-Leistung", f"{st.session_state['projekt_daten']['pv_planung']['total_kwp']} kWp"],
        ["Speicher", f"{st.session_state['projekt_daten']['speicher']['kapazitaet']} kWh"]
    ]
    t = Table(data, colWidths=[5*cm, 10*cm])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(t)
    
    doc.build(story)
    return buf.getvalue()

# --- UI: SIDEBAR NAVIGATION ---
with st.sidebar:
    if st.button("🏠 Zurück zu Home / Projekt schließen"):
        go_home()
    
    st.divider()
    if st.session_state["active_project"]:
        st.success(f"Aktiv: {st.session_state['active_project']}")
        
        # SPEICHERN OPTIONEN
        st.subheader("💾 Projekt Speichern")
        save_mode = st.radio("Ziel wählen:", ["Cloud (Server)", "Lokal (Rechner)"])
        
        if st.button("Jetzt Speichern"):
            # Cloud/Server Speicherung
            p_slug = st.session_state["active_project"]
            pdir = os.path.join(PROJECTS_DIR, p_slug)
            save_json(os.path.join(pdir, "data_state.json"), st.session_state["projekt_daten"])
            
            if save_mode == "Lokal (Rechner)":
                st.info("Klicken Sie unten auf Download.")
            else:
                st.success("Erfolgreich in der Cloud gespeichert!")
        
        if save_mode == "Lokal (Rechner)":
            json_str = json.dumps(st.session_state["projekt_daten"], indent=2)
            st.download_button("📥 JSON Download", json_str, f"{st.session_state['active_project']}.json", "application/json")

# --- HAUPTSEITE ---
if not st.session_state["active_project"]:
    st.title("☀️ Projektzentrale")
    
    # Installateur Stammdaten
    with st.expander("🏢 Installateur-Stammdaten & Logo"):
        inst = load_json(INSTALLER_JSON, {"name": "", "addr": "", "contact": ""})
        c1, c2 = st.columns(2)
        with c1:
            i_name = st.text_input("Firmenname", inst['name'])
            i_addr = st.text_area("Adresse", inst['addr'])
        with c2:
            logo_up = st.file_uploader("Logo PNG", type="png")
            if st.button("Stammdaten speichern"):
                save_json(INSTALLER_JSON, {"name": i_name, "addr": i_addr})
                if logo_up:
                    with open(LOGO_PATH, "wb") as f: f.write(logo_up.getvalue())
                st.rerun()

    # Projekt Öffnen / Neu
    t1, t2 = st.tabs(["📂 Projekt öffnen", "➕ Neues Projekt"])
    
    with t1:
        st.subheader("Vom Server/Cloud laden")
        projs = [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]
        for p in projs:
            if st.button(f"Projekt laden: {p}"):
                st.session_state["active_project"] = p
                st.session_state["projekt_daten"] = load_json(os.path.join(PROJECTS_DIR, p, "data_state.json"))
                st.rerun()
        
        st.divider()
        st.subheader("Vom Rechner hochladen")
        uploaded_json = st.file_uploader("JSON Projektdatei wählen", type="json")
        if uploaded_json:
            st.session_state["projekt_daten"] = json.load(uploaded_json)
            st.session_state["active_project"] = st.session_state["projekt_daten"]["metadata"]["project_name"]
            st.success("Lokal geladen! Gehen Sie zur Planung.")

    with t2:
        with st.form("new_proj"):
            p_name = st.text_input("Projektname")
            c_name = st.text_input("Kunde")
            if st.form_submit_button("Projekt anlegen"):
                slug = p_name.lower().replace(" ", "_")
                os.makedirs(os.path.join(PROJECTS_DIR, slug), exist_ok=True)
                st.session_state["active_project"] = slug
                st.session_state["projekt_daten"]["metadata"]["project_name"] = p_name
                st.session_state["projekt_daten"]["metadata"]["customer_name"] = c_name
                st.rerun()

else:
    st.title(f"🏠 Projekt: {st.session_state['active_project']}")
    st.info("Nutzen Sie die Seitenleiste links, um zwischen Planung, Analyse und Bericht zu wechseln.")
    
    # Übersichtskarte
    c1, c2, c3 = st.columns(3)
    c1.metric("PV Geplant", f"{st.session_state['projekt_daten']['pv_planung']['total_kwp']} kWp")
    c2.metric("Speicher", f"{st.session_state['projekt_daten']['speicher']['kapazitaet']} kWh")
    c3.metric("Kunde", st.session_state['projekt_daten']['metadata']['customer_name'])
