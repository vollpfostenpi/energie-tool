from __future__ import annotations
import io
import os
import json
import shutil
from datetime import datetime
from typing import Any, Dict
import pandas as pd
import streamlit as st

# --- KONFIGURATION & PFADE ---
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
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except: pass
    return default or {}

def save_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def slugify(s: str):
    return "".join(c if c.isalnum() else "-" for c in s.strip()).lower() or "projekt"

ensure_dirs()
st.set_page_config(page_title="Energy Tool Home", page_icon="🏠", layout="wide")

# --- GLOBALER SESSION STATE ---
if "projekt_daten" not in st.session_state:
    st.session_state["projekt_daten"] = {
        'daecher': [], 
        'bat_systeme': [], 
        'lade_punkte': [], 
        'total_kwp': 0.0, 
        'total_kwh': 0.0,
        'customer': {},
        'lastgang_vorhanden': False
    }
if "active_project" not in st.session_state:
    st.session_state["active_project"] = None

# --- UI HEADER ---
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
with col_title:
    st.title("☀️ Energy Expert Pro – Home")
    st.info("Willkommen in der Projektverwaltung. Wählen Sie ein Projekt aus, um es zu bearbeiten.")

# --- SEKTION: INSTALLATEUR-DATEN ---
with st.sidebar.expander("🏢 Installateur-Stammdaten"):
    inst = load_json(INSTALLER_JSON, {"name": "", "addr": "", "contact": ""})
    i_name = st.text_input("Firmenname", inst['name'])
    i_addr = st.text_area("Adresse", inst['addr'])
    i_contact = st.text_input("Kontakt", inst['contact'])
    logo_up = st.file_uploader("Logo hochladen (PNG)", type=["png"])
    if st.button("Stammdaten speichern"):
        save_json(INSTALLER_JSON, {"name": i_name, "addr": i_addr, "contact": i_contact})
        if logo_up:
            with open(LOGO_PATH, "wb") as f: f.write(logo_up.getvalue())
        st.success("Gespeichert!")

# --- PROJEKT-MANAGEMENT ---
t1, t2 = st.tabs(["🆕 Neues Projekt", "📂 Projektliste"])

with t1:
    with st.form("new_proj"):
        p_name = st.text_input("Projektname / Referenz")
        c_name = st.text_input("Kundenname")
        if st.form_submit_button("Projekt anlegen"):
            if p_name and c_name:
                slug = slugify(p_name)
                pdir = os.path.join(PROJECTS_DIR, slug)
                os.makedirs(pdir, exist_ok=True)
                meta = {"project_name": p_name, "customer_name": c_name, "created": datetime.now().strftime("%d.%m.%Y")}
                save_json(os.path.join(pdir, "metadata.json"), meta)
                st.success(f"Projekt {p_name} erstellt.")
                st.rerun()

with t2:
    projs = [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]
    for p_slug in projs:
        meta = load_json(os.path.join(PROJECTS_DIR, p_slug, "metadata.json"))
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.write(f"**{meta.get('project_name')}** ({meta.get('customer_name')})")
            if c2.button("📂 Öffnen", key=f"open_{p_slug}"):
                st.session_state["active_project"] = p_slug
                st.session_state["project_name"] = meta.get('project_name')
                saved_data = load_json(os.path.join(PROJECTS_DIR, p_slug, "data_state.json"))
                if saved_data: st.session_state["projekt_daten"] = saved_data
                st.success(f"Projekt geladen.")
            if c3.button("🗑️", key=f"del_{p_slug}"):
                shutil.rmtree(os.path.join(PROJECTS_DIR, p_slug))
                st.rerun()

if st.session_state["active_project"]:
    st.sidebar.divider()
    st.sidebar.success(f"Aktiv: {st.session_state['project_name']}")
    if st.sidebar.button("💾 Stand jetzt speichern"):
        p_slug = st.session_state["active_project"]
        save_json(os.path.join(PROJECTS_DIR, p_slug, "data_state.json"), st.session_state["projekt_daten"])
        st.sidebar.toast("Fortschritt gesichert!")
