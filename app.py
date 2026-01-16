import streamlit as st
import os
import json
import base64
import shutil
from datetime import datetime

# SYSTEM-PFADE
CONFIG_DIR = "config"
PROJECTS_DIR = "projects"
MASTER_CONFIG = os.path.join(CONFIG_DIR, "installer_master.json")

for d in [CONFIG_DIR, PROJECTS_DIR]:
    os.makedirs(d, exist_ok=True)

st.set_page_config(page_title="Energy Architect Pro", layout="wide", page_icon="🛡️")

# --- HELPER: LOGO & DATEN ---
def get_installer():
    if os.path.exists(MASTER_CONFIG):
        with open(MASTER_CONFIG, "r") as f: return json.load(f)
    return {"name": "", "anschrift": "", "logo": None, "web": "", "email": ""}

def save_installer(data):
    with open(MASTER_CONFIG, "w") as f: json.dump(data, f)

# --- UI NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/807/807351.png", width=80)
    st.title("Enterprise Suite")
    menu = st.radio("Navigation", ["Dashboard & Projekte", "Installateur-Stammdaten", "System-Archiv"])
    st.divider()
    if st.session_state.get("active_slug"):
        st.success(f"Aktiv: {st.session_state['active_slug']}")
        if st.button("Projekt schließen"):
            st.session_state["active_slug"] = None
            st.rerun()

# --- MODUL: INSTALLATEUR-STAMMDATEN ---
if menu == "Installateur-Stammdaten":
    st.header("🏢 Unternehmensprofil & Branding")
    inst = get_installer()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Branding")
        logo_up = st.file_uploader("Firmenlogo (High-Res)", type=["png", "jpg", "svg"])
        if logo_up:
            inst["logo"] = base64.b64encode(logo_up.read()).decode()
        if inst["logo"]:
            st.image(base64.b64decode(inst["logo"]), use_container_width=True)
            if st.button("Logo entfernen"):
                inst["logo"] = None
                save_installer(inst)
                st.rerun()

    with col2:
        st.subheader("Kontaktdaten für Berichte")
        inst["name"] = st.text_input("Firmenname", inst["name"])
        inst["anschrift"] = st.text_area("Vollständige Anschrift", inst["anschrift"])
        inst["email"] = st.text_input("Zentrale E-Mail", inst["email"])
        inst["web"] = st.text_input("Webseite (URL)", inst["web"])
        if st.button("💾 Stammdaten global sichern"):
            save_installer(inst)
            st.toast("Firmendaten aktualisiert!")

# --- MODUL: PROJEKT-MANAGER ---
elif menu == "Dashboard & Projekte":
    st.header("📂 Projekt-Management")
    t1, t2 = st.tabs(["Projektliste", "🆕 Neue Erfassung"])
    
    with t1:
        projs = [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]
        if not projs: st.info("Keine Projekte vorhanden.")
        for p in projs:
            p_file = os.path.join(PROJECTS_DIR, p, "state.json")
            if os.path.exists(p_file):
                with open(p_file, "r") as f: data = json.load(f)
                with st.container(border=True):
                    c1, c2, c3 = st.columns([4, 1, 1])
                    c1.subheader(data['metadata']['name'])
                    c1.caption(f"Typ: {data['metadata']['type']} | Status: {data['metadata']['status']}")
                    if c2.button("Öffnen", key=f"o_{p}"):
                        st.session_state["active_slug"] = p
                        st.session_state["projekt_daten"] = data
                        st.switch_page("pages/2_🏗️_Planung.py")
                    if c3.button("Löschen", key=f"d_{p}"):
                        shutil.rmtree(os.path.join(PROJECTS_DIR, p))
                        st.rerun()

    with t2:
        with st.form("new_proj"):
            name = st.text_input("Projekt-Bezeichnung / ID")
            p_type = st.selectbox("Projekt-Kategorie", ["Einfamilienhaus", "Gewerbe / Industrie", "Landwirtschaft", "Solarpark"])
            customer = st.text_input("Kundenname / Kontakt")
            if st.form_submit_button("Projekt initialisieren"):
                slug = name.lower().replace(" ", "_")
                new_data = {
                    "metadata": {"name": name, "type": p_type, "customer": customer, "status": "Planung", "date": str(datetime.now())},
                    "pv": {"fields": [], "ac_power": 0.0, "full_feed": False},
                    "storage": {"capacity": 0.0, "arbitrage": False, "spread": 0.15},
                    "mobility": {"fleets": []},
                    "economics": {"invest": 0.0, "price_buy": 0.35, "price_sell": 0.082}
                }
                os.makedirs(os.path.join(PROJECTS_DIR, slug), exist_ok=True)
                with open(os.path.join(PROJECTS_DIR, slug, "state.json"), "w") as f:
                    json.dump(new_data, f)
                st.session_state["active_slug"] = slug
                st.session_state["projekt_daten"] = new_data
                st.success("Erfolgreich angelegt!")
