import streamlit as st
import os
import json
import base64
import shutil
from datetime import datetime

# --- SYSTEMKONFIGURATION ---
BASE_DIR = os.getcwd()
PROJECTS_ROOT = os.path.join(BASE_DIR, "projects")
CONFIG_ROOT = os.path.join(BASE_DIR, "config")
INSTALLER_CONFIG = os.path.join(CONFIG_ROOT, "installer_master.json")

for path in [PROJECTS_ROOT, CONFIG_ROOT]:
    os.makedirs(path, exist_ok=True)

st.set_page_config(page_title="Energy Expert Pro", layout="wide", page_icon="🏢")

# --- CORE FUNKTIONEN ---
def load_installer():
    if os.path.exists(INSTALLER_CONFIG):
        with open(INSTALLER_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"firma": "", "straße": "", "plz_ort": "", "telefon": "", "email": "", "web": "", "logo_base64": None}

def save_installer(data):
    with open(INSTALLER_CONFIG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# --- UI NAVIGATION ---
st.title("🛡️ Enterprise Energy Management System")

tab_explorer, tab_new_project, tab_installer = st.tabs([
    "📂 PROJEKT-VERWALTUNG (Lokal/Cloud)", "➕ NEUE KUNDENAKTE ANLEGEN", "🏢 INSTALLATEUR-STAMMDATEN"
])

# 1. INSTALLATEUR-STAMMDATEN (LOGO & FIRMA)
with tab_installer:
    st.header("Installateur-Stammdaten & Branding")
    inst = load_installer()
    col_logo, col_info = st.columns([1, 2])
    
    with col_logo:
        st.subheader("Firmenlogo")
        uploaded_logo = st.file_uploader("Logo hochladen (PNG/JPG)", type=["png", "jpg", "jpeg"], key="logo_up")
        if uploaded_logo:
            inst["logo_base64"] = base64.b64encode(uploaded_logo.read()).decode()
        if inst["logo_base64"]:
            st.image(base64.b64decode(inst["logo_base64"]), width=250)
            if st.button("Logo entfernen"):
                inst["logo_base64"] = None
                save_installer(inst)
                st.rerun()

    with col_info:
        st.subheader("Unternehmensdetails")
        inst["firma"] = st.text_input("Firmenname", inst["firma"])
        inst["straße"] = st.text_input("Straße / Nr.", inst["straße"])
        inst["plz_ort"] = st.text_input("PLZ / Ort", inst["plz_ort"])
        c1, c2 = st.columns(2)
        inst["telefon"] = c1.text_input("Telefon (Zentrale)", inst["telefon"])
        inst["email"] = c2.text_input("E-Mail (Zentrale)", inst["email"])
        inst["web"] = st.text_input("Webseite", inst["web"])

    if st.button("💾 STAMMDATEN GLOBAL SPEICHERN", use_container_width=True):
        save_installer(inst)
        st.success("Firmendaten dauerhaft gesichert.")

# 2. NEUE KUNDENAKTE (VOLLSTÄNDIGE DATENERFASSUNG)
with tab_new_project:
    st.header("Vollständige Kundendatenerfassung")
    with st.form("new_customer_form"):
        st.subheader("📍 Projekt-Basis")
        p_id = st.text_input("Projekt-Kennziffer / ID (Eindeutig)", placeholder="z.B. PV-2024-001-Müller")
        
        st.subheader("👤 Kundendaten")
        col_k1, col_k2 = st.columns(2)
        k_name = col_k1.text_input("Vollständiger Name / Firma")
        k_ansprech = col_k2.text_input("Ansprechpartner (optional)")
        k_str = col_k1.text_input("Straße / Hausnummer")
        k_ort = col_k2.text_input("PLZ / Ort")
        k_mail = col_k1.text_input("E-Mail Adresse")
        k_tel = col_k2.text_input("Telefonnummer")
        
        st.subheader("📑 Technische & Steuerliche Eckpunkte")
        ct1, ct2, ct3 = st.columns(3)
        k_zaehl = ct1.text_input("Zählernummer / MSB-ID")
        k_steuer = ct2.selectbox("Steuer-Status", ["0% (Privat)", "19% (Gewerbe)", "Kleinunternehmer"])
        k_status = ct3.selectbox("Projektstatus", ["Anfrage", "Planung", "Angebot erstellt", "In Umsetzung"])
        
        if st.form_submit_button("💾 PROJEKT-AKTE LOKAL ANLEGEN"):
            if not p_id or not k_name:
                st.error("Bitte Projekt-ID und Kundennamen angeben!")
            else:
                slug = p_id.lower().replace(" ", "_")
                p_path = os.path.join(PROJECTS_ROOT, slug)
                os.makedirs(p_path, exist_ok=True)
                
                project_data = {
                    "metadata": {"id": p_id, "erstellt": str(datetime.now()), "status": k_status},
                    "kunde": {
                        "name": k_name, "ansprechpartner": k_ansprech, "straße": k_str,
                        "plz_ort": k_ort, "email": k_mail, "telefon": k_tel,
                        "zählernummer": k_zaehl, "steuer_status": k_steuer
                    },
                    "pv": {"felder": []}, "speicher": {"kapazität": 0.0}, "mobilität": {"fahrzeuge": []}
                }
                with open(os.path.join(p_path, "state.json"), "w", encoding="utf-8") as f:
                    json.dump(project_data, f, indent=4)
                st.success(f"Projekt {p_id} erfolgreich im System angelegt.")
                st.rerun()

# 3. PROJEKT-VERWALTUNG (LOKAL & CLOUD-FUNKTIONEN)
with tab_explorer:
    col_l, col_r = st.columns([2, 1])
    with col_l: st.header("Gespeicherte Kundenakten")
    with col_r:
        st.subheader("📥 Projekt Import (Lokal -> Cloud)")
        uploaded_file = st.file_uploader("JSON-Projektdatei wählen", type=["json"])
        if uploaded_file:
            try:
                imp_data = json.load(uploaded_file)
                slug = imp_data['metadata']['id'].lower().replace(" ", "_")
                os.makedirs(os.path.join(PROJECTS_ROOT, slug), exist_ok=True)
                with open(os.path.join(PROJECTS_ROOT, slug, "state.json"), "w", encoding="utf-8") as f:
                    json.dump(imp_data, f, indent=4)
                st.success("Projekt erfolgreich importiert.")
                st.rerun()
            except: st.error("Dateiformat ungültig.")

    st.divider()
    projs = [d for d in os.listdir(PROJECTS_ROOT) if os.path.isdir(os.path.join(PROJECTS_ROOT, d))]
    
    for p in projs:
        s_path = os.path.join(PROJECTS_ROOT, p, "state.json")
        if os.path.exists(s_path):
            with open(s_path, "r", encoding="utf-8") as f: d = json.load(f)
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                c1.write(f"### {d['metadata']['id']}")
                c1.write(f"**Kunde:** {d['kunde']['name']} | **Ort:** {d['kunde']['plz_ort']}")
                
                if c2.button("📂 ÖFFNEN", key=f"open_{p}", use_container_width=True):
                    st.session_state["active_slug"] = p
                    st.session_state["projekt_daten"] = d
                    st.switch_page("pages/2_🏗️_Planung.py")
                
                c3.download_button("📤 EXPORT (LOKAL)", data=json.dumps(d, indent=4), file_name=f"{p}.json", key=f"exp_{p}", use_container_width=True)
                
                if c4.button("🗑️ LÖSCHEN", key=f"del_{p}", use_container_width=True):
                    shutil.rmtree(os.path.join(PROJECTS_ROOT, p))
                    st.rerun()
