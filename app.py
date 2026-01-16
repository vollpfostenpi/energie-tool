import streamlit as st
import os
import json
import base64
import shutil
from datetime import datetime

# =================================================================
# 1. SYSTEM-KONFIGURATION & PFADE
# =================================================================
BASE_DIR = os.getcwd()
PROJECTS_ROOT = os.path.join(BASE_DIR, "projects")
CONFIG_ROOT = os.path.join(BASE_DIR, "config")
INSTALLER_CONFIG = os.path.join(CONFIG_ROOT, "installer_master.json")

for path in [PROJECTS_ROOT, CONFIG_ROOT]:
    os.makedirs(path, exist_ok=True)

st.set_page_config(page_title="Energy Architect Pro", layout="wide", page_icon="🛡️")

# =================================================================
# 2. HILFSFUNKTIONEN
# =================================================================

def load_installer_data():
    if os.path.exists(INSTALLER_CONFIG):
        with open(INSTALLER_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"firma": "", "anschrift": "", "logo_base64": None}

def save_installer_data(data):
    with open(INSTALLER_CONFIG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# =================================================================
# 3. UI - NAVIGATION
# =================================================================

st.title("🛡️ Enterprise Energy Management")

tab_explorer, tab_installer = st.tabs(["📂 PROJEKT-VERWALTUNG", "🏢 INSTALLATEUR-STAMMDATEN"])

# --- TAB: INSTALLATEUR-STAMMDATEN ---
with tab_installer:
    st.header("Firmenprofil & Branding")
    inst = load_installer_data()
    col_l, col_r = st.columns([1, 2])
    with col_l:
        up_logo = st.file_uploader("Logo hochladen", type=["png", "jpg", "jpeg"])
        if up_logo:
            inst["logo_base64"] = base64.b64encode(up_logo.read()).decode()
        if inst["logo_base64"]:
            st.image(base64.b64decode(inst["logo_base64"]), width=200)
    with col_r:
        inst["firma"] = st.text_input("Firmenname", inst["firma"])
        inst["anschrift"] = st.text_area("Anschrift", inst["anschrift"])
    if st.button("💾 STAMMDATEN SPEICHERN"):
        save_installer_data(inst)
        st.success("Gespeichert!")

# --- TAB: PROJEKT-VERWALTUNG (EXPLORER / IMPORT / LÖSCHEN) ---
with tab_explorer:
    col_head, col_import = st.columns([2, 1])
    with col_head:
        st.header("Vorhandene Projekte")
    
    with col_import:
        st.subheader("📥 Projekt Import")
        uploaded_file = st.file_uploader("Projekt-Datei (.json) auswählen", type=["json"])
        if uploaded_file is not None:
            try:
                import_data = json.load(uploaded_file)
                p_name = import_data['metadata']['projekt_id']
                slug = p_name.lower().replace(" ", "_")
                p_path = os.path.join(PROJECTS_ROOT, slug)
                os.makedirs(p_path, exist_ok=True)
                with open(os.path.join(p_path, "state.json"), "w", encoding="utf-8") as f:
                    json.dump(import_data, f, indent=4)
                st.success(f"Projekt '{p_name}' erfolgreich importiert!")
                st.rerun()
            except Exception as e:
                st.error(f"Fehler beim Import: {e}")

    st.divider()

    # Liste der Projekte
    projs = [d for d in os.listdir(PROJECTS_ROOT) if os.path.isdir(os.path.join(PROJECTS_ROOT, d))]
    
    if not projs:
        st.info("Keine Projekte gefunden. Erstelle ein neues oder importiere eins.")
    
    for p in projs:
        state_path = os.path.join(PROJECTS_ROOT, p, "state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                
                c1.subheader(f"📄 {p_data['metadata']['projekt_id']}")
                c1.write(f"**Kunde:** {p_data['kunde']['name']} | **Erstellt:** {p_data['metadata']['erstellt_am']}")
                
                # Button: Öffnen
                if c2.button("📂 ÖFFNEN", key=f"open_{p}", use_container_width=True):
                    st.session_state["active_slug"] = p
                    st.session_state["projekt_daten"] = p_data
                    st.switch_page("pages/2_🏗️_Planung.py")
                
                # Button: Export (Download vom Browser)
                export_json = json.dumps(p_data, indent=4)
                c3.download_button(
                    label="📥 EXPORT",
                    data=export_json,
                    file_name=f"Projekt_{p}.json",
                    mime="application/json",
                    key=f"exp_{p}",
                    use_container_width=True
                )
                
                # Button: Löschen
                if c4.button("🗑️ LÖSCHEN", key=f"del_{p}", use_container_width=True):
                    # Sicherheitsabfrage via Session State
                    st.session_state[f"confirm_delete_{p}"] = True
                
                if st.session_state.get(f"confirm_delete_{p}"):
                    st.warning(f"Soll '{p}' wirklich gelöscht werden?")
                    col_yes, col_no = st.columns(2)
                    if col_yes.button("JA, Unwiderruflich löschen", key=f"yes_{p}"):
                        shutil.rmtree(os.path.join(PROJECTS_ROOT, p))
                        del st.session_state[f"confirm_delete_{p}"]
                        st.rerun()
                    if col_no.button("Abbrechen", key=f"no_{p}"):
                        del st.session_state[f"confirm_delete_{p}"]
                        st.rerun()

# Projekt Neuanlage Bereich
st.divider()
with st.expander("➕ NEUES PROJEKT MANUELL ANLEGEN"):
    with st.form("new_p"):
        new_id = st.text_input("Projektbezeichnung")
        k_name = st.text_input("Kundenname")
        if st.form_submit_button("Anlegen"):
            slug = new_id.lower().replace(" ", "_")
            p_path = os.path.join(PROJECTS_ROOT, slug)
            os.makedirs(p_path, exist_ok=True)
            new_data = {
                "metadata": {"projekt_id": new_id, "erstellt_am": datetime.now().strftime("%d.%m.%Y")},
                "kunde": {"name": k_name},
                "pv": {"felder": []}, "speicher": {"kapazität": 0.0}, "mobilität": {"fahrzeuge": []}
            }
            with open(os.path.join(p_path, "state.json"), "w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=4)
            st.rerun()
