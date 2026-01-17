import streamlit as st
import json
import os
import zipfile
import io
import shutil
from datetime import datetime

# =================================================================
# 1. KONFIGURATION & SYSTEM-PFADE
# =================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

os.makedirs(PROJECTS_DIR, exist_ok=True)

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"inst_name": "Mein Solarbetrieb", "inst_str": "", "inst_ort": "", "logo_path": None}

def save_settings(s):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=4, ensure_ascii=False)

# Session State Initialisierung
if "active_slug" not in st.session_state:
    st.session_state["active_slug"] = None

# =================================================================
# 2. BUSINESS-LOGIK (IMPORT/EXPORT/SUMMARY)
# =================================================================
def get_projects():
    return [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]

def export_project(slug):
    path = os.path.join(PROJECTS_DIR, slug)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(path):
            for file in files:
                z.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), PROJECTS_DIR))
    return buf.getvalue()

# =================================================================
# 3. UI-DESIGN & SEITEN-KONFIGURATION
# =================================================================
st.set_page_config(page_title="Energy Intelligence OS", layout="wide", page_icon="⚡")

# --- CUSTOM CSS FÜR ECHTEN SOFTWARE-LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 5px; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: SYSTEM-EINSTELLUNGEN & PROJEKT-MANAGER ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055683.png", width=50)
    st.title("EnergyOS v2.6")
    
    # 1. INSTALLATEUR PROFIL
    with st.expander("🛠️ Installateur-Profil", expanded=False):
        sett = load_settings()
        sett['inst_name'] = st.text_input("Firmenname", sett['inst_name'])
        sett['inst_str'] = st.text_input("Straße", sett['inst_str'])
        sett['inst_ort'] = st.text_input("PLZ / Ort", sett['inst_ort'])
        logo = st.file_uploader("Logo hochladen", type=["png", "jpg"])
        if logo:
            with open(os.path.join(BASE_DIR, "logo.png"), "wb") as f:
                f.write(logo.getbuffer())
            sett['logo_path'] = "logo.png"
        if st.button("Profil speichern", use_container_width=True):
            save_settings(sett)
            st.success("Profil aktualisiert!")

    st.divider()

    # 2. PROJEKT-AUSWAHL
    st.subheader("📁 Projekt-Akte")
    projs = get_projects()
    
    # Import
    imp_file = st.file_uploader("ZIP-Import", type="zip", label_visibility="collapsed")
    if imp_file:
        with zipfile.ZipFile(imp_file, "r") as z:
            z.extractall(PROJECTS_DIR)
        st.success("Import erfolgreich!")
        st.rerun()

    selected = st.selectbox("Wähle ein Projekt", ["-- NEU ANLEGEN --"] + projs)

    if selected == "-- NEU ANLEGEN --":
        with st.form("new_p"):
            n_name = st.text_input("Kundenname")
            if st.form_submit_button("➕ Projekt erstellen", use_container_width=True):
                slug = f"{datetime.now().strftime('%y%m%d')}_{n_name.replace(' ', '_').lower()}"
                p_path = os.path.join(PROJECTS_DIR, slug)
                os.makedirs(os.path.join(p_path, "documents"), exist_ok=True)
                init = {
                    "metadata": {"id": slug, "created": str(datetime.now())},
                    "kunde": {"name": n_name, "straße": "", "plz_ort": "", "email": "", "tel": ""},
                    "planung": {"pv_prio": "Eigenverbrauch", "status": "Planung"},
                    "notizen": ""
                }
                with open(os.path.join(p_path, "state.json"), "w", encoding="utf-8") as f:
                    json.dump(init, f, indent=4, ensure_ascii=False)
                st.session_state["active_slug"] = slug
                st.rerun()
    else:
        st.session_state["active_slug"] = selected

    # 3. AKTIONEN FÜR AKTIVES PROJEKT
    if st.session_state["active_slug"]:
        st.divider()
        st.info(f"Aktiv: **{st.session_state['active_slug']}**")
        
        # Export
        z_data = export_project(st.session_state["active_slug"])
        st.download_button("📥 Projekt-Export (ZIP)", z_data, f"{st.session_state['active_slug']}.zip", use_container_width=True)
        
        if st.button("🗑️ Projekt löschen", use_container_width=True, type="secondary"):
            shutil.rmtree(os.path.join(PROJECTS_DIR, st.session_state["active_slug"]))
            st.session_state["active_slug"] = None
            st.rerun()
        
        if st.button("❌ Schließen", use_container_width=True):
            st.session_state["active_slug"] = None
            st.rerun()

# =================================================================
# 4. HAUPTBEREICH: DAS DASHBOARD
# =================================================================
if not st.session_state["active_slug"]:
    # WILLKOMMENS-SEITE
    st.title("⚡ Energy Intelligence Planning System")
    st.write("Wählen Sie ein Projekt in der Seitenleiste aus oder erstellen Sie ein neues, um die Planung zu starten.")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info("### Letzte Projekte\n" + "\n".join([f"- {p}" for p in projs[-5:]]))
    with col_info2:
        st.success("### System-Status\n- Datenbank: Online\n- Wetter-API: Verbunden\n- Tarife 2026: Geladen")
else:
    # PROJEKT-DASHBOARD
    p_path = os.path.join(PROJECTS_DIR, st.session_state["active_slug"], "state.json")
    with open(p_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Header mit Firma
    sett = load_settings()
    h_col1, h_col2 = st.columns([3, 1])
    h_col1.title(f"Projekt: {data['kunde']['name']}")
    if sett.get('logo_path') and os.path.exists(sett['logo_path']):
        h_col2.image(sett['logo_path'], width=150)
    
    # KENNZAHLEN-LEISTE (Werte kommen aus dem State)
    st.write("---")
    m1, m2, m3, m4 = st.columns(4)
    # Simulation von Werten, falls sie in Phase 2 berechnet wurden
    pv_val = data.get("pv", {}).get("total_kwp", 0.0)
    bat_val = data.get("speicher", {}).get("kap_netto", 0.0)
    status = data.get("planung", {}).get("status", "Planung")
    
    m1.metric("PV-Leistung", f"{pv_val} kWp")
    m2.metric("Speicher", f"{bat_val} kWh")
    m3.metric("Projektstatus", status)
    m4.metric("Kunde", data['kunde']['plz_ort'] if data['kunde']['plz_ort'] else "n.a.")

    # PROZESS-NAVIGATION
    st.write("### 🧭 Workflows")
    n1, n2, n3 = st.columns(3)
    with n1:
        st.button("📊 PHASE 1: Lastganganalyse", use_container_width=True, on_click=lambda: st.switch_page("pages/1_Lastganganalyse.py"))
    with n2:
        st.button("🏗️ PHASE 2: Planung & ROI", use_container_width=True, on_click=lambda: st.switch_page("pages/2_Planung.py"))
    with n3:
        st.button("📄 PHASE 3: Berichtwesen", use_container_width=True, on_click=lambda: st.switch_page("pages/3_Bericht.py"))

    # STAMMDATEN & DOKUMENTE
    st.write("---")
    tab_data, tab_docs, tab_notes = st.tabs(["👤 Stammdaten", "📂 Dokumente", "📝 Notizen"])
    
    with tab_data:
        with st.form("sd_form"):
            c_sd1, c_sd2 = st.columns(2)
            data['kunde']['name'] = c_sd1.text_input("Name / Firma", data['kunde']['name'])
            data['kunde']['straße'] = c_sd1.text_input("Straße & Nr.", data['kunde']['straße'])
            data['kunde']['plz_ort'] = c_sd2.text_input("PLZ & Ort", data['kunde']['plz_ort'])
            data['kunde']['email'] = c_sd2.text_input("E-Mail", data['kunde']['email'])
            data['planung']['status'] = st.selectbox("Status", ["Akquise", "Planung", "Angebot", "Bau", "Archiv"])
            if st.form_submit_button("Änderungen speichern"):
                with open(p_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                st.toast("Daten gesichert!")

    with tab_docs:
        st.subheader("Hochgeladene Projektdateien")
        doc_dir = os.path.join(PROJECTS_DIR, st.session_state["active_slug"], "documents")
        files = os.listdir(doc_dir)
        if files:
            for f in files:
                col_f1, col_f2 = st.columns([4, 1])
                col_f1.write(f"📄 {f}")
                with open(os.path.join(doc_dir, f), "rb") as file_bytes:
                    col_f2.download_button("Download", file_bytes, f, key=f"dl_{f}")
        else:
            st.info("Noch keine Dokumente (Datenblätter/Lastgänge) hochgeladen.")

    with tab_notes:
        data['notizen'] = st.text_area("Interne Projektnotizen", data.get('notizen', ''), height=200)
        if st.button("Notizen speichern"):
            with open(p_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            st.success("Notiz gespeichert.")
