import streamlit as st
import json
import os
from datetime import datetime

# =================================================================
# 1. SYSTEM-PRÜFUNG (VERHINDERT DEN SWITCH-PAGE FEHLER)
# =================================================================

def check_pages_exist():
    """Überprüft, ob die physischen Dateien vorhanden sind."""
    expected_files = [
        "pages/1_Lastganganalyse.py",
        "pages/2_Planung.py",
        "pages/3_Bericht.py"
    ]
    missing = [f for f in expected_files if not os.path.exists(f)]
    return missing

# =================================================================
# 2. PROJEKT-DATEN-HANDLING
# =================================================================

def get_projects():
    if not os.path.exists("projects"):
        os.makedirs("projects")
    return [d for d in os.listdir("projects") if os.path.isdir(os.path.join("projects", d))]

def load_project_data(slug):
    path = os.path.join("projects", slug, "state.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# Initialisierung
if "active_slug" not in st.session_state:
    st.session_state["active_slug"] = None

st.set_page_config(
    page_title="Energy Intelligence OS",
    page_icon="⚡",
    layout="wide"
)

# Fehlerprüfung anzeigen
missing_files = check_pages_exist()
if missing_files:
    st.error(f"⚠️ KRITISCHER FEHLER: Dateien nicht gefunden: {missing_files}")
    st.info("Bitte benenne deine Dateien im Ordner 'pages' exakt so um, wie oben aufgeführt (ohne Emojis!).")

# =================================================================
# 3. SIDEBAR
# =================================================================

with st.sidebar:
    st.title("📁 Projekt-Verwaltung")
    all_projects = get_projects()
    
    selected = st.selectbox(
        "Projekt wählen", 
        ["-- NEU ANLEGEN --"] + all_projects,
        index=0 if not st.session_state["active_slug"] else all_projects.index(st.session_state["active_slug"]) + 1
    )

    if selected == "-- NEU ANLEGEN --":
        with st.form("new_proj"):
            n_name = st.text_input("Kundenname")
            if st.form_submit_button("Anlegen"):
                slug = f"{datetime.now().strftime('%Y%m%d')}_{n_name.replace(' ', '_').lower()}"
                p_path = os.path.join("projects", slug)
                os.makedirs(os.path.join(p_path, "documents"), exist_ok=True)
                init_state = {
                    "metadata": {"id": slug, "created": str(datetime.now())},
                    "kunde": {"name": n_name, "plz_ort": "", "straße": ""},
                    "status": "In Planung"
                }
                with open(os.path.join(p_path, "state.json"), "w", encoding="utf-8") as f:
                    json.dump(init_state, f, indent=4, ensure_ascii=False)
                st.session_state["active_slug"] = slug
                st.rerun()
    else:
        st.session_state["active_slug"] = selected

    if st.session_state["active_slug"] and st.button("🔴 Projekt schließen"):
        st.session_state["active_slug"] = None
        st.rerun()

# =================================================================
# 4. HAUPTNAVIGATION (DIE BUTTONS)
# =================================================================

if not st.session_state["active_slug"]:
    st.title("⚡ Willkommen beim Energy-Planer")
    st.write("Bitte wählen Sie links ein Projekt aus.")
else:
    data = load_project_data(st.session_state["active_slug"])
    st.title(f"🚀 Projekt: {data['kunde']['name']}")

    st.write("### 🧭 Prozess-Schritte")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        with st.container(border=True):
            st.subheader("1. Lastgang")
            if st.button("Analyse öffnen", use_container_width=True):
                st.switch_page("pages/1_Lastganganalyse.py")

    with c2:
        with st.container(border=True):
            st.subheader("2. Planung")
            if st.button("Technische Planung", use_container_width=True):
                st.switch_page("pages/2_Planung.py")

    with c3:
        with st.container(border=True):
            st.subheader("3. Bericht")
            if st.button("Export & Bericht", use_container_width=True):
                st.switch_page("pages/3_Bericht.py")

    # Stammdaten-Editor
    with st.expander("Kunden-Stammdaten"):
        with st.form("edit_data"):
            data['kunde']['name'] = st.text_input("Name", data['kunde']['name'])
            data['kunde']['plz_ort'] = st.text_input("Ort", data['kunde']['plz_ort'])
            if st.form_submit_button("Speichern"):
                with open(os.path.join("projects", st.session_state["active_slug"], "state.json"), "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                st.toast("Gespeichert!")
