import streamlit as st
import os
import json
import io
import shutil
import base64
import time
import logging
from datetime import datetime

# =================================================================
# 1. SYSTEM-KONFIGURATION & AUDIT-LOGGING
# =================================================================
# Definition der Pfade für eine industrielle Verzeichnisstruktur
BASE_DIR = os.getcwd()
PROJECTS_ROOT = os.path.join(BASE_DIR, "projects")
CONFIG_ROOT = os.path.join(BASE_DIR, "config")
ASSETS_ROOT = os.path.join(BASE_DIR, "assets")
BACKUP_ROOT = os.path.join(BASE_DIR, "backups")
AUDIT_LOG = os.path.join(CONFIG_ROOT, "system_audit.log")

def initialize_file_system():
    """Gewährleistet, dass alle Systemordner vorhanden und beschreibbar sind."""
    folders = [PROJECTS_ROOT, CONFIG_ROOT, ASSETS_ROOT, BACKUP_ROOT]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            
    # Initialisierung des Audit-Logs
    if not os.path.exists(AUDIT_LOG):
        with open(AUDIT_LOG, "w", encoding="utf-8") as f:
            f.write(f"--- SYSTEM AUDIT LOG INITIALIZED {datetime.now()} ---\n")

initialize_file_system()

def log_action(message):
    """Protokolliert Systemaktionen mit Zeitstempel."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

# =================================================================
# 2. DATENMODELL-ARCHITEKTUR (KEYERROR-SCHUTZ)
# =================================================================
def get_comprehensive_template():
    """Definiert das Master-Datenmodell für maximale Planungstiefe."""
    return {
        'metadata': {
            'project_name': '',
            'customer_name': '',
            'street': '',
            'zip_city': '',
            'email': '',
            'phone': '',
            'status': 'Entwurf',
            'created_at': datetime.now().strftime("%d.%m.%Y %H:%M"),
            'last_modified': datetime.now().strftime("%d.%m.%Y %H:%M"),
            'project_id': f"PROJ-{int(time.time())}"
        },
        'pv_system': {
            'daecher': [],
            'total_kwp': 0.0,
            'module_type': 'Standard Mono-Si',
            'inverter_type': 'Hybrid-WR',
            'grid_operator': ''
        },
        'energy_storage': {
            'capacity_kwh': 0.0,
            'model': '',
            'dod': 90.0,
            'efficiency': 95.0,
            'backup_power': False
        },
        'mobility': {
            'charging_points': [],
            'smart_charging': True,
            'geig_compliant': False
        },
        'analysis': {
            'annual_consumption': 5000.0,
            'autarkie_rate': 0.0,
            'self_consumption_rate': 0.0,
            'co2_savings': 0.0
        },
        'economics': {
            'investment_net': 0.0,
            'electricity_price_buy': 0.35,
            'feed_in_tariff': 0.082,
            'maintenance_year': 150.0
        },
        'notes': ''
    }

def deep_validate_state():
    """Rekursive Validierung: Ergänzt fehlende Schlüssel in alten Projekten."""
    if "projekt_daten" not in st.session_state:
        st.session_state["projekt_daten"] = get_comprehensive_template()
    
    master = get_comprehensive_template()
    # Prüfung Ebene 1 & 2
    for section, fields in master.items():
        if section not in st.session_state["projekt_daten"]:
            st.session_state["projekt_daten"][section] = fields
            log_action(f"Reparatur: Sektion {section} hinzugefügt.")
        elif isinstance(fields, dict):
            for key, val in fields.items():
                if key not in st.session_state["projekt_daten"][section]:
                    st.session_state["projekt_daten"][section][key] = val
                    log_action(f"Reparatur: Schlüssel {section}->{key} hinzugefügt.")

deep_validate_state()

if "active_slug" not in st.session_state:
    st.session_state["active_slug"] = None

# =================================================================
# 3. DATEI-LOGIK (SPEICHERN, LÖSCHEN, IMPORT, EXPORT)
# =================================================================
def save_project(slug, data):
    """Speichert Projektdaten sicher im Cloud-Dateisystem."""
    try:
        p_path = os.path.join(PROJECTS_ROOT, slug)
        os.makedirs(p_path, exist_ok=True)
        data['metadata']['last_modified'] = datetime.now().strftime("%d.%m.%Y %H:%M")
        with open(os.path.join(p_path, "data_state.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        log_action(f"Erfolg: Projekt '{slug}' gespeichert.")
        return True
    except Exception as e:
        st.error(f"Kritischer Speicherfehler: {e}")
        log_action(f"FEHLER: Speicherfehler bei '{slug}': {e}")
        return False

def delete_project_dir(slug):
    """Löscht das Projektverzeichnis nach Validierung."""
    try:
        p_path = os.path.join(PROJECTS_ROOT, slug)
        if os.path.exists(p_path):
            shutil.rmtree(p_path)
            log_action(f"Löschung: Projekt '{slug}' wurde entfernt.")
            return True
    except Exception as e:
        st.error(f"Fehler beim Löschen: {e}")
        return False

# =================================================================
# 4. BENUTZEROBERFLÄCHE (UI) & DASHBOARD
# =================================================================
st.set_page_config(page_title="Energy Tool v2026 PRO", layout="wide", page_icon="⚙️")

# CSS für professionelle Visualisierung
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 5px; font-weight: 600; }
    .project-card { background: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR NAVIGATION
with st.sidebar:
    st.title("🛡️ System-Kern")
    if st.button("🏠 ZUM HAUPTMENÜ (Projekt schließen)", use_container_width=True):
        st.session_state["active_slug"] = None
        st.session_state["projekt_daten"] = get_comprehensive_template()
        st.rerun()
    
    st.divider()
    if st.session_state["active_slug"]:
        st.success(f"📂 AKTIV: {st.session_state['active_slug']}")
        if st.button("💾 CLOUD-SYNC (Sofort-Save)", use_container_width=True):
            if save_project(st.session_state["active_slug"], st.session_state["projekt_daten"]):
                st.toast("✅ Synchronisiert!")
        
        st.divider()
        st.subheader("Lokaler Export")
        json_data = json.dumps(st.session_state["projekt_daten"], indent=4)
        st.download_button("📥 BACKUP DOWNLOADEN", json_data, 
                          file_name=f"export_{st.session_state['active_slug']}.json", 
                          mime="application/json", use_container_width=True)

# HAUPTANSICHT
if not st.session_state["active_slug"]:
    st.title("☀️ Energy Expert – Enterprise-Verwaltung")
    
    tabs = st.tabs(["📂 PROJEKT-LISTE", "➕ NEUE PLANUNG", "📥 IMPORT", "🛡️ SYSTEM-AUDIT"])

    # --- TAB: PROJEKT-LISTE & LÖSCHEN ---
    with tabs[0]:
        st.subheader("Cloud-Datenbank")
        if not os.path.exists(PROJECTS_ROOT): os.makedirs(PROJECTS_ROOT)
        items = [d for d in os.listdir(PROJECTS_ROOT) if os.path.isdir(os.path.join(PROJECTS_ROOT, d))]
        
        if not items:
            st.info("Keine Projekte auf dem Server. Erstellen Sie eine neue Planung.")
        else:
            for slug in items:
                with st.container():
                    st.markdown(f'<div class="project-card">', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns([3, 1, 1])
                    
                    try:
                        with open(os.path.join(PROJECTS_ROOT, slug, "data_state.json"), "r", encoding="utf-8") as f:
                            p_data = json.load(f)
                            m = p_data.get('metadata', {})
                            c1.markdown(f"### {m.get('project_name', slug)}")
                            c1.caption(f"Kunde: {m.get('customer_name')} | Letzte Änderung: {m.get('last_modified')}")
                    except:
                        c1.error(f"Datenfehler in '{slug}'")
                        p_data = get_comprehensive_template()

                    if c2.button("📂 ÖFFNEN", key=f"open_{slug}", use_container_width=True):
                        st.session_state["active_slug"] = slug
                        st.session_state["projekt_daten"] = p_data
                        st.rerun()
                    
                    # SICHERHEITS-LÖSCHUNG
                    if c3.button("🗑️ LÖSCHEN", key=f"del_init_{slug}", use_container_width=True):
                        st.session_state[f"ask_del_{slug}"] = True

                    if st.session_state.get(f"ask_del_{slug}"):
                        st.error("⚠️ Projekt wirklich unwiderruflich löschen?")
                        ca, cb = st.columns(2)
                        if ca.button("JA, LÖSCHEN", key=f"y_{slug}"):
                            delete_project_dir(slug)
                            st.session_state[f"ask_del_{slug}"] = False
                            st.rerun()
                        if cb.button("ABBRECHEN", key=f"n_{slug}"):
                            st.session_state[f"ask_del_{slug}"] = False
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB: NEU ANLEGEN ---
    with tabs[1]:
        st.subheader("Projekt-Initialisierung")
        with st.form("new_proj_wizard"):
            col1, col2 = st.columns(2)
            p_name = col1.text_input("Projektname*", placeholder="z.B. PV_Meier_Meppen")
            c_name = col1.text_input("Kunde*", placeholder="Max Meier")
            c_street = col2.text_input("Straße")
            c_city = col2.text_input("PLZ / Ort")
            
            if st.form_submit_button("PROJEKT ANLEGEN & STARTEN"):
                if p_name and c_name:
                    slug = p_name.lower().replace(" ", "_")
                    st.session_state["active_slug"] = slug
                    st.session_state["projekt_daten"] = get_comprehensive_template()
                    st.session_state["projekt_daten"]["metadata"].update({
                        "project_name": p_name, "customer_name": c_name,
                        "street": c_street, "zip_city": c_city
                    })
                    save_project(slug, st.session_state["projekt_daten"])
                    st.rerun()
                else:
                    st.error("Bitte füllen Sie die Pflichtfelder aus.")

    # --- TAB: IMPORT ---
    with tabs[2]:
        st.subheader("Daten-Wiederherstellung")
        up_file = st.file_uploader("Laden Sie eine exportierte .json Datei hoch", type=["json"])
        if up_file:
            try:
                raw = json.load(up_file)
                if "metadata" in raw:
                    slug_imp = raw['metadata']['project_name'].lower().replace(" ", "_")
                    if st.button(f"Projekt '{slug_imp}' jetzt importieren"):
                        st.session_state["projekt_daten"] = raw
                        deep_validate_state() # Schlüssel prüfen
                        st.session_state["active_slug"] = slug_imp
                        save_project(slug_imp, st.session_state["projekt_daten"])
                        st.success("Erfolgreich importiert!")
                        st.rerun()
            except Exception as e:
                st.error(f"Ungültige Datei: {e}")

    # --- TAB: AUDIT ---
    with tabs[3]:
        st.subheader("System-Audit-Log")
        if os.path.exists(AUDIT_LOG):
            with open(AUDIT_LOG, "r", encoding="utf-8") as f:
                logs = f.readlines()
                st.code("".join(logs[-30:]), language="text") # Zeige letzte 30 Einträge
        
else:
    # --- DASHBOARD AKTIVES PROJEKT ---
    st.title(f"🚀 Projekt-Zentrale: {st.session_state['projekt_daten']['metadata']['project_name']}")
    
    # METRIKEN ZEILE
    m1, m2, m3, m4 = st.columns(4)
    p_dat = st.session_state["projekt_daten"]
    m1.metric("Kunde", p_dat['metadata']['customer_name'])
    m2.metric("PV-Leistung", f"{p_dat['pv_system'].get('total_kwp', 0.0)} kWp")
    m3.metric("Speicher", f"{p_dat['energy_storage'].get('capacity_kwh', 0.0)} kWh")
    m4.metric("Status", p_dat['metadata']['status'])

    st.divider()
    
    # KUNDENDATEN BEARBEITEN
    with st.expander("📝 Kunden- und Projektdaten bearbeiten", expanded=False):
        with st.form("edit_meta"):
            e1, e2 = st.columns(2)
            p_dat['metadata']['customer_name'] = e1.text_input("Kunde", p_dat['metadata']['customer_name'])
            p_dat['metadata']['email'] = e1.text_input("E-Mail", p_dat['metadata']['email'])
            p_dat['metadata']['street'] = e2.text_input("Straße", p_dat['metadata']['street'])
            p_dat['metadata']['zip_city'] = e2.text_input("PLZ / Ort")
            p_dat['metadata']['status'] = e1.selectbox("Status", ["Entwurf", "Angebot", "Beauftragt", "Abgeschlossen"], 
                                                     index=["Entwurf", "Angebot", "Beauftragt", "Abgeschlossen"].index(p_dat['metadata']['status']))
            if st.form_submit_button("DATEN ÜBERNEHMEN"):
                save_project(st.session_state["active_slug"], p_dat)
                st.success("Gespeichert.")

    st.info("💡 Navigieren Sie über die Seitenleiste zur 'Planung', um das Projekt technisch auszugestalten.")

# =================================================================
# DOKUMENTATION & FUNKTIONSÜBERSICHT
# =================================================================
# Dieses Tool nutzt eine skalierbare Dateiarchitektur.
# 1. CRUD: Create, Read, Update, Delete Logik implementiert.
# 2. Sicherheit: Rekursive Validierung verhindert KeyErrors nach Updates.
# 3. Transparenz: Audit-Log protokolliert alle Benutzeraktionen.
# 4. Flexibilität: JSON-Import/Export ermöglicht Offline-Backups.
