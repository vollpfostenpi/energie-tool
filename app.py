import streamlit as st
import json
import os
import zipfile
import io
import shutil
import re
from datetime import datetime
from pathlib import Path
from html import escape

# =================================================================
# 1) KONFIGURATION & SYSTEM-PFADE
# =================================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = BASE_DIR / "projects"
DATA_DIR = BASE_DIR / "_data"
SETTINGS_FILE = DATA_DIR / "settings.json"
BRANDING_DIR = DATA_DIR / "branding"

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
BRANDING_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SETTINGS = {
    "inst_name": "Mein Solarbetrieb",
    "inst_str": "",
    "inst_ort": "",
    "logo_path": None,  # relativ zu BASE_DIR (z.B. "_data/branding/logo.png")
}

def load_settings():
    if SETTINGS_FILE.exists():
        try:
            with SETTINGS_FILE.open("r", encoding="utf-8") as f:
                s = json.load(f)
            for k, v in DEFAULT_SETTINGS.items():
                s.setdefault(k, v)
            return s
        except Exception:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()

def save_settings(s: dict):
    for k, v in DEFAULT_SETTINGS.items():
        s.setdefault(k, v)
    with SETTINGS_FILE.open("w", encoding="utf-8") as f:
        json.dump(s, f, indent=4, ensure_ascii=False)

if "active_slug" not in st.session_state:
    st.session_state["active_slug"] = None

# =================================================================
# 2) HELPERS (Slug / ZIP Sicherheit / Projekte)
# =================================================================
def slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "projekt"

def get_projects():
    projs = []
    if PROJECTS_DIR.exists():
        for d in PROJECTS_DIR.iterdir():
            if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("_"):
                projs.append(d.name)
    projs.sort(key=lambda x: (PROJECTS_DIR / x).stat().st_mtime, reverse=True)
    return projs

def ensure_project_structure(slug: str):
    p = PROJECTS_DIR / slug
    (p / "documents").mkdir(parents=True, exist_ok=True)

def _read_state(slug: str) -> dict:
    state_path = PROJECTS_DIR / slug / "state.json"
    if not state_path.exists():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _zip_members(z: zipfile.ZipFile):
    # Normalisieren auf forward slashes
    names = []
    for info in z.infolist():
        n = info.filename.replace("\\", "/").lstrip("/")
        if n and not n.endswith("/"):
            names.append(n)
    return names

def _detect_zip_root_slug(names: list[str]) -> str | None:
    """
    Erkennt, ob das ZIP sauber ein Root-Dir hat: <slug>/state.json
    """
    for n in names:
        if n.count("/") >= 1 and n.endswith("state.json"):
            root = n.split("/", 1)[0]
            return root
    return None

def safe_extract_zip_to_temp(zip_bytes_or_file, temp_dir: Path):
    """
    ZipSlip-Schutz: extrahiert in temp_dir.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_bytes_or_file, "r") as z:
        for member in z.infolist():
            name = member.filename.replace("\\", "/")
            if name.startswith("/") or name.startswith("\\"):
                raise ValueError("Unsicheres ZIP (absolute Pfade).")
            target = (temp_dir / name).resolve()
            if not str(target).startswith(str(temp_dir.resolve())):
                raise ValueError("Unsicheres ZIP (Pfad-Traversal erkannt).")
        z.extractall(temp_dir)

def import_project_zip(uploaded_file):
    """
    Importiert ein Projekt-ZIP robust:
    - Wenn ZIP bereits 'slug/...' enthält: copy nach projects/slug (Merge/Override optional)
    - Wenn ZIP flach ist (state.json + documents/...): Erzeuge slug aus state.json oder Timestamp
    """
    temp_root = DATA_DIR / "_import_tmp"
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    # 1) ZIP in temp extrahieren
    safe_extract_zip_to_temp(uploaded_file, temp_root)

    # 2) prüfen ob root-folder existiert
    # Suche state.json
    state_files = list(temp_root.rglob("state.json"))
    if not state_files:
        raise ValueError("ZIP enthält keine state.json – kein gültiges Projekt.")

    # Wenn mehrere state.json -> nimm die, die am nächsten an root liegt
    state_files.sort(key=lambda p: len(p.parts))
    state_path = state_files[0]

    # root slug bestimmen (Ordner in dem state.json liegt)
    proj_folder = state_path.parent  # könnte temp_root/<slug> sein oder temp_root/
    if proj_folder == temp_root:
        # flaches ZIP -> slug aus state.json
        try:
            with state_path.open("r", encoding="utf-8") as f:
                stt = json.load(f)
            slug = stt.get("metadata", {}).get("id") or slugify(stt.get("kunde", {}).get("name", "")) or "projekt"
        except Exception:
            slug = f"import_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        slug = slugify(slug)
    else:
        slug = proj_folder.name

    # slug-Kollision lösen
    base_slug = slug
    i = 2
    while (PROJECTS_DIR / slug).exists():
        slug = f"{base_slug}_{i}"
        i += 1

    dest = PROJECTS_DIR / slug
    dest.mkdir(parents=True, exist_ok=True)

    # 3) Inhalte kopieren
    # Wenn proj_folder != temp_root: kopiere diesen Ordner, sonst kopiere temp_root/
    src = proj_folder if proj_folder != temp_root else temp_root

    # copytree in bestehendes Ziel: manuell mergen
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(src)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)

    # 4) sicherstellen structure
    ensure_project_structure(slug)

    # 5) state.json id anpassen falls nötig
    state_dest = dest / "state.json"
    try:
        with state_dest.open("r", encoding="utf-8") as f:
            d = json.load(f)
        d.setdefault("metadata", {})
        d["metadata"]["id"] = slug
        with state_dest.open("w", encoding="utf-8") as f:
            json.dump(d, f, indent=4, ensure_ascii=False)
    except Exception:
        pass

    return slug

# =================================================================
# 3) EXPORT-FUNKTIONEN
# =================================================================
def export_project(slug: str) -> bytes:
    """Exportiert komplettes Projektverzeichnis als ZIP -> Download = lokal speichern."""
    path = PROJECTS_DIR / slug
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(path):
            for file in files:
                full = Path(root) / file
                rel = full.relative_to(PROJECTS_DIR)  # => slug/...
                z.write(full, rel.as_posix())
    return buf.getvalue()

def export_report_zip(slug: str) -> bytes:
    """
    ZIP mit index.html + state.json + documents/*
    """
    proj_dir = PROJECTS_DIR / slug
    state = _read_state(slug)
    docs_dir = proj_dir / "documents"

    sett = load_settings()
    inst_name = sett.get("inst_name", "")
    inst_str = sett.get("inst_str", "")
    inst_ort = sett.get("inst_ort", "")

    kunde = state.get("kunde", {})
    meta = state.get("metadata", {})
    pv_val = state.get("pv", {}).get("total_kwp", 0.0)
    bat_val = state.get("speicher", {}).get("kap_netto", state.get("speicher", {}).get("kap", 0.0))
    status = state.get("planung", {}).get("status", "Planung")

    created = meta.get("created") or ""
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    attachments = []
    if docs_dir.exists():
        for p in docs_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(proj_dir).as_posix()
                attachments.append(rel)
    attachments.sort()

    def li(link, label=None):
        label = label or link
        return f'<li><a href="{escape(link)}" target="_blank">{escape(label)}</a></li>'

    att_list = "\n".join([li(a) for a in attachments]) if attachments else "<li><i>Keine Anlagen vorhanden.</i></li>"

    index_html = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Projektbericht {escape(slug)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; background: #f7f8fa; color:#111; }}
    .card {{ background:#fff; border-radius:12px; padding:18px; box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-bottom:16px; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:12px; }}
    .kpi {{ display:flex; gap:12px; flex-wrap:wrap; }}
    .pill {{ background:#eef2ff; padding:6px 10px; border-radius:999px; font-size: 12px; }}
    ul {{ margin: 8px 0 0 18px; }}
    .muted {{ color:#666; font-size: 12px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; }}
    a {{ color:#1f5eff; text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Projektbericht: {escape(kunde.get("name","(ohne Name)"))}</h1>
    <div class="muted">Erstellt: {escape(report_time)} • Projekt-ID: <span class="mono">{escape(slug)}</span></div>
    <div class="kpi" style="margin-top:10px">
      <div class="pill">PV: {escape(str(pv_val))} kWp</div>
      <div class="pill">Speicher: {escape(str(bat_val))} kWh</div>
      <div class="pill">Status: {escape(str(status))}</div>
    </div>
  </div>

  <div class="card">
    <h2>Installateur</h2>
    <div>{escape(inst_name)}</div>
    <div class="muted">{escape(inst_str)} • {escape(inst_ort)}</div>
  </div>

  <div class="card">
    <h2>Stammdaten Kunde</h2>
    <div class="grid">
      <div><b>Name</b><br/>{escape(kunde.get("name",""))}</div>
      <div><b>Ort</b><br/>{escape(kunde.get("plz_ort",""))}</div>
      <div><b>Straße</b><br/>{escape(kunde.get("straße",""))}</div>
      <div><b>E-Mail</b><br/>{escape(kunde.get("email",""))}</div>
    </div>
    <div class="muted" style="margin-top:10px">Projekt created: {escape(str(created))}</div>
  </div>

  <div class="card">
    <h2>Anlagen / Dateien</h2>
    <ul>{att_list}</ul>
    <div class="muted" style="margin-top:10px">Links funktionieren nach Entpacken des ZIP lokal.</div>
  </div>

  <div class="card">
    <h2>Rohdaten</h2>
    <ul>{li("state.json", "state.json (Projektzustand)")}</ul>
  </div>
</body>
</html>"""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("index.html", index_html)
        state_path = proj_dir / "state.json"
        if state_path.exists():
            z.write(state_path, "state.json")
        if docs_dir.exists():
            for p in docs_dir.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(proj_dir).as_posix()
                    z.write(p, rel)
    return buf.getvalue()

# =================================================================
# 4) UI / PAGE CONFIG
# =================================================================
st.set_page_config(page_title="Energy Intelligence OS", layout="wide", page_icon="⚡")

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 8px; }
    .stMetric { background-color: white; padding: 15px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """,
    unsafe_allow_html=True,
)

# =================================================================
# 5) SIDEBAR
# =================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055683.png", width=48)
    st.title("EnergyOS v2.6")

    with st.expander("🛠️ Installateur-Profil", expanded=False):
        sett = load_settings()
        sett["inst_name"] = st.text_input("Firmenname", sett.get("inst_name", ""))
        sett["inst_str"] = st.text_input("Straße", sett.get("inst_str", ""))
        sett["inst_ort"] = st.text_input("PLZ / Ort", sett.get("inst_ort", ""))

        logo = st.file_uploader("Logo hochladen", type=["png", "jpg", "jpeg"])
        if logo:
            ext = Path(logo.name).suffix.lower() or ".png"
            out_logo = BRANDING_DIR / f"logo{ext}"
            with out_logo.open("wb") as f:
                f.write(logo.getbuffer())
            sett["logo_path"] = str(out_logo.relative_to(BASE_DIR))
            st.success("Logo gespeichert.")

        if st.button("Profil speichern", use_container_width=True):
            save_settings(sett)
            st.success("Profil aktualisiert!")

    st.divider()
    st.subheader("📁 Projekte (lokal speichern / importieren)")

    # IMPORT (vom lokalen Rechner hochladen)
    imp_file = st.file_uploader("Projekt-ZIP importieren", type="zip", help="ZIP auswählen (vom PC) und importieren.")
    if imp_file:
        try:
            new_slug = import_project_zip(imp_file)
            st.success(f"Import erfolgreich: {new_slug}")
            st.session_state["active_slug"] = new_slug
            st.rerun()
        except Exception as e:
            st.error(f"Import fehlgeschlagen: {e}")

    projs = get_projects()
    selected = st.selectbox("Wähle ein Projekt", ["-- NEU ANLEGEN --"] + projs)

    if selected == "-- NEU ANLEGEN --":
        with st.form("new_p"):
            n_name = st.text_input("Kundenname", placeholder="z.B. Muster GmbH")
            if st.form_submit_button("➕ Projekt erstellen", use_container_width=True):
                if not n_name.strip():
                    st.error("Bitte einen Kundenname eingeben.")
                else:
                    base_slug = f"{datetime.now().strftime('%y%m%d')}_{slugify(n_name)}"
                    slug = base_slug
                    i = 2
                    while (PROJECTS_DIR / slug).exists():
                        slug = f"{base_slug}_{i}"
                        i += 1

                    p_path = PROJECTS_DIR / slug
                    (p_path / "documents").mkdir(parents=True, exist_ok=True)

                    init = {
                        "metadata": {"id": slug, "created": str(datetime.now())},
                        "kunde": {"name": n_name, "straße": "", "plz_ort": "", "email": "", "tel": ""},
                        "planung": {"pv_prio": "Eigenverbrauch", "status": "Planung"},
                        "pv": {"total_kwp": 0.0},
                        "speicher": {"kap_netto": 0.0},
                        "notizen": "",
                    }
                    with (p_path / "state.json").open("w", encoding="utf-8") as f:
                        json.dump(init, f, indent=4, ensure_ascii=False)

                    st.session_state["active_slug"] = slug
                    st.rerun()
    else:
        st.session_state["active_slug"] = selected if selected in projs else None

    # Aktionen
    if st.session_state["active_slug"]:
        st.divider()
        st.info(f"Aktiv: **{st.session_state['active_slug']}**")
        ensure_project_structure(st.session_state["active_slug"])

        # EXPORT (lokal speichern = Download)
        z_data = export_project(st.session_state["active_slug"])
        st.download_button(
            "💾 Projekt lokal speichern (ZIP-Export)",
            z_data,
            file_name=f"{st.session_state['active_slug']}.zip",
            use_container_width=True,
            help="Speichert das Projekt als ZIP auf deinem PC (Download).",
        )

        rep = export_report_zip(st.session_state["active_slug"])
        st.download_button(
            "📦 Bericht + Anlagen (ZIP + index.html)",
            rep,
            file_name=f"bericht_{st.session_state['active_slug']}.zip",
            use_container_width=True,
        )

        st.divider()
        confirm = st.checkbox("Löschen bestätigen", value=False)
        if st.button("🗑️ Projekt löschen", use_container_width=True, type="secondary", disabled=not confirm):
            shutil.rmtree(PROJECTS_DIR / st.session_state["active_slug"], ignore_errors=True)
            st.session_state["active_slug"] = None
            st.rerun()

        if st.button("❌ Schließen", use_container_width=True):
            st.session_state["active_slug"] = None
            st.rerun()

# =================================================================
# 6) HAUPTBEREICH
# =================================================================
if not st.session_state["active_slug"]:
    st.title("⚡ Energy Intelligence Planning System")
    st.write("Wähle ein Projekt in der Seitenleiste aus oder erstelle ein neues, um die Planung zu starten.")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        last = get_projects()[:5]
        st.info("### Letzte Projekte\n" + ("\n".join([f"- {p}" for p in last]) if last else "- (keine)"))
    with col_info2:
        ok_projects = "✅" if PROJECTS_DIR.exists() else "❌"
        ok_profiles = "✅" if (BASE_DIR / "assets" / "profiles").exists() else "⚠️"
        st.success(
            "### System-Status\n"
            f"- Projekte: {ok_projects}\n"
            f"- Musterprofile: {ok_profiles}\n"
            f"- Zeit: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

else:
    slug = st.session_state["active_slug"]
    proj_dir = PROJECTS_DIR / slug
    state_path = proj_dir / "state.json"

    if not state_path.exists():
        st.error("state.json fehlt im Projekt. Bitte Projekt erneut importieren/erstellen.")
        st.stop()

    with state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sett = load_settings()

    h_col1, h_col2 = st.columns([3, 1])
    h_col1.title(f"Projekt: {data.get('kunde', {}).get('name', '(ohne Name)')}")
    logo_rel = sett.get("logo_path")
    if logo_rel:
        logo_abs = BASE_DIR / logo_rel
        if logo_abs.exists():
            h_col2.image(str(logo_abs), width=160)

    st.write("---")
    m1, m2, m3, m4 = st.columns(4)
    pv_val = data.get("pv", {}).get("total_kwp", 0.0)
    bat_val = data.get("speicher", {}).get("kap_netto", data.get("speicher", {}).get("kap", 0.0))
    status = data.get("planung", {}).get("status", "Planung")

    m1.metric("PV-Leistung", f"{pv_val} kWp")
    m2.metric("Speicher", f"{bat_val} kWh")
    m3.metric("Projektstatus", status)
    m4.metric("Kunde", data.get("kunde", {}).get("plz_ort") or "n.a.")

    st.write("### 🧭 Workflows")
    n1, n2, n3 = st.columns(3)
    with n1:
        st.button("📊 PHASE 1: Lastganganalyse", use_container_width=True, on_click=lambda: st.switch_page("pages/1_Lastganganalyse.py"))
    with n2:
        st.button("🏗️ PHASE 2: Planung & ROI", use_container_width=True, on_click=lambda: st.switch_page("pages/2_Planung.py"))
    with n3:
        st.button("📄 PHASE 3: Berichtwesen", use_container_width=True, on_click=lambda: st.switch_page("pages/3_Bericht.py"))

    st.write("---")
    tab_data, tab_docs, tab_notes = st.tabs(["👤 Stammdaten", "📂 Dokumente", "📝 Notizen"])

    with tab_data:
        with st.form("sd_form"):
            c_sd1, c_sd2 = st.columns(2)
            data.setdefault("kunde", {})
            data.setdefault("planung", {})

            data["kunde"]["name"] = c_sd1.text_input("Name / Firma", data["kunde"].get("name", ""))
            data["kunde"]["straße"] = c_sd1.text_input("Straße & Nr.", data["kunde"].get("straße", ""))
            data["kunde"]["plz_ort"] = c_sd2.text_input("PLZ & Ort", data["kunde"].get("plz_ort", ""))
            data["kunde"]["email"] = c_sd2.text_input("E-Mail", data["kunde"].get("email", ""))
            data["kunde"]["tel"] = c_sd2.text_input("Telefon", data["kunde"].get("tel", ""))

            data["planung"]["status"] = st.selectbox(
                "Status",
                ["Akquise", "Planung", "Angebot", "Bau", "Archiv"],
                index=["Akquise", "Planung", "Angebot", "Bau", "Archiv"].index(data["planung"].get("status", "Planung")),
            )

            if st.form_submit_button("Änderungen speichern"):
                with state_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                st.toast("Daten gesichert!")

    with tab_docs:
        st.subheader("Projektdateien (Datenblätter / Lastgänge / Fotos etc.)")
        doc_dir = proj_dir / "documents"
        doc_dir.mkdir(parents=True, exist_ok=True)

        up = st.file_uploader("Datei(en) hochladen", accept_multiple_files=True)
        if up:
            for uf in up:
                out = doc_dir / Path(uf.name).name
                with out.open("wb") as f:
                    f.write(uf.getbuffer())
            st.success("Upload abgeschlossen.")
            st.rerun()

        files = sorted([p for p in doc_dir.rglob("*") if p.is_file()], key=lambda p: p.name.lower())
        if files:
            for p in files:
                rel = p.relative_to(doc_dir).as_posix()
                c1, c2, c3 = st.columns([5, 2, 1])
                c1.write(f"📄 {rel}")
                c2.caption(f"{p.stat().st_size/1024:.1f} KB")
                with p.open("rb") as fb:
                    c3.download_button("Download", fb, file_name=p.name, key=f"dl_{rel}")

                dcol1, dcol2 = st.columns([1, 6])
                if dcol1.button("🗑️", key=f"del_{rel}", help="Datei löschen"):
                    try:
                        p.unlink()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Löschen fehlgeschlagen: {e}")
        else:
            st.info("Noch keine Dokumente hochgeladen.")

    with tab_notes:
        data["notizen"] = st.text_area("Interne Projektnotizen", data.get("notizen", ""), height=220)
        if st.button("Notizen speichern"):
            with state_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            st.success("Notiz gespeichert.")
