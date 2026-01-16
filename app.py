# app.py
from __future__ import annotations
import io
import os
import json
import shutil
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.common import inject_css, card, fmt_de

# -----------------------------
# Pfade / Helper
# -----------------------------
APP_DIR = os.getcwd()
CONFIG_DIR = os.path.join(APP_DIR, "config")
ASSETS_DIR = os.path.join(APP_DIR, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
INSTALLER_JSON = os.path.join(CONFIG_DIR, "installer.json")
PROJECTS_DIR = os.path.join(APP_DIR, "projects")

def ensure_dirs():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(PROJECTS_DIR, exist_ok=True)

def load_json(path: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default or {}

def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def slugify(s: str) -> str:
    keep = "-_"
    s = "".join(ch if ch.isalnum() or ch in keep else "-" for ch in s.strip())
    while "--" in s: s = s.replace("--", "-")
    return s.strip("-").lower() or "projekt"

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

ensure_dirs()
inject_css()
st.set_page_config(page_title="Energie-Tool", page_icon="⚙️", layout="wide")

# -----------------------------
# Header / Logo
# -----------------------------
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, caption="Installateur-Logo", use_container_width=True)
with col_title:
    st.title("⚙️ Energie-Tool – Start")

# -----------------------------
# Installateur-Stammdaten (persistent)
# -----------------------------
with card("🏢 Installateur – Stammdaten (persistiert)"):
    inst = load_json(INSTALLER_JSON, {
        "installer_name": "",
        "installer_addr": "",
        "installer_contact": "",
        "default_city": "",
    })
    c1, c2 = st.columns(2)
    with c1:
        installer_name = st.text_input("Firmenname", value=inst.get("installer_name",""),
                                       help="Wird in Berichten als Absender genutzt.")
        installer_addr = st.text_area("Adresse", value=inst.get("installer_addr",""),
                                      help="Mehrzeilig, erscheint im Berichtskopf.")
        installer_contact = st.text_input("Kontakt (Telefon/E-Mail)", value=inst.get("installer_contact",""),
                                          help="Für Rückfragen im Bericht.")
        default_city = st.text_input("Standard-Ort/Region (optional)", value=inst.get("default_city",""),
                                     help="Voreinstellung für Projekte.")
    with c2:
        st.markdown("**Logo** (PNG, empfohlen mind. 300×300)")
        logo_up = st.file_uploader("Logo hochladen", type=["png","jpg","jpeg"],
                                   help="Logo wird lokal unter assets/logo.png gespeichert.")
        if st.button("Logo speichern"):
            if logo_up:
                os.makedirs(ASSETS_DIR, exist_ok=True)
                # immer als PNG speichern (Streamlit gibt Bytes, wir kopieren 1:1)
                with open(LOGO_PATH, "wb") as f:
                    f.write(logo_up.getvalue())
                st.success("Logo gespeichert.")
                st.experimental_rerun()
            else:
                st.warning("Bitte zuerst ein Logo auswählen.")

        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, caption="Aktuelles Logo", use_container_width=True)

    if st.button("Stammdaten speichern"):
        data = {
            "installer_name": installer_name.strip(),
            "installer_addr": installer_addr.strip(),
            "installer_contact": installer_contact.strip(),
            "default_city": default_city.strip(),
        }
        save_json(INSTALLER_JSON, data)
        st.success("Stammdaten gespeichert und bleiben erhalten (bis zur nächsten Änderung).")
        # in Session spiegeln (für Berichte auf Unterseiten)
        st.session_state["installer_name"] = data["installer_name"]
        st.session_state["installer_addr"] = data["installer_addr"]
        st.session_state["installer_contact"] = data["installer_contact"]

# -----------------------------
# Neues Projekt: Kundendaten & Notizen
# -----------------------------
with card("🧑‍🤝‍🧑 Neues Projekt anlegen"):
    c1, c2, c3 = st.columns(3)
    with c1:
        project_name = st.text_input("Projektname", value="", help="Eindeutiger Name, z. B. 'Familie Müller EFH'.")
        customer_name = st.text_input("Kunde – Name/Firma", value="", help="Kundenname für Bericht/Ordner.")
        site_name = st.text_input("Standort/Objekt", value=load_json(INSTALLER_JSON).get("default_city",""),
                                  help="Adresse/Ort des Projekts.")
    with c2:
        customer_street = st.text_input("Kunde – Straße & Nr.", value="", help="Straße und Hausnummer.")
        customer_zip = st.text_input("Kunde – PLZ", value="", help="Postleitzahl.")
        customer_city = st.text_input("Kunde – Ort", value="", help="Ort.")
    with c3:
        customer_phone = st.text_input("Kunde – Telefon", value="", help="Telefonnummer.")
        customer_email = st.text_input("Kunde – E-Mail", value="", help="E-Mail-Adresse.")
        customer_ref = st.text_input("Interne Referenz (optional)", value="", help="z. B. Angebotsnummer.")

    notes = st.text_area("🧾 Notizen / benötigte Unterlagen",
                         help="z. B. Vollmacht, Zählernummer, Fotos, Netzanfrage, etc.")

    proj_files = st.file_uploader("Projektunterlagen hinzufügen (mehrere möglich)",
                                  type=["pdf","png","jpg","jpeg","csv","xlsx","zip"],
                                  accept_multiple_files=True,
                                  help="Beliebige Anhänge; werden im Projektordner gespeichert.")

    if st.button("➕ Projekt anlegen"):
        if not project_name or not customer_name:
            st.error("Bitte mindestens **Projektname** und **Kundenname** angeben.")
        else:
            slug = slugify(project_name)
            pdir = os.path.join(PROJECTS_DIR, slug)
            os.makedirs(pdir, exist_ok=True)
            # Metadaten schreiben
            meta = {
                "status": "open",
                "created": now_ts(),
                "project_name": project_name,
                "customer_name": customer_name,
                "site_name": site_name,
                "customer": {
                    "street": customer_street, "zip": customer_zip, "city": customer_city,
                    "phone": customer_phone, "email": customer_email, "ref": customer_ref
                },
                "notes": notes,
            }
            save_json(os.path.join(pdir, "metadata.json"), meta)
            # Uploads speichern
            if proj_files:
                uf_dir = os.path.join(pdir, "uploads")
                os.makedirs(uf_dir, exist_ok=True)
                for f in proj_files:
                    with open(os.path.join(uf_dir, f.name), "wb") as out:
                        out.write(f.getvalue())
            st.success(f"Projekt '{project_name}' angelegt (Ordner: projects/{slug})")

# -----------------------------
# Aktuelle Sitzungsdaten sichern in Projekt
# -----------------------------
def _session_series_to_csv(path: str, ser: pd.Series):
    if ser is None: return
    df = ser.to_frame()
    df.to_csv(path, index=True, encoding="utf-8")

def build_pdf_bytes(mode: str) -> bytes:
    """Erzeugt PDF wie im Export-Tab (leicht kompakt), basierend auf Session-Daten."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as e:
        st.error(f"ReportLab fehlt: {e}. Bitte `pip install reportlab` ausführen.")
        return b""

    lg_p = st.session_state.get("lg_p")        # kW
    pv15 = st.session_state.get("pv15")        # kW
    sim15 = st.session_state.get("sim_15min")  # Ergebnisse vom Export-Tab (falls schon genutzt)
    comps = st.session_state.get("components", {})
    roof_layouts = st.session_state.get("roof_layouts", {})

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    story = []

    inst = load_json(INSTALLER_JSON, {})
    installer_name = inst.get("installer_name","")
    installer_addr = inst.get("installer_addr","")
    installer_contact = inst.get("installer_contact","")

    project_name = st.session_state.get("project_name", "Projekt")
    customer_name = st.session_state.get("customer_name", "")
    customer_contact = st.session_state.get("customer_contact", "")
    site_name = st.session_state.get("site_name", "")

    story.append(Paragraph(f"Energieanalyse – {project_name}", styles["Title"]))
    story.append(Spacer(1, 0.2*cm))
    left = f"<b>{installer_name}</b><br/>{installer_addr.replace('\\n','<br/>')}<br/>{installer_contact}"
    right = f"<b>Kunde:</b> {customer_name}<br/>{customer_contact}<br/><b>Standort:</b> {site_name}"
    tbl = Table([[Paragraph(left, styles["Normal"]), Paragraph(right, styles["Normal"])]], colWidths=[10*cm, 8.5*cm])
    tbl.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(tbl)
    story.append(Spacer(1, 0.3*cm))

    # Wenn Simulation nicht vorhanden → Kurzstatistik aus lg_p/pv15
    if sim15 is None and (lg_p is not None) and (pv15 is not None):
        df = pd.DataFrame({"load": lg_p.reindex(pv15.index).interpolate(limit_direction="both"), "pv": pv15})
        df["self_kWh"] = (df[["load","pv"]].min(axis=1) * 0.25)
        df["import_kWh"] = (pd.Series((df["load"]-df["pv"]).clip(lower=0)) * 0.25)
        df["export_kWh"] = (pd.Series((df["pv"]-df["load"]).clip(lower=0)) * 0.25)
        total_pv_kwh = float((df["pv"]*0.25).sum())
        total_load_kwh = float((df["load"]*0.25).sum())
        self_cons_kwh = float(df["self_kWh"].sum())
        import_kwh = float(df["import_kWh"].sum())
        export_kwh = float(df["export_kWh"].sum())
        scr = self_cons_kwh / total_pv_kwh if total_pv_kwh > 0 else None
        ssr = self_cons_kwh / total_load_kwh if total_load_kwh > 0 else None
    else:
        s = sim15 or {}
        total_pv_kwh = s.get("total_pv_kwh", 0.0)
        total_load_kwh = s.get("total_load_kwh", 0.0)
        self_cons_kwh = s.get("self_cons_kwh", 0.0)
        import_kwh = s.get("import_kwh", 0.0)
        export_kwh = s.get("export_kwh", 0.0)
        scr = s.get("scr")
        ssr = s.get("ssr")

    story.append(Paragraph("Ergebnisse", styles["Heading2"]))
    data = [
        ["PV-Erzeugung [kWh]", fmt_de(total_pv_kwh,0)],
        ["Last [kWh]", fmt_de(total_load_kwh,0)],
        ["Eigenverbrauch [kWh]", fmt_de(self_cons_kwh,0)],
        ["Netzbezug [kWh]", fmt_de(import_kwh,0)],
        ["Einspeisung [kWh]", fmt_de(export_kwh,0)],
        ["EV-Quote", f"{scr:.2%}" if scr is not None else "—"],
        ["Autarkiegrad", f"{ssr:.2%}" if ssr is not None else "—"],
    ]
    t = Table(data, hAlign="LEFT", colWidths=[7*cm,7*cm])
    t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.3,colors.grey)]))
    story.append(t)
    story.append(Spacer(1, 0.2*cm))

    # Komponenten (modusabhängig)
    story.append(Paragraph("Komponenten", styles["Heading2"]))
    if mode == "Kunde":
        rows = [
            ["Modul", f"Leistung {fmt_de(comps.get('module',{}).get('wp',0),0)} Wp pro Modul"],
            ["Wechselrichter", f"AC Nennleistung {fmt_de(comps.get('inverter',{}).get('ac_kw',0),1)} kW"],
            ["Speicher", f"Kapazität {fmt_de(comps.get('battery',{}).get('kWh',0),0)} kWh"],
        ]
    else:
        rows = [
            ["Modul", f"{comps.get('module',{}).get('manufacturer','')} – {comps.get('module',{}).get('type','')} – {fmt_de(comps.get('module',{}).get('wp',0),0)} Wp"],
            ["Wechselrichter", f"{comps.get('inverter',{}).get('manufacturer','')} – {comps.get('inverter',{}).get('type','')} – {fmt_de(comps.get('inverter',{}).get('ac_kw',0),1)} kW AC"],
            ["Speicher", f"{comps.get('battery',{}).get('manufacturer','')} – {comps.get('battery',{}).get('type','')} – {fmt_de(comps.get('battery',{}).get('kWh',0),0)} kWh"],
        ]
    t2 = Table(rows, hAlign="LEFT", colWidths=[6*cm, 9*cm])
    t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.3,colors.grey)]))
    story.append(t2)

    # Belegungspläne
    if roof_layouts:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Belegungspläne", styles["Heading2"]))
        for name, meta in roof_layouts.items():
            if meta and meta.get("bytes"):
                story.append(Paragraph(str(name), styles["Heading3"]))
                story.append(RLImage(io.BytesIO(meta["bytes"]), width=16*cm))
                story.append(Spacer(1, 0.2*cm))

    doc.build(story)
    return buf.getvalue()

with card("💾 Sitzungsdaten in Projekt speichern"):
    st.caption("Speichere aktuelle Lastgänge, PV-Profile und Berichte in einem Projektordner.")
    proj_select = None
    # Liste existierender Projekte
    projects = sorted([d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR,d))])
    if projects:
        proj_label = st.selectbox("Projekt auswählen", projects, help="Zielordner für die Ablage.")
        proj_select = os.path.join(PROJECTS_DIR, proj_label)
    else:
        st.info("Noch keine Projekte vorhanden. Lege oben ein Projekt an, um Daten zu speichern.")

    col_a, col_b = st.columns(2)
    with col_a:
        mode_pdf = st.radio("PDF-Modus", ["Kunde", "Intern"], horizontal=True,
                            help="Kunde: nur Eckdaten. Intern: inkl. Typen & Datenblätter.")
    with col_b:
        also_zip = st.checkbox("ZIP (PDF + Datenblätter) miterzeugen", value=True,
                               help="Bei 'Intern' werden Datenblätter in die ZIP gepackt (falls vorhanden).")

    if st.button("Speichern / (Neu)Erzeugen"):
        if not proj_select:
            st.error("Bitte ein Projekt auswählen.")
        else:
            # 1) Lastgang / PV in CSV ablegen
            lg_p = st.session_state.get("lg_p")
            pv15 = st.session_state.get("pv15")
            out_data = os.path.join(proj_select, "data")
            os.makedirs(out_data, exist_ok=True)
            if isinstance(lg_p, pd.Series):
                _session_series_to_csv(os.path.join(out_data, "lastgang_kW.csv"), lg_p)
            if isinstance(pv15, pd.Series):
                _session_series_to_csv(os.path.join(out_data, "pv_15min_kW.csv"), pv15)

            # 2) PDF erzeugen und speichern
            pdf_bytes = build_pdf_bytes(mode_pdf)
            if pdf_bytes:
                pdf_name = f"Bericht_{mode_pdf}_{now_ts()}.pdf"
                with open(os.path.join(proj_select, pdf_name), "wb") as f:
                    f.write(pdf_bytes)
                st.success(f"PDF gespeichert: {pdf_name}")
                st.download_button("PDF herunterladen", data=pdf_bytes, file_name=pdf_name, mime="application/pdf")

            # 3) (optional) ZIP erzeugen
            if also_zip:
                zname = f"Doku_{mode_pdf}_{now_ts()}.zip"
                import zipfile
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    if pdf_bytes:
                        zf.writestr(pdf_name, pdf_bytes)
                    if mode_pdf == "Intern":
                        # Datenblätter hinzufügen
                        for key in ("module","inverter","battery"):
                            ds = (st.session_state.get("datasheets") or {}).get(key)
                            if ds and ds.get("bytes"):
                                zf.writestr(f"datenblaetter/{ds.get('name') or key+'.bin'}", ds["bytes"])
                    # Rohdaten (falls vorhanden)
                    if isinstance(lg_p, pd.Series):
                        zf.writestr("data/lastgang_kW.csv", lg_p.to_csv().encode("utf-8"))
                    if isinstance(pv15, pd.Series):
                        zf.writestr("data/pv_15min_kW.csv", pv15.to_csv().encode("utf-8"))
                with open(os.path.join(proj_select, zname), "wb") as f:
                    f.write(zbuf.getvalue())
                st.success(f"ZIP gespeichert: {zname}")

            # 4) Metadaten aktualisieren (bleibt 'open'; Abschluss unten)
            meta_path = os.path.join(proj_select, "metadata.json")
            meta = load_json(meta_path, {})
            meta["updated"] = now_ts()
            save_json(meta_path, meta)

# -----------------------------
# Projektübersicht
# -----------------------------
with card("📂 Projekte"):
    tabs = st.tabs(["🟡 In Bearbeitung", "🟢 Abgeschlossen", "🗑️ Verwalten"])
    # Einlesen aller Projekte
    all_projects = []
    for d in sorted([d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR,d))]):
        meta = load_json(os.path.join(PROJECTS_DIR, d, "metadata.json"), {})
        meta["_slug"] = d
        all_projects.append(meta)

    def render_table(status: str):
        lst = [p for p in all_projects if p.get("status","open") == status]
        if not lst:
            st.info("Keine Projekte gefunden.")
            return
        for p in lst:
            with st.expander(f"{p.get('project_name','(ohne Namen)')} — {p.get('customer_name','')}", expanded=False):
                st.write(f"**Angelegt:** {p.get('created','-')}  ·  **Zuletzt:** {p.get('updated','-')}")
                st.write(f"**Standort:** {p.get('site_name','-')}")
                st.write(f"**Notizen:** {p.get('notes','—')}")
                pdir = os.path.join(PROJECTS_DIR, p["_slug"])

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("PDF (Kunde) neu erzeugen", key=f"pdf_k_{p['_slug']}"):
                        pdf = build_pdf_bytes("Kunde")
                        if pdf:
                            fname = f"Bericht_Kunde_{now_ts()}.pdf"
                            with open(os.path.join(pdir, fname), "wb") as f:
                                f.write(pdf)
                            st.download_button("Download PDF", data=pdf, file_name=fname, mime="application/pdf", key=f"dld1_{p['_slug']}")
                with col2:
                    if st.button("PDF (Intern) neu erzeugen", key=f"pdf_i_{p['_slug']}"):
                        pdf = build_pdf_bytes("Intern")
                        if pdf:
                            fname = f"Bericht_Intern_{now_ts()}.pdf"
                            with open(os.path.join(pdir, fname), "wb") as f:
                                f.write(pdf)
                            st.download_button("Download PDF", data=pdf, file_name=fname, mime="application/pdf", key=f"dld2_{p['_slug']}")
                with col3:
                    if status == "open" and st.button("Als abgeschlossen markieren", key=f"close_{p['_slug']}"):
                        meta = load_json(os.path.join(pdir, "metadata.json"), {})
                        meta["status"] = "done"
                        meta["closed"] = now_ts()
                        save_json(os.path.join(pdir, "metadata.json"), meta)
                        st.success("Projekt abgeschlossen.")
                        st.experimental_rerun()
                with col4:
                    # Liste vorhandener Dateien zur direkten Auswahl
                    files = [f for f in os.listdir(pdir) if os.path.isfile(os.path.join(pdir,f)) and f.lower().endswith((".pdf",".zip"))]
                    if files:
                        pick = st.selectbox("Vorhandene Dateien", files, key=f"files_{p['_slug']}")
                        if pick:
                            with open(os.path.join(pdir, pick), "rb") as f:
                                data = f.read()
                            st.download_button("Ausgewählte Datei herunterladen", data=data, file_name=pick, key=f"dl_any_{p['_slug']}")

                # Rohdaten-Vorschau
                data_dir = os.path.join(pdir, "data")
                if os.path.isdir(data_dir):
                    lg_csv = os.path.join(data_dir, "lastgang_kW.csv")
                    pv_csv = os.path.join(data_dir, "pv_15min_kW.csv")
                    if os.path.exists(lg_csv) and os.path.exists(pv_csv):
                        df = pd.DataFrame({
                            "load": pd.read_csv(lg_csv, index_col=0, parse_dates=True).iloc[:96,0],
                            "pv": pd.read_csv(pv_csv, index_col=0, parse_dates=True).iloc[:96,0],
                        })
                        st.plotly_chart(px.area(df, title="Vorschau (erster Tag)"), use_container_width=True)

    with tabs[0]:
        render_table("open")
    with tabs[1]:
        render_table("done")
    with tabs[2]:
        st.warning("Vorsicht: Löschen ist endgültig.")
        del_proj = st.selectbox("Projekt zum Löschen wählen", ["—"] + [p["_slug"] for p in all_projects])
        if del_proj != "—" and st.button("Projekt löschen"):
            shutil.rmtree(os.path.join(PROJECTS_DIR, del_proj), ignore_errors=True)
            st.success("Projekt gelöscht.")
            st.experimental_rerun()

# -----------------------------
# Hinweis
# -----------------------------
st.caption("Tipp: Erzeuge Lastgänge & PV auf den Seiten 📊/☀️. Den Bericht kannst du hier jederzeit neu erzeugen und projektweise speichern.")
