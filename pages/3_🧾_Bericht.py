import streamlit as st
import io
from datetime import datetime
from core.core import get_official_sources
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(page_title="Bericht Export", layout="wide")
st.title("🧾 Experten-Bericht")

st.markdown("""
Generieren Sie hier eine Zusammenfassung der Ergebnisse für Kunden oder die Geschäftsführung.
Alle Daten basieren auf den offiziellen Berechnungsgrundlagen.
""")

def create_pdf_report(client_name):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    sources = get_official_sources()

    # Titel
    story.append(Paragraph(f"Energetische Analyse: {client_name}", styles['Title']))
    story.append(Paragraph(f"Erstellt am: {datetime.now().strftime('%d.%m.%Y')}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Sektion: THG & E-Flotte
    story.append(Paragraph("1. THG-Quote & E-Mobilität", styles['Heading2']))
    story.append(Paragraph(
        "Die Integration von Elektrofahrzeugen ermöglicht die Generierung von Zusatzerlösen "
        "über die Treibhausgasminderungsquote. Dies ist ein wesentlicher Faktor für die TCO-Berechnung."
    , styles['Normal']))
    story.append(Spacer(1, 10))
    
    # Quellen-Tabelle
    data = [["Thema", "Offizielle Quelle"],
            ["THG-Quote", "NOW GmbH"],
            ["CO2-Preis", "Umweltbundesamt / BEHG"],
            ["Marktdaten", "SMARD.de / BNetzA"]]
    
    t = Table(data, colWidths=[100, 300])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                           ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                           ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                           ('GRID', (0, 0), (-1, -1), 0.5, colors.black)]))
    story.append(t)
    story.append(Spacer(1, 20))

    story.append(Paragraph(f"Referenzlink THG: {sources['THG-Quote']}", styles['Italic']))

    doc.build(story)
    return buffer.getvalue()

client_name = st.text_input("Kundenname / Projektbezeichnung", value="Spedition Beispiel")

if st.button("📄 PDF-Bericht generieren"):
    pdf_bytes = create_pdf_report(client_name)
    st.download_button(
        label="📥 Download PDF",
        data=pdf_bytes,
        file_name=f"Energiebericht_{client_name.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )
