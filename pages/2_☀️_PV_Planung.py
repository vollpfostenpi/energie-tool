import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Seitenkonfiguration
st.set_page_config(page_title="PV & Energie Planung", page_icon="☀️", layout="wide")

st.title("☀️ Integrierte Energieplanung")

# --- DATEN-CHECK (Holen der Lastgangdaten aus Session State oder Default) ---
# Falls auf Seite 1 ein Lastgang geladen wurde, könnten wir ihn hier nutzen.
# Hier definieren wir einen Dummy-Verbrauch, falls nichts geladen ist.
jahresverbrauch_default = 50000 

# --- TABS ERSTELLEN ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Übersicht & Empfehlung", 
    "🔌 Photovoltaik", 
    "🔋 Speicheroptimierung", 
    "🚗 Ladeinfrastruktur"
])

# --- SIDEBAR FÜR GLOBALE PARAMETER ---
st.sidebar.header("Globale Parameter")
strompreis = st.sidebar.slider("Strompreis (Cent/kWh)", 20, 60, 35) / 100
co2_faktor = 0.385 

# --- TAB 2: PHOTOVOLTAIK (Eingabe zuerst, damit Übersicht rechnen kann) ---
with tab2:
    st.header("PV-Anlagenplanung")
    col_pv1, col_pv2 = st.columns(2)
    with col_pv1:
        kwp_leistung = st.number_input("Geplante Leistung (kWp)", 0, 1000, 30)
        dachausrichtung = st.selectbox("Ausrichtung", ["Süd", "Ost/West", "Nord"])
    with col_pv2:
        invest_pv = st.number_input("Investitionskosten PV (€/kWp)", value=1100)
        ertrag_faktor = {"Süd": 1050, "Ost/West": 850, "Nord": 600}[dachausrichtung]
    
    pv_ertrag_jahr = kwp_leistung * ertrag_faktor

# --- TAB 3: SPEICHEROPTIMIERUNG ---
with tab3:
    st.header("Batteriespeicher & Arbitrage")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        speicher_geplant = st.toggle("Speicher planen?", value=False)
        speicher_kapazitaet = st.number_input("Speicherkapazität (kWh)", 0, 500, 10 if speicher_geplant else 0)
    with col_s2:
        arbitrage_wunsch = st.checkbox("Arbitrage-Handel (Börsenpreis-Optimierung)")
        if arbitrage_wunsch:
            st.info("💡 System gleicht Ladezyklen mit Prognosen der Strombörse (EPEX Spot) ab.")

# --- TAB 4: LADEINFRASTRUKTUR ---
with tab4:
    st.header("Ladeinfrastruktur (Wallboxen)")
    anzahl_lp = st.number_input("Anzahl Ladepunkte", 0, 50, 2)
    fahrleistung_jahr = st.slider("Ø Fahrleistung pro Fahrzeug (km/Jahr)", 5000, 40000, 15000)
    ev_verbrauch = 0.20 # 20 kWh/100km
    
    zusatzbedarf_ev = anzahl_lp * (fahrleistung_jahr / 100) * ev_verbrauch

# --- TAB 1: ÜBERSICHT & EMPFEHLUNG (Die Logik-Zentrale) ---
with tab1:
    st.header("Zusammenfassung & Strategie")
    
    # Empfehlungs-Logik
    empfehlung = ""
    if kwp_leistung > 0 and not speicher_geplant:
        empfehlung = f"⚠️ **Empfehlung:** Bei {kwp_leistung} kWp PV-Leistung wird ein Speicher von ca. {kwp_leistung * 0.8:.1f} kWh zur Eigenverbrauchsoptimierung empfohlen."
    elif kwp_leistung == 0 and anzahl_lp > 0:
        empfehlung = "⚠️ **Analyse:** Ladeinfrastruktur ohne PV geplant. Hohe Kosten durch Netzbezug. PV-Anlage prüfen!"
    else:
        empfehlung = "✅ Die Systemkonfiguration sieht stimmig aus."

    st.success(empfehlung)

    # Kennzahlen
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PV-Ertrag", f"{pv_ertrag_jahr:,.0f} kWh/a")
    m2.metric("EV-Bedarf", f"{zusatzbedarf_ev:,.0f} kWh/a")
    m3.metric("Speicher", f"{speicher_kapazitaet} kWh")
    m4.metric("CO2-Ersparnis", f"{(pv_ertrag_jahr * co2_faktor)/1000:.1f} t/a")

    # Arbitrage / Börsen-Visualisierung (Dummy-Prognose)
    if arbitrage_wunsch:
        st.divider()
        st.subheader("Börsenpreis-Prognose & Ladestrategie")
        st.caption("Beispielhafter Verlauf für Arbitrage-Handel")
        
        stunden = list(range(24))
        preise = [15, 12, 10, 9, 11, 14, 22, 28, 25, 20, 18, 17, 16, 15, 18, 24, 32, 35, 30, 25, 22, 20, 18, 16]
        df_boerse = pd.DataFrame({"Stunde": stunden, "Preis (ct/kWh)": preise})
        
        fig_boerse = px.line(df_boerse, x="Stunde", y="Preis (ct/kWh)", color_discrete_sequence=['#FFA500'])
        # Markierung der Ladefenster
        fig_boerse.add_vrect(x0=2, x1=5, fillcolor="green", opacity=0.2, annotation_text="Laden")
        fig_boerse.add_vrect(x1=18, x0=16, fillcolor="red", opacity=0.2, annotation_text="Entladen")
        
        st.plotly_chart(fig_boerse, use_container_width=True)

    # Investitions-Übersicht
    st.divider()
    st.subheader("Kostenübersicht")
    total_invest = (kwp_leistung * invest_pv) + (speicher_kapazitaet * 700) + (anzahl_lp * 1500)
    st.info(f"Geschätztes Gesamtinvest: **{total_invest:,.2f} €**")
