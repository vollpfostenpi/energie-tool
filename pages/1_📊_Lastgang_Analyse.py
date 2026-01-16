import streamlit as st
import pandas as pd
import os
import plotly.express as px

# 1. Seitenkonfiguration
st.set_page_config(page_title="Lastgang Analyse", page_icon="📊", layout="wide")

st.title("📊 Lastgang Analyse")

# 2. Pfad zu den Musterprofilen (Unterordner)
# WICHTIG: Auf GitHub muss die Struktur so sein: assets/profiles/deine_datei.csv
assets_path = os.path.join("assets", "profiles")

# 3. Die Auswahlmöglichkeit (Toggle) in der Sidebar
st.sidebar.header("Datenquelle")
source_option = st.sidebar.radio(
    "Wie möchten Sie die Daten bereitstellen?",
    ["Eigenes Profil hochladen", "Standard Profile (Muster)"]
)

df = None

# --- FALL A: MANUELLER UPLOAD ---
if source_option == "Eigenes Profil hochladen":
    st.subheader("Eigene Datei hochladen")
    uploaded_file = st.file_uploader("Wählen Sie eine CSV- oder Excel-Datei aus", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
            else:
                df = pd.read_excel(uploaded_file)
            st.success("Eigene Datei erfolgreich geladen!")
        except Exception as e:
            st.error(f"Fehler beim Lesen der Datei: {e}")

# --- FALL B: STANDARD PROFILE AUS ASSETS/PROFILES ---
else:
    st.subheader("Standard Profile auswählen")
    if os.path.exists(assets_path):
        # Scannt den Unterordner nach Dateien
        muster_files = [f for f in os.listdir(assets_path) if f.endswith(('.csv', '.xlsx', '.xls'))]
        
        if muster_files:
            selected_muster = st.selectbox("Wählen Sie ein Musterprofil aus:", muster_files)
            full_path = os.path.join(assets_path, selected_muster)
            try:
                if selected_muster.endswith('.csv'):
                    df = pd.read_csv(full_path, sep=None, engine='python')
                else:
                    df = pd.read_excel(full_path)
                st.info(f"Standard-Profil '{selected_muster}' ist aktiv.")
            except Exception as e:
                st.error(f"Fehler beim Laden des Standard-Profils: {e}")
        else:
            st.warning(f"Keine Dateien im Ordner '{assets_path}' gefunden.")
    else:
        st.error(f"Verzeichnis nicht gefunden: {assets_path}")
        st.info("Hinweis: Stellen Sie sicher, dass der Ordner 'assets' einen Unterordner 'profiles' hat.")

# --- DATENANZEIGE & ANALYSE (erscheint nur, wenn df geladen wurde) ---
if df is not None:
    st.divider() # Optische Trennlinie
    
    with st.expander("Vorschau der geladenen Daten"):
        st.dataframe(df.head(10), use_container_width=True)

    # Spaltenauswahl für die Grafik
    cols = df.columns.tolist()
    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("Spalte für Zeitachse (X)", cols, index=0)
    with c2:
        y_col = st.selectbox("Spalte für Leistung in kW (Y)", cols, index=1 if len(cols) > 1 else 0)

    # Visualisierung
    try:
        fig = px.line(df, x=x_col, y=y_col, title=f"Lastgang Verlauf: {y_col}")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistische Auswertung
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Spitzenlast", f"{df[y_col].max():.2f} kW")
        kpi2.metric("Ø-Leistung", f"{df[y_col].mean():.2f} kW")
        kpi3.metric("Anzahl Messpunkte", len(df))
    except Exception as e:
        st.error(f"Fehler bei der Grafik-Erstellung: {e}")
else:
    # Hilfe-Hinweis, wenn noch nichts gewählt wurde
    st.lightbulb("Nutzen Sie das Menü auf der linken Seite, um Daten zu laden.")
