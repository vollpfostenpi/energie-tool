# ui/common.py
from __future__ import annotations
import io
import pandas as pd
import streamlit as st
from contextlib import contextmanager

# ---------- Daten/Excel ----------
def to_excel_bytes(dfs: dict[str, pd.DataFrame | pd.Series]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        for name, d in dfs.items():
            if isinstance(d, pd.Series):
                d = d.to_frame()
            d.to_excel(w, sheet_name=name)
    return out.getvalue()

def money_eur(x: float) -> str:
    return fmt_de(x, 0) + " €"

def fmt_de(x: float | int | None, decimals: int = 0) -> str:
    """Deutsch: Punkt = Tausender, Komma = Dezimal."""
    if x is None:
        return "-"
    s = f"{float(x):,.{decimals}f}"  # US: 123,456.78
    return s.replace(",", "§").replace(".", ",").replace("§", ".")

# ---------- Styles / Layout ----------
def inject_css():
    """Einmal pro App-Lauf globale CSS-Regeln injizieren."""
    if st.session_state.get("_css_injected"):
        return
    st.session_state["_css_injected"] = True
    st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    header, #MainMenu, footer { visibility: hidden; }
    .card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .card h3 { margin: 0 0 8px 0; font-size: 1.05rem; }
    .subtle { color:#6B7280; font-size: 0.9rem; margin-top:-4px; }
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div,
    .stTextArea textarea {
        padding: 6px 10px !important;
        min-height: 36px !important;
    }
    .stDownloadButton button, .stButton button {
        border-radius: 10px; padding: .5rem 1rem;
    }
    details { border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

@contextmanager
def card(title: str, subtitle: str | None = None):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subtle">{subtitle}</div>', unsafe_allow_html=True)
    yield
    st.markdown("</div>", unsafe_allow_html=True)
