import streamlit as st
import os


def apply_styles():
    """
    Loads and applies the global CSS stylesheet.
    """
    css_file_path = os.path.join("static", "main.css")
    try:
        with open(css_file_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Fatal Error: Global stylesheet not found at '{css_file_path}'.")
