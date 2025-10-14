import streamlit as st
import os


def apply_styles():
    """
    전역 CSS 스타일시트를 로드하고 적용합니다.
    """
    css_file_path = os.path.join("static", "main.css")
    try:
        with open(css_file_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"치명적 오류: 전역 스타일시트를 '{css_file_path}'에서 찾을 수 없습니다.")