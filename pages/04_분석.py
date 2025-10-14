import os
import platform
import sys
from pathlib import Path

import pyproj
import streamlit as st

from utils.config import get_db_engine
from utils.dem_processor import run_full_analysis

# --- PROJ 데이터 디렉토리 설정 (모든 플랫폼 호환 최종 버전) ---
try:
    conda_prefix = Path(sys.prefix)
    if platform.system() == "Windows":
        proj_data_dir = conda_prefix / "Library" / "share" / "proj"
    else:
        proj_data_dir = conda_prefix / "share" / "proj"

    if proj_data_dir.exists():
        pyproj.datadir.set_data_dir(str(proj_data_dir))
except Exception:
    pass
# --- 설정 종료 ---

import shutil
import time

from utils.theme_util import apply_styles

# --- 1. 페이지 설정 및 스타일링 ---
st.set_page_config(page_title="분석 실행 - 지형 분석 서비스",
                   page_icon="⚙️",
                   layout="wide",
                   initial_sidebar_state="collapsed")
apply_styles()

# --- 2. 세션 상태 및 DB 확인 ---
if 'gdf' not in st.session_state:
    st.warning("분석할 데이터가 없습니다. 홈 페이지로 돌아가 파일을 먼저 업로드해주세요.")
    if st.button("홈으로 돌아가기"):
        st.switch_page("app.py")
    st.stop()


try:
    engine = get_db_engine()
    with engine.connect() as connection:
        pass
except Exception as e:
    st.error(f"데이터베이스에 연결할 수 없습니다. PostGIS DB가 실행 중인지 확인하세요. 에러: {e}")
    st.stop()

# --- 3. 도우미 함수 ---


# --- 4. 페이지 헤더 ---
st.markdown('''<div class="page-header" style="margin-top: -1.5rem;"><h1>분석 실행</h1><p>선택하신 항목에 대한 분석을 실행합니다. 이 작업은 몇 분 정도 소요될 수 있습니다.</p></div>''', unsafe_allow_html=True)

# --- 5. 주요 분석 로직 ---
if 'dem_results' not in st.session_state:
    user_gdf_original = st.session_state.gdf
    selected_types = st.session_state.get('selected_analysis_types', [])
    uploaded_file_name = st.session_state.get('uploaded_file_name', '')
    subbasin_name = uploaded_file_name.split('_')[0].split('.')[
        0] if uploaded_file_name else ''

    with st.status("분석 진행 중...", expanded=True) as status:
        try:
            dem_results = run_full_analysis(
                user_gdf_original, selected_types, subbasin_name)
            st.session_state.dem_results = dem_results
            status.update(label="분석 완료!", state="complete", expanded=False)

        except Exception as e:
            import traceback
            st.error(f"분석에 실패했습니다: {e}")
            st.code(traceback.format_exc())
            st.stop()

    st.success("모든 분석이 완료되었습니다. 결과 페이지로 이동합니다.")
    time.sleep(3)
    st.switch_page("pages/05_자료다운.py")