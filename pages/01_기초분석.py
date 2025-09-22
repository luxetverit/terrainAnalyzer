import os
import sys
from pathlib import Path
import platform
import pyproj
import streamlit as st

# --- PROJ Data Directory Configuration (Cross-Platform Final Version) ---
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
# --- End of Configuration ---

import streamlit.components.v1 as components
import logging
import traceback

import utils.map_index_finder as map_index_finder
from utils.file_processor import process_uploaded_file
from utils.region_finder import get_region_info
from utils.simple_address_finder import get_location_name
from utils.theme_util import apply_styles

# --- 1. Page Configuration and Styling ---
st.set_page_config(page_title="기초 분석 - 지형 분석 서비스",
                   page_icon="🗺️",
                   layout="wide",
                   initial_sidebar_state="collapsed")
apply_styles()

# --- 2. Session State Check ---
if 'temp_file_path' not in st.session_state:
    st.warning("업로드된 파일이 없습니다. 홈 페이지로 돌아가 파일을 업로드해주세요.")
    if st.button("홈으로 돌아가기"):
        st.switch_page("app.py")
    st.stop()

# --- 3. Initial File Processing ---
# This block runs only once after file upload
if 'initial_analysis_done' not in st.session_state:
    with st.spinner("파일을 분석하고 관련 데이터를 조회하는 중입니다..."):
        try:
            temp_file_path = st.session_state.temp_file_path
            epsg_code = st.session_state.epsg_code

            gdf = process_uploaded_file(temp_file_path, epsg_code)
            if gdf is None or gdf.empty:
                st.error("파일에서 유효한 지리 정보를 추출할 수 없습니다. 다른 파일을 시도해주세요.")
                st.stop()

            st.session_state.gdf = gdf

            # Find map sheets
            map_results = map_index_finder.find_overlapping_sheets(
                gdf, epsg_code)
            st.session_state.map_index_results = map_results
            st.session_state.matched_sheets = map_results.get(
                'matched_sheets', [])

            from utils.config import KAKAO_API_KEY
            location_info = get_location_name(gdf, epsg_code, KAKAO_API_KEY)
            st.session_state.location_info = location_info

            st.session_state.initial_analysis_done = True

        except Exception as e:
            # Log the full traceback to the terminal
            tb_str = traceback.format_exc()
            logging.error("An error occurred during file processing.")
            logging.error(tb_str)

            # Display a detailed error message in the Streamlit app
            st.error(f"""파일 처리 중 심각한 오류가 발생했습니다.

**오류 내용:**
```
{e}
```

**상세 정보 (터미널 로그 확인):**
```
{tb_str}
```
""")
            if st.button("홈으로 돌아가기"):
                # Clear session state to allow for a fresh start
                for key in list(st.session_state.keys()):
                    if key != 'upload_counter':
                        del st.session_state[key]
                st.switch_page("app.py")
            st.stop()

# --- 4. Display Analysis Results ---
st.markdown("### 파일 분석 결과")
loc_info = st.session_state.location_info
map_sheets = st.session_state.matched_sheets

st.markdown(
    f"#### 📍 위치 정보 (_{st.session_state.get('selected_epsg_name', '알 수 없음')}_)")

# Display address information cleanly
road_address = loc_info.get('road_address', '정보 없음')
jibun_address = loc_info.get('address', '정보 없음')

if '정보 없음' in road_address or '실패' in road_address or '오류' in road_address:
    st.markdown(f"**주소**: {jibun_address}")
else:
    st.markdown(f"**주소**: {road_address} (지번: {jibun_address})")

# Display Kakao Map
lat = loc_info.get('lat')
lon = loc_info.get('lon')

if lat and lon:
    with st.expander("🗺️ 위치 개요도 보기"):
        map_url = f"https://map.kakao.com/link/map/분석지역,{lat},{lon}"
        components.html(
            f'<iframe src="{map_url}" width="100%" height="400" style="border:none;"></iframe>', height=410)

st.markdown(f"#### 🗺️ 관련 도엽 번호 ({len(map_sheets)}개)")
if map_sheets:
    st.info(f"대표 도엽: **{map_sheets[0]}** 외 {len(map_sheets) - 1}개")
    with st.expander("전체 도엽 번호 및 위치 보기"):
        st.write(map_sheets)
        preview_image = st.session_state.map_index_results.get(
            'preview_image')
        if preview_image:
            st.image(preview_image, caption="도엽 참조 위치")
else:
    st.warning("관련된 도엽 정보를 찾을 수 없습니다.")


# --- 5. Analysis Options Selection ---
st.markdown("### 분석 항목 선택")

# Redefine analysis items to group DEM analysis
analysis_items = {
    "DEM 분석 (표고+경사+경사향)": "dem_group",
    "토지이용 현황": "landcover",
    "토양도": "soil",
    "수문학적 토양군": "hsg",
}

option_labels = list(analysis_items.keys())

selected_label = st.radio(
    "분석할 항목을 선택해주세요.",
    options=option_labels,
    index=0,  # Default to the first item
    horizontal=True,
)

# --- 6. Navigation ---
col1, col2 = st.columns(2)
with col1:
    if st.button("다른 파일 업로드", use_container_width=True):
        # Clear session state before going back
        for key in list(st.session_state.keys()):
            if key != 'upload_counter':
                del st.session_state[key]
        st.switch_page("app.py")

with col2:
    if st.button("선택한 항목으로 분석 진행", type="primary", use_container_width=True):
        if selected_label:
            selected_key = analysis_items[selected_label]
            
            if selected_key == 'dem_group':
                st.session_state.selected_analysis_types = ['elevation', 'slope', 'aspect']
            else:
                st.session_state.selected_analysis_types = [selected_key]
            
            st.switch_page("pages/03_처리중.py")
        else:
            st.warning("분석 항목을 선택해주세요.")
