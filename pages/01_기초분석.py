import os
import platform
import sys
from pathlib import Path

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

import logging
import traceback

import folium
import geopandas as gpd
import streamlit.components.v1 as components
from streamlit_folium import st_folium

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

            # --- Validation for too many map sheets ---
            if len(st.session_state.matched_sheets) > 60:
                st.error("분석 영역이 너무 넓습니다(관련 도엽 60개 초과). 더 작은 영역을 선택하여 다시 시도해주세요.")
                # Clear session state before going back
                for key in list(st.session_state.keys()):
                    if key != 'upload_counter':
                        del st.session_state[key]
                if st.button("홈으로 돌아가기"):
                    st.switch_page("app.py")
                st.stop()
            # --- End Validation ---

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
cols = st.columns([0.95, 0.05])
with cols[0]:
    st.markdown("### 파일 분석 결과")
with cols[1]:
    if st.button("🏠", help="홈 화면으로 돌아갑니다.", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key != 'upload_counter':
                del st.session_state[key]
        st.switch_page("app.py")
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

st.markdown(f"#### 🗺️ 관련 도엽 번호 및 위치 ({len(map_sheets)}개)")
if map_sheets:
    st.info(f"대표 도엽: **{map_sheets[0]}** 외 {len(map_sheets) - 1}개")
    with st.expander("상세 지도 보기", expanded=True):
        # --- New Folium Map Implementation ---
        try:
            # Retrieve data from session state
            target_gdf = st.session_state.gdf
            map_results = st.session_state.map_index_results
            index_gdf = map_results.get('index_gdf')
            target_sheets = st.session_state.matched_sheets

            if index_gdf is None:
                st.warning("지도 시각화를 위한 도엽 색인 원본 데이터가 없습니다.")
            else:
                # ===== 시각화용 좌표계 변환 (EPSG:4326) =====
                index_4326 = index_gdf.to_crs(epsg=4326)
                target_4326 = target_gdf.to_crs(epsg=4326)
                target_poly = index_4326[index_4326['MAPIDCD_NO'].isin(
                    target_sheets)]

                # ===== Folium 지도 생성 =====
                if not target_4326.empty:
                    center = target_4326.union_all().centroid
                    # OpenStreetMap을 기본 지도로 설정하여 한글 지명 지원 및 기본 선택
                    m = folium.Map(
                        location=[center.y, center.x], zoom_start=12, tiles="OpenStreetMap")

                    # Add other tile layers as options
                    # folium.TileLayer(
                    #    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri', name='위성 지도').add_to(m)

                    # 원본 쉐이프 파일 경계 추가 (초록색)
                    folium.GeoJson(
                        target_4326,
                        name="분석 영역",
                        style_function=lambda x: {
                            'color': 'green', 'weight': 3, 'fill': False}
                    ).add_to(m)

                    # 해당 도엽 추가 (빨간색)
                    folium.GeoJson(
                        target_poly,
                        name="관련 도엽",
                        style_function=lambda x: {
                            'color': 'red', 'weight': 2, 'fillOpacity': 0.3},
                        tooltip=folium.GeoJsonTooltip(
                            fields=['MAPIDCD_NO'], aliases=['도엽 번호:'])
                    ).add_to(m)

                    folium.LayerControl().add_to(m)

                    # Render the map
                    st_folium(m, width='100%', height=500)
                else:
                    st.warning("분석 영역의 지오메트리가 비어 있어 지도를 표시할 수 없습니다.")

        except Exception as e:
            st.error(f"지도 생성 중 오류가 발생했습니다: {e}")
            # Fallback to old image if it exists
            preview_image = st.session_state.map_index_results.get(
                'preview_image')
            if preview_image:
                st.image(preview_image, caption="오류 발생: 도엽 참조 위치 이미지로 대체 표시합니다.")
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

# --- Initialize selection state ---
if 'analysis_selections' not in st.session_state:
    st.session_state.analysis_selections = {
        label: (analysis_items[label] == 'dem_group') 
        for label in analysis_items
    }

# --- Display stateful buttons in columns ---
cols = st.columns(4)
for i, label in enumerate(analysis_items.keys()):
    with cols[i % 4]:
        # Use primary type for selected, secondary for unselected
        button_type = "primary" if st.session_state.analysis_selections.get(label) else "secondary"
        if st.button(
            label,
            key=f"btn_{label}",
            use_container_width=True,
            type=button_type
        ):
            # Manually toggle the state
            st.session_state.analysis_selections[label] = not st.session_state.analysis_selections[label]
            # Provide feedback and force a rerun
            new_state = "선택됨" if st.session_state.analysis_selections[label] else "선택 해제됨"
            st.toast(f'{label}: {new_state}', icon='✅')
            st.rerun()


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
        final_analysis_types = []
        for label, is_selected in st.session_state.analysis_selections.items():
            if is_selected:
                selected_key = analysis_items[label]
                if selected_key == 'dem_group':
                    final_analysis_types.extend(['elevation', 'slope', 'aspect'])
                else:
                    final_analysis_types.append(selected_key)
        
        if final_analysis_types:
            # Remove duplicates and store in session state
            st.session_state.selected_analysis_types = list(dict.fromkeys(final_analysis_types))
            st.switch_page("pages/03_처리중.py")
        else:
            st.warning("분석 항목을 하나 이상 선택해주세요.")
