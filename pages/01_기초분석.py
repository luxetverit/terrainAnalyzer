import os
import platform
import sys
from pathlib import Path

import pyproj
import streamlit as st

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

# --- 1. 페이지 설정 및 스타일링 ---
st.set_page_config(page_title="기초 분석 - 지형 분석 서비스",
                   page_icon="🗺️",
                   layout="wide",
                   initial_sidebar_state="collapsed")
apply_styles()





# --- 2. 세션 상태 확인 ---
if 'temp_file_path' not in st.session_state:
    st.warning("업로드된 파일이 없습니다. 홈 페이지로 돌아가 파일을 업로드해주세요.")
    if st.button("홈으로 돌아가기"):
        st.switch_page("app.py")
    st.stop()

# --- 3. 초기 파일 처리 ---
# 이 블록은 파일 업로드 후 한 번만 실행됩니다
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

            # 지도 도엽 찾기
            map_results = map_index_finder.find_overlapping_sheets(
                gdf, epsg_code)
            st.session_state.map_index_results = map_results
            st.session_state.matched_sheets = map_results.get(
                'matched_sheets', [])

            # --- 너무 많은 지도 도엽에 대한 유효성 검사 ---
            if len(st.session_state.matched_sheets) > 60:
                st.error("분석 영역이 너무 넓습니다(관련 도엽 60개 초과). 더 작은 영역을 선택하여 다시 시도해주세요.")
                # 돌아가기 전에 세션 상태 지우기
                for key in list(st.session_state.keys()):
                    if key != 'upload_counter':
                        del st.session_state[key]
                if st.button("홈으로 돌아가기"):
                    st.switch_page("app.py")
                st.stop()
            # --- 유효성 검사 종료 ---

            from utils.config import KAKAO_API_KEY
            location_info = get_location_name(gdf, epsg_code, KAKAO_API_KEY)
            st.session_state.location_info = location_info

            st.session_state.initial_analysis_done = True

        except Exception as e:
            # 전체 트레이스백을 터미널에 기록
            tb_str = traceback.format_exc()
            logging.error("An error occurred during file processing.")
            logging.error(tb_str)

            # Streamlit 앱에 상세 오류 메시지 표시
            st.error(f'''파일 처리 중 심각한 오류가 발생했습니다.

**오류 내용:**
```
{e}
```

**상세 정보 (터미널 로그 확인):**
```
{tb_str}
```
''')
            if st.button("홈으로 돌아가기"):
                # 새로운 시작을 위해 세션 상태 지우기
                for key in list(st.session_state.keys()):
                    if key != 'upload_counter':
                        del st.session_state[key]
                st.switch_page("app.py")
            st.stop()

# --- 4. 분석 결과 표시 ---
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

# 주소 정보를 깔끔하게 표시
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
        # --- 새로운 Folium 지도 구현 ---
        try:
            # 세션 상태에서 데이터 검색
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

                    # 다른 타일 레이어를 옵션으로 추가
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

                    # 지도 렌더링
                    st_folium(m, width='100%', height=500)
                else:
                    st.warning("분석 영역의 지오메트리가 비어 있어 지도를 표시할 수 없습니다.")

        except Exception as e:
            st.error(f"지도 생성 중 오류가 발생했습니다: {e}")
            # 이미지가 있는 경우 이전 이미지로 대체
            preview_image = st.session_state.map_index_results.get(
                'preview_image')
            if preview_image:
                st.image(preview_image, caption="오류 발생: 도엽 참조 위치 이미지로 대체 표시합니다.")
else:
    st.warning("관련된 도엽 정보를 찾을 수 없습니다.")


# --- 5. 분석 옵션 선택 ---
st.markdown("### 분석 항목 선택")

# DEM 분석을 그룹화하기 위해 분석 항목 재정의
analysis_items = {
    "DEM 분석 (표고+경사+경사향)": "dem_group",
    "토지이용 현황": "landcover",
    "토양도": "soil",
    "수문학적 토양군": "hsg",
}

# --- 선택 상태 초기화 ---
if 'analysis_selections' not in st.session_state:
    st.session_state.analysis_selections = {
        label: (analysis_items[label] == 'dem_group') 
        for label in analysis_items
    }

# --- 열에 상태 저장 버튼 표시 ---
cols = st.columns(4)
for i, label in enumerate(analysis_items.keys()):
    with cols[i % 4]:
        # 선택된 항목에는 primary, 선택되지 않은 항목에는 secondary 유형 사용
        button_type = "primary" if st.session_state.analysis_selections.get(label) else "secondary"
        if st.button(
            label,
            key=f"btn_{label}",
            use_container_width=True,
            type=button_type
        ):
            # 수동으로 상태 전환
            st.session_state.analysis_selections[label] = not st.session_state.analysis_selections[label]
            # 피드백을 제공하고 다시 실행
            new_state = "선택됨" if st.session_state.analysis_selections[label] else "선택 해제됨"
            st.toast(f'{label}: {new_state}', icon='✅')
            st.rerun()


# --- 6. 탐색 ---
col1, col2 = st.columns(2)
with col1:
    if st.button("다른 파일 업로드", use_container_width=True):
        # 돌아가기 전에 세션 상태 지우기
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
            # 중복 제거 및 세션 상태에 저장
            st.session_state.selected_analysis_types = list(dict.fromkeys(final_analysis_types))
            st.switch_page("pages/03_처리중.py")
        else:
            st.warning("분석 항목을 하나 이상 선택해주세요.")