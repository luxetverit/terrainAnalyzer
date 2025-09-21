import streamlit as st
import streamlit.components.v1 as components
import os
from utils.file_processor import process_uploaded_file
from utils.dem_analyzer import analyze_elevation
from utils.region_finder import get_region_info
from utils.simple_address_finder import get_location_name
from utils.theme_util import apply_styles
import utils.map_index_finder as map_index_finder

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

            # --- Diagnostic Information ---
            st.markdown("---")
            st.info("진단 정보")
            st.json({
                "입력 파일 좌표계 (EPSG)": epsg_code,
                "찾은 도엽 개수": len(st.session_state.matched_sheets),
                "찾은 도엽 번호 (최대 5개)": st.session_state.matched_sheets[:5]
            })
            st.markdown("---")

            # Get location info (You may want to secure this key)
            kakao_api_key = "bc9e52aa60d3c71a19742019b5ca3eaf"
            location_info = get_location_name(gdf, epsg_code, kakao_api_key)
            st.session_state.location_info = location_info

            st.session_state.initial_analysis_done = True

        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
            if st.button("다시 시도하기"):
                del st.session_state.initial_analysis_done
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
    st.markdown("##### 🗺️ 위치 개요도")
    map_url = f"https://map.kakao.com/link/map/분석지역,{lat},{lon}"
    components.html(f'<iframe src="{map_url}" width="100%" height="400" style="border:none;"></iframe>', height=410)

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

dem_items = {
    "elevation": "표고 분석",
    "slope": "경사 분석",
    "aspect": "경사향 분석",
}
db_items = {
    "landcover": "토지이용 현황",
    "soil": "토양도",
    "hsg": "수문학적 토양군",
}

# Combine for session state initialization
analysis_items = {**dem_items, **db_items}
if 'selected_analysis' not in st.session_state:
    st.session_state.selected_analysis = {key: False for key in analysis_items}

st.markdown("##### DEM 기반 분석 (지형 형태)")
cols1 = st.columns(3)
for i, (key, name) in enumerate(dem_items.items()):
    with cols1[i]:
        st.session_state.selected_analysis[key] = st.toggle(
            name, value=st.session_state.selected_analysis.get(key, False), key=key)

st.markdown("---")
st.markdown("##### 데이터베이스 중첩 분석 (영역 특성)")
cols2 = st.columns(3)
for i, (key, name) in enumerate(db_items.items()):
    with cols2[i]:
        st.session_state.selected_analysis[key] = st.toggle(
            name, value=st.session_state.selected_analysis.get(key, False), key=key)

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
        selected_count = sum(st.session_state.selected_analysis.values())
        if selected_count > 0:
            st.session_state.selected_analysis_types = [
                k for k, v in st.session_state.selected_analysis.items() if v]
            st.switch_page("pages/02_분석옵션.py")
        else:
            st.warning("하나 이상의 분석 항목을 선택해주세요.")