import streamlit as st
import os
import base64
from utils.color_palettes import ALL_PALETTES, get_palette_preview_html
from utils.theme_util import apply_theme_toggle
from utils.landcover_visualizer import get_landcover_palette_preview_html


def get_image_as_base64(image_path):
    """이미지 파일을 base64로 인코딩하여 반환"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# 페이지 설정
st.set_page_config(page_title="또초자료 운사원 - 분석 옵션",
                   page_icon="🗺️",
                   layout="wide",
                   initial_sidebar_state="collapsed")

# 테마 토글 적용
main_col = apply_theme_toggle()

# 커스텀 CSS 스타일
st.markdown("""
<style>
.big-text {
    font-size: 24px !important;
    font-weight: bold;
}
.stButton>button {
    width: 100%;
    border-radius: 20px !important;
    font-size: 18px !important;
    padding: 10px 24px !important;
}
.color-option {
    border: 2px solid #dddddd;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    margin: 10px;
    cursor: pointer;
}
.color-option img {
    max-width: 100%;
    height: auto;
}
.palette-preview {
    width: 100%;
    margin: 8px 0;
    border-radius: 4px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.palette-label {
    margin-top: 4px;
    font-size: 14px;
    text-align: center;
}
.palette-selected {
    border: 3px solid #1E88E5;
    box-shadow: 0 0 8px rgba(30,136,229,0.5);
}
</style>
""",
            unsafe_allow_html=True)

# 업로드된 파일이 없는 경우 메인 페이지로 리다이렉트
if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file is None:
    st.error("업로드된 파일이 없습니다. 메인 페이지로 돌아가세요.")
    if st.button("메인 페이지로 돌아가기"):
        st.switch_page("app.py")
    st.stop()

# 분석 유형 확인
if 'selected_analysis_types' not in st.session_state:
    st.error("분석 유형이 선택되지 않았습니다. 이전 페이지로 돌아가세요.")
    if st.button("이전 페이지로 돌아가기"):
        st.switch_page("pages/01_기초분석.py")
    st.stop()

# 세션 상태 초기화
if 'elevation_palette' not in st.session_state:
    st.session_state.elevation_palette = 'spectral'  # 기본값: 스펙트럼

if 'slope_palette' not in st.session_state:
    st.session_state.slope_palette = 'terrain'  # 기본값: 지형

# 헤더
st.markdown("## 어떤 스타일(색상)로 만들어 드릴까요?")

# 선택된 분석 유형에 따라 색상 팔레트 선택 UI 표시
selected_types = st.session_state.selected_analysis_types

# 팔레트 키 목록
palette_keys = list(ALL_PALETTES.keys())

# 초기화
if 'landcover_palette' not in st.session_state:
    st.session_state.landcover_palette = 'landcover'  # 기본값
if 'aspect_palette' not in st.session_state:
    st.session_state.aspect_palette = 'rainbow'  # 기본값
if 'soil_palette' not in st.session_state:
    st.session_state.soil_palette = 'tab10'  # 기본값
if 'hsg_palette' not in st.session_state:
    st.session_state.hsg_palette = 'Set3'  # 기본값

# 색상 팔레트 선택을 위한 레이아웃
left_col, right_col = st.columns(2)

# 표고 분석 UI (왼쪽 컬럼)
with left_col:
    st.markdown("<h3 style='text-align: center;'>표고 분석 색상</h3>",
                unsafe_allow_html=True)

    # 현재 선택된 팔레트 인덱스
    current_index = palette_keys.index(st.session_state.elevation_palette)

    # 선택기 행 (이전/다음 버튼과 현재 선택 표시)
    cols = st.columns([1, 3, 1])

    # 이전 버튼
    if cols[0].button("◀", key="prev_elevation"):
        if 'elevation' in selected_types:  # 선택된 경우에만 실제로 팔레트 변경
            new_index = (current_index - 1) % len(palette_keys)
            st.session_state.elevation_palette = palette_keys[new_index]
            st.rerun()

    # 현재 선택된 팔레트 이름
    cols[1].markdown(
        f"<div style='text-align: center; font-weight: bold;'>{ALL_PALETTES[st.session_state.elevation_palette]['name']}</div>",
        unsafe_allow_html=True)

    # 다음 버튼
    if cols[2].button("▶", key="next_elevation"):
        if 'elevation' in selected_types:  # 선택된 경우에만 실제로 팔레트 변경
            new_index = (current_index + 1) % len(palette_keys)
            st.session_state.elevation_palette = palette_keys[new_index]
            st.rerun()

    # 축소된 팔레트 미리보기 (컬러바 시각화) - 100% 폭
    st.markdown(
        f"<div style='width: 100%; margin: 10px auto;'>{get_palette_preview_html(st.session_state.elevation_palette)}</div>",
        unsafe_allow_html=True)

    # 선택된 팔레트에 맞는 샘플 이미지 표시
    sample_image_path = f"assets/palette_samples/elevation_{st.session_state.elevation_palette}.png"

    # 이미지와 선택 상태를 함께 표시
    st.markdown(f"""
    <style>
    .image-container {{
        position: relative;
        width: 100%;
        display: flex;
        justify-content: center;
    }}
    .sample-image {{
        width: auto; 
        height: auto;
        max-width: 100%;
    }}
    .overlay-text {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: rgba(255, 255, 255, 0.8);
        color: black;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 18px;
        z-index: 10;
    }}
    </style>
    <div class="image-container">
        <img src="data:image/png;base64,{get_image_as_base64(sample_image_path)}" style="width: 100%;">
        {f'<div class="overlay-text">선택하지 않았어요</div>' if 'elevation' not in selected_types else ''}
    </div>
    <p style="text-align: center; margin-top: 5px; color: gray;">표고 분석 예시</p>
    """,
                unsafe_allow_html=True)
                
    # 토지피복도 분석 UI - 표고 분석 바로 아래 추가
    st.markdown("<h3 style='text-align: center;'>토지피복도 분석</h3>",
                unsafe_allow_html=True)
    
    # 토지피복도 팔레트 미리보기
    st.markdown(
        f"<div style='width: 100%; margin: 10px auto;'>{get_landcover_palette_preview_html()}</div>",
        unsafe_allow_html=True)
    
    # 토지피복도 샘플 이미지 표시
    landcover_sample_path = f"assets/palette_samples/landcover_landcover.png"
    
    # 이미지와 선택 상태를 함께 표시
    st.markdown(f"""
    <div class="image-container">
        <img src="data:image/png;base64,{get_image_as_base64(landcover_sample_path)}" style="width: 100%;">
        {f'<div class="overlay-text">선택하지 않았어요</div>' if 'landcover' not in selected_types else ''}
    </div>
    <p style="text-align: center; margin-top: 5px; color: gray;">토지피복도 분석 예시</p>
    """,
                unsafe_allow_html=True)


# 경사 분석 UI (오른쪽 컬럼)
with right_col:
    st.markdown("<h3 style='text-align: center;'>경사 분석 색상</h3>",
                unsafe_allow_html=True)

    # 현재 선택된 팔레트 인덱스
    current_index = palette_keys.index(st.session_state.slope_palette)

    # 선택기 행 (이전/다음 버튼과 현재 선택 표시)
    cols = st.columns([1, 3, 1])

    # 이전 버튼
    if cols[0].button("◀", key="prev_slope"):
        if 'slope' in selected_types:  # 선택된 경우에만 실제로 팔레트 변경
            new_index = (current_index - 1) % len(palette_keys)
            st.session_state.slope_palette = palette_keys[new_index]
            st.rerun()

    # 현재 선택된 팔레트 이름
    cols[1].markdown(
        f"<div style='text-align: center; font-weight: bold;'>{ALL_PALETTES[st.session_state.slope_palette]['name']}</div>",
        unsafe_allow_html=True)

    # 다음 버튼
    if cols[2].button("▶", key="next_slope"):
        if 'slope' in selected_types:  # 선택된 경우에만 실제로 팔레트 변경
            new_index = (current_index + 1) % len(palette_keys)
            st.session_state.slope_palette = palette_keys[new_index]
            st.rerun()

    # 축소된 팔레트 미리보기 (컬러바 시각화) - 100% 폭
    st.markdown(
        f"<div style='width: 100%; margin: 10px auto;'>{get_palette_preview_html(st.session_state.slope_palette)}</div>",
        unsafe_allow_html=True)

    # 선택된 팔레트에 맞는 샘플 이미지 표시
    sample_image_path = f"assets/palette_samples/slope_{st.session_state.slope_palette}.png"

    # 이미지와 선택 상태를 함께 표시
    st.markdown(f"""
    <div class="image-container">
        <img src="data:image/png;base64,{get_image_as_base64(sample_image_path)}" style="width: 100%;">
        {f'<div class="overlay-text">선택하지 않았어요</div>' if 'slope' not in selected_types else ''}
    </div>
    <p style="text-align: center; margin-top: 5px; color: gray;">경사 분석 예시</p>
    """,
                unsafe_allow_html=True)

# 3x2 그리드로 모든 분석 샘플 표시
st.markdown("## 분석 결과 샘플")

# 공통 스타일 정의
st.markdown("""
<style>
.sample-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    margin-bottom: 15px;
}
.sample-item {
    position: relative;
    text-align: center;
}
.sample-item img {
    max-width: 220px;
    height: auto;
    margin: 0 auto;
    display: block;
    border-radius: 7px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.overlay-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background-color: rgba(255, 255, 255, 0.8);
    color: black;
    padding: 10px 20px;
    border-radius: 5px;
    font-weight: bold;
    font-size: 18px;
    z-index: 10;
}
.sample-title {
    margin-top: 5px;
    color: gray;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# 첫 번째 행: 표고 분석, 경사 분석
elevation_sample = f"assets/palette_samples/elevation_{st.session_state.elevation_palette}.png"
if not os.path.exists(elevation_sample):
    elevation_sample = "assets/palette_samples/elevation_viridis.png"
    
slope_sample = f"assets/palette_samples/slope_{st.session_state.slope_palette}.png"
if not os.path.exists(slope_sample):
    slope_sample = "assets/palette_samples/slope_viridis.png"

# 경사향 분석 이미지와 토지피복도 이미지
aspect_sample = "assets/palette_samples/aspect.png"
landcover_sample = "assets/palette_samples/landcover_landcover.png"

# 토양도와 수문학적 토양군 이미지
soil_sample = "assets/palette_samples/soil_map.png"
hsg_sample = "assets/palette_samples/hydrologic_soil.png"

# 3x2 그리드 HTML 생성
st.markdown(f"""
<div class="sample-grid">
    <!-- 첫 번째 행: 표고, 경사 -->
    <div class="sample-item">
        <img src="data:image/png;base64,{get_image_as_base64(elevation_sample)}">
        {f'<div class="overlay-text">선택하지 않았어요</div>' if 'elevation' not in selected_types else ''}
        <div class="sample-title">표고 분석</div>
    </div>
    <!--
    <div class="sample-item">
        <img src="data:image/png;base64,{get_image_as_base64(slope_sample)}">
        {f'<div class="overlay-text">선택하지 않았어요</div>' if 'slope' not in selected_types else ''}
        <div class="sample-title">경사 분석</div>
    </div>
    -->
    
    <!-- 두 번째 행: 경사향, 토지피복도 -->
    <div class="sample-item">
        <img src="data:image/png;base64,{get_image_as_base64(aspect_sample)}">
        {f'<div class="overlay-text">선택하지 않았어요</div>' if 'aspect' not in selected_types else ''}
        <div class="sample-title">경사향 분석</div>
    </div>
    <div class="sample-item">
        <img src="data:image/png;base64,{get_image_as_base64(landcover_sample)}">
        {f'<div class="overlay-text">선택하지 않았어요</div>' if 'landcover' not in selected_types else ''}
        <div class="sample-title">토지피복도 분석</div>
    </div>
    
    <!-- 세 번째 행: 토양도, 수문학적 토양군 -->
    <div class="sample-item">
        <img src="data:image/png;base64,{get_image_as_base64(soil_sample)}">
        {f'<div class="overlay-text">선택하지 않았어요</div>' if 'soil' not in selected_types else ''}
        <div class="sample-title">토양도 분석</div>
    </div>
    <div class="sample-item">
        <img src="data:image/png;base64,{get_image_as_base64(hsg_sample)}">
        {f'<div class="overlay-text">선택하지 않았어요</div>' if 'hsg' not in selected_types else ''}
        <div class="sample-title">수문학적 토양군 분석</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 버튼 영역
col1, col2 = st.columns(2)

with col1:
    if st.button("다른자료\n선택하기", use_container_width=True):
        st.switch_page("pages/01_기초분석.py")

with col2:
    if st.button("이 스타일로\n진행할게요", use_container_width=True):
        st.session_state.style_option = "default"  # 기본 스타일 선택
        st.switch_page("pages/03_처리중.py")
