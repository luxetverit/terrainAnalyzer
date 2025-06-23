import streamlit as st
import os
from utils.file_processor import validate_file

# 페이지 설정 (사이드바 비활성화)
st.set_page_config(page_title="또초자료 운사원",
                   page_icon="🗺️",
                   layout="wide",
                   initial_sidebar_state="collapsed")

# 전체 페이지에 대한 맞춤 스타일
st.markdown("""
<style>
/* 상단 툴바 제거 (다크 모드에서 검정색으로 표시되는 상단 바) */
header {
    visibility: hidden;
    height: 0px;
}

/* 상단 패딩 제거 */
.main .block-container {
    padding-top: 30px;
    padding-bottom: 30px;
}
</style>
""",
            unsafe_allow_html=True)

# 세션 상태에 테마 설정 초기화
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'  # 기본값: 라이트 모드


# 테마 변경 함수
def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'


# 헤더 영역에 테마 전환 버튼 추가 (우측 상단)
col1, col2 = st.columns([0.9, 0.1])
with col2:
    # 현재 테마 상태 확인
    current_theme = st.session_state.theme
    is_dark = current_theme == 'dark'

    # 토글 버튼 라벨 - 간단하게 유지 (한 줄로 표시되도록 짧게 유지)
    theme_label = "🌙" if is_dark else "☀️"

    # 전체 CSS 스타일 적용
    st.markdown("""
    <style>
    /* 고정 위치 설정 - 공통 */
    div.row-widget.stToggleButton {
        position: absolute;
        top: 10px;
        right: 20px;
        width: 50px !important;
    }
    
    /* 토글 버튼 전체 스타일 */
    div[data-testid="stToggleButton"] {
        background-color: #1E293B !important;
        border: 2px solid #4F8BF9 !important;
        border-radius: 20px;
        padding: 2px;
        width: 50px !important;
        max-width: 50px !important;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* 토글 버튼 라벨 스타일 */
    div[data-testid="stToggleButton"] > label {
        color: white !important;
        font-weight: bold !important;
        font-size: 14px !important;
        text-align: center !important;
        width: 100% !important;
    }
    </style>
    """,
                unsafe_allow_html=True)

    # 라이트/다크 모드별 추가 스타일 적용
    if is_dark:
        # 다크 모드일 때 위치 조정
        st.markdown("""
        <style>
        div.row-widget.stToggleButton {
            top: 12px;
        }
        </style>
        """,
                    unsafe_allow_html=True)
    else:
        # 라이트 모드일 때 텍스트 색상 조정
        st.markdown("""
        <style>
        div[data-testid="stToggleButton"] > label {
            color: white !important;
            text-shadow: 0px 0px 1px black !important;
        }
        </style>
        """,
                    unsafe_allow_html=True)

    # 토글 버튼
    if st.toggle(theme_label, value=is_dark, key="theme_toggle"):
        if not is_dark:  # 라이트에서 다크로 변경
            st.session_state.theme = 'dark'
            st.rerun()
    else:
        if is_dark:  # 다크에서 라이트로 변경
            st.session_state.theme = 'light'
            st.rerun()

# 현재 테마에 따른 CSS 적용
if st.session_state.theme == 'dark':
    theme_bg = "#0E1117"
    theme_text = "#FAFAFA"
    theme_card_bg = "#262730"

    # 다크 모드 CSS 적용
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* 상단 메뉴바 제거 */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    /* 업로드 박스 스타일 리셋 */
    .uploadedFile {
        all: unset !important;
        width: 100% !important;
        height: auto !important;
        min-height: 300px !important;
        background-color: #1E1E1E !important;
        border: 2px dashed #4F8BF9 !important;
        border-radius: 8px !important;
        padding: 80px 20px !important;
        box-sizing: border-box !important;
        display: block !important;
    }
    .uploadedFile p, .uploadedFile span, .uploadedFile div {
        color: #FAFAFA !important;
    }
    .css-1kyxreq {
        background-color: #262730 !important;
    }
    .stButton button {
        background-color: #4F8BF9 !important;
        color: white !important;
    }
    /* 셀렉트 박스 스타일 적용 */
    div[data-baseweb="select"] {
        background-color: #262730 !important;
    }
    div[data-baseweb="select"] * {
        color: #FAFAFA !important;
    }
    div[data-baseweb="select"] span {
        color: #FAFAFA !important;
    }
    /* 셀렉트 박스 드롭다운 스타일 */
    div[data-baseweb="popover"] {
        background-color: #262730 !important;
    }
    div[data-baseweb="popover"] * {
        color: #FAFAFA !important;
    }
    </style>
    """,
                unsafe_allow_html=True)
else:
    theme_bg = "#FFFFFF"
    theme_text = "#31333F"
    theme_card_bg = "#F0F2F6"

    # 라이트 모드 CSS 적용 (기본)
    st.markdown("""
    <style>
    /* 기본 앱 배경 및 텍스트 */
    .stApp {
        background-color: #FFFFFF;
        color: #31333F;
    }
    /* 상단 메뉴바 제거 */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    h1, h2, h3, h4, h5, h6, p, li {
        color: #31333F !important;
    }
    
    /* 파일 업로드 박스 - 라이트 모드 */
    [data-testid="stFileUploader"] {
    background-color: #F9F9F9 !important;
    border: 2px dashed #555 !important;
    border-radius: 8px !important;
    padding: 60px 20px !important;
    color: #333 !important;
}
[data-testid="stFileUploader"] section span {
    color: #333 !important;
}
[data-testid="stFileUploader"] section {
    background-color: #F9F9F9 !important;
}
[data-testid="stFileUploader"] div {
    background-color: #F9F9F9 !important;
}
[data-testid="stFileUploader"] p {
    color: #333 !important;
    background-color: #F9F9F9 !important;
}
[data-testid="stFileUploader"] label {
    color: #333 !important;
    background-color: #F9F9F9 !important;
}
.css-90vs21, .css-1q7744g, .css-8ojfln p, .css-1r6slb0 {
    color: #333 !important;
    background-color: #F9F9F9 !important;
    }
    
    /* 다음 버튼 - 라이트 모드 */
    .stButton button {
        background-color: white !important;
        color: black !important;
        border: 2px solid black !important;
    }
    
    /* EPSG 선택 박스 - 라이트 모드 (테두리 중복 해결) */
    .css-1qrvfrg, .css-1s2u09g {
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    
    /* 셀렉트박스 외부 컨테이너 */
    div.stSelectbox {
        border: none !important;
        background-color: white !important;
    }
    
    /* 실제 셀렉트박스 */
    div.stSelectbox > div > div {
        border: 2px solid black !important;
        border-radius: 4px !important;
        background-color: white !important;
    }
    
    /* 모든 selectbox 요소 라이트 모드로 통일 */
    div[data-baseweb="select"] {
        background-color: white !important;
        border: none !important;
    }
    div[data-baseweb="select"] * {
        color: black !important;
        background-color: white !important;
    }
    div[data-baseweb="select"] span {
        color: black !important;
        background-color: white !important;
    }
    /* 드롭다운 메뉴 스타일 */
    div[data-baseweb="popover"] {
        background-color: white !important;
    }
    div[data-baseweb="popover"] * {
        color: black !important;
        background-color: white !important;
    }
    div[data-baseweb="popover"] div {
        background-color: white !important;
    }
    div[role="listbox"] {
        background-color: white !important;
    }
    div[role="listbox"] * {
        background-color: white !important;
        color: black !important;
    }
    </style>
    """,
                unsafe_allow_html=True)

# 상단 텍스트
with col1:
    st.markdown("## 안녕하세요! 오늘도 반복되는 기초자료조사...")
    st.markdown("### 지금까지 많이 힘드셨죠?")
    st.markdown("### 또초자료 윤사원이 해결해드릴게요!")

# 기타 공통 스타일
st.markdown("""
<style>
/* 공통 기본 스타일 */
.stButton>button {
    width: 100%;
    font-size: 18px !important;
}
div.stMarkdown p {
    text-align: center;
}
.big-stats {
    font-size: 24px !important;
    font-weight: bold !important;
}
.css-90vs21, .css-1q7744g {
    text-align: center !important;
}
/* 커서 효과 제거 */
.stSelectbox div, .stSelectbox span, .stSelectbox svg, .stSelectbox *, 
div[role="listbox"], div[role="listbox"] *, 
div[data-baseweb="select"], div[data-baseweb="select"] *,
div[data-baseweb="popover"], div[data-baseweb="popover"] * {
    cursor: default !important;
}
</style>
""",
            unsafe_allow_html=True)

# 업로드 카운터 관리 - 매번 고유한 키 생성을 위해 사용
if 'upload_counter' not in st.session_state:
    st.session_state.upload_counter = 0

# 파일 업로드 위젯 (고유 키 사용)
upload_key = f"file_uploader_{st.session_state.upload_counter}"
uploaded_file = st.file_uploader(
    "Drag & Drop Box\n\n조사하고 싶은 공간의 SHP 파일(ZIP으로 압축)이나, DXF 파일을 업로드해주세요\n\n무료 기간에는 ip 당 1회의 업로드만 가능해요(size 00m 이하)",
    type=["dxf", "zip"],
    key=upload_key,
    label_visibility="collapsed")

# 하단 정보
st.markdown("---")
st.markdown(
    "표고자료에서는 EPSG 5186(GRS80) 를 기본 좌표계로 사용하고 있어요. 업로드 할때 좌표계를 꼭 확인해주세요!")
st.markdown("업로드한 자료의 좌표계를 선택해주세요(미설정시 기본 좌표계 사용)")

# 좌표계 선택 드롭다운
epsg_options = {
    "EPSG:5186 / Korea 2000 / Central Belt 2010": 5186,
    "EPSG:5183 / Korea 2000 / East Belt": 5183,
    "EPSG:5185 / Korea 2000 / West Belt 2010": 5185,
    "EPSG:5187 / Korea 2000 / East Belt 2010": 5187,
    "EPSG:5179 / Korea 2000 / Unified CS": 5179,
    "EPSG:5175 / Korea 1985 / Modified Central Belt Jeju": 5175,
    "EPSG:5174 / Korea 1985 / Central Belt": 5174,
    "EPSG:5178 / Korea 1985 / Modified Central Belt": 5178,
    "ESRI:102088 / Korean 1985 / Modified Korea East Belt": 102088,
    "ESRI:102086 / Korean 1985 / Modified Korea Central Belt": 102086,
    "ESRI:102081 / Korean 1985 / Modified Korea West Belt": 102081,
    "EPSG:2097 / Korean 1985 / Central Belt": 2097,
    "EPSG:4326 / WGS84": 4326
}
selected_epsg = st.selectbox("EPSG 선택",
                             options=list(epsg_options.keys()),
                             index=0)
epsg_code = epsg_options[selected_epsg]

# 파일이 업로드된 경우에만 처리
temp_file_path_for_next = None
if uploaded_file:
    with st.spinner("파일 확인 중..."):
        is_valid, message, temp_file_path = validate_file(uploaded_file)

    if is_valid:
        temp_file_path_for_next = temp_file_path
        st.success(f"'{uploaded_file.name}' 파일이 성공적으로 업로드되었습니다.")
        st.info("좌표계를 선택하고 아래의 '다음' 버튼을 클릭하여 계속 진행하세요.")
    else:
        st.error(f"파일 업로드 오류: {message}")

# 다음 버튼 - 파일 업로드 및 좌표계 선택 후 클릭해야 다음 페이지로 이동
next_button = st.button("다음", use_container_width=True)
if next_button:
    if uploaded_file and temp_file_path_for_next:
        # 세션 상태에 저장 (다음 페이지로 이동 직전에만)
        st.session_state.uploaded_file = uploaded_file
        st.session_state.temp_file_path = temp_file_path_for_next
        st.session_state.epsg_code = epsg_code
        st.session_state.epsg_options = epsg_options
        st.session_state.selected_epsg = selected_epsg

        # 다음 업로드를 위해 카운터 증가
        st.session_state.upload_counter += 1

        # 다음 페이지로 이동
        st.switch_page("pages/01_기초분석.py")
    else:
        st.error("파일을 업로드해주세요.")

# 상태 정보 영역
st.markdown("---")
st.markdown("<p class='big-stats'>오늘 총 0000 개의 기초자료 조사를 수행했어요</p>",
            unsafe_allow_html=True)
st.markdown("<p class='big-stats'>지금까지 총 0000 개의 기초자료 조사를 수행했어요</p>",
            unsafe_allow_html=True)
