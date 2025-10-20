import time

import streamlit as st

from utils.theme_util import apply_styles

# --- 1. 페이지 설정 및 스타일링 ---
st.set_page_config(
    page_title="분석 중 - 지형 분석 서비스",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_styles()

# --- 2. 세션 상태 확인 ---
if "selected_analysis_types" not in st.session_state:
    st.warning(
        "분석 항목이 선택되지 않았습니다. 이전 페이지로 돌아가 항목을 선택해주세요."
    )
    if st.button("이전 페이지로 돌아가기"):
        st.switch_page("pages/01_기초분석.py")
    st.stop()

# --- 3. 처리 시뮬레이션 ---
st.markdown(
    """
    <div class="page-header" style="margin-top: -1.5rem;">
        <h1>분석 실행 중</h1>
        <p>선택하신 항목에 대한 분석을 진행하고 있습니다. 잠시만 기다려주세요.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.container():
    # 실제 실행 분석을 위한 임시 페이지

    progress_text = "분석을 준비하고 있습니다..."
    progress_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.02)  # 작업 시뮬레이션
        if percent_complete < 20:
            progress_text = "분석 환경을 설정하고 있습니다..."
        elif percent_complete < 60:
            progress_text = "데이터를 불러오고 있습니다..."
        elif percent_complete < 90:
            progress_text = "분석 엔진을 초기화하는 중입니다..."
        else:
            progress_text = "거의 다 끝났습니다!"
        progress_bar.progress(percent_complete + 1, text=progress_text)

    st.success("분석 준비가 완료되었습니다. 분석 페이지로 이동합니다.")
    time.sleep(1)

# --- 4. 다음 페이지로 전환 ---
st.switch_page("pages/04_분석.py")
