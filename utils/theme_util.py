"""
테마 유틸리티 - 라이트/다크 모드 전환 기능을 모든 페이지에서 사용할 수 있도록 함
"""
import streamlit as st

def apply_theme_toggle():
    """
    페이지에 테마 토글 버튼을 추가하고 현재 테마에 맞는 스타일을 적용합니다.
    
    모든 페이지에서 이 함수를 호출하여 동일한 테마 전환 기능을 사용할 수 있습니다.
    """
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
        
        # 토글 버튼 라벨 설정
        theme_label = "🌙" if is_dark else "☀️"
        
        # 버튼 스타일 공통 부분 정의
        button_style = """
        <style>
        /* 테마 버튼 위치 및 정렬 */
        div.row-widget.stButton {
            position: fixed;
            top: 10px;
            right: 20px;
            z-index: 10000;
        }
        
        /* 버튼 기본 스타일 */
        div.stButton > button:first-child {
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            background-color: #FF4B4B !important;
            color: white !important;
            border: none !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 20px !important;
            padding: 0px !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3) !important;
        }
        
        /* 호버 효과 */
        div.stButton > button:hover {
            background-color: #FF6B6B !important;
            transform: scale(1.05) !important;
            transition: all 0.2s ease !important;
        }
        </style>
        """
        
        # 스타일 적용
        st.markdown(button_style, unsafe_allow_html=True)
        
        # 버튼 생성
        if st.button(theme_label, key="theme_toggle", on_click=toggle_theme):
            pass
    
    # 현재 테마에 따른 스타일 적용
    if st.session_state.theme == 'dark':
        # 다크 모드 CSS 적용
        st.markdown("""
        <style>
        /* 기본 앱 배경 및 텍스트 */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* 상단 메뉴바 제거 */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        /* 메인 텍스트 스타일 */
        h1, h2, h3, h4, h5, h6, p, li {
            color: #FAFAFA !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        # 라이트 모드 CSS 적용
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
        /* 메인 텍스트 스타일 */
        h1, h2, h3, h4, h5, h6, p, li {
            color: #31333F !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # col1 반환하여 상단 제목 등에 사용할 수 있도록 함
    return col1