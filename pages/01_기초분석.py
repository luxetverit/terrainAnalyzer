import streamlit as st
import os
from utils.file_processor import process_uploaded_file
from utils.dem_analyzer import analyze_elevation
from utils.region_finder import get_region_info
from utils.simple_address_finder import get_location_name
from utils.theme_util import apply_theme_toggle

# 페이지 설정
st.set_page_config(page_title="또초자료 운사원 - 분석 결과",
                   page_icon="🗺️",
                   layout="wide",
                   initial_sidebar_state="collapsed")

# 테마 토글 적용
main_col = apply_theme_toggle()

# 커스텀 CSS 스타일
st.markdown("""
<style>
/* 큰 글씨 */
.big-text {
    font-size: 24px !important;
    font-weight: bold;
}

/* 기본 버튼 스타일 */
.stButton > button {
    width: 100%;
    min-width: 220px;
    max-width: 100%;
    min-height: 80px;
    padding: 16px 8px !important;
    border-radius: 18px !important;
    border: 2.5px solid #1976D2 !important;
    background: linear-gradient(90deg, #E3F2FD 0%, #FFFFFF 100%) !important;
    color: #154075 !important;
    font-weight: 700 !important;
    font-size: 26px !important;
    line-height: 1.32 !important;
    white-space: pre-line !important;  /* 줄바꿈 허용 */
    margin-top: 12px;
    margin-bottom: 12px;
    box-shadow: 0 4px 16px rgba(25, 118, 210, 0.12);
    transition: 0.2s;
    cursor: pointer;
    text-align: center;
}
            
stButton > button:hover {
    background: #BBDEFB !important;
    border-color: #0D47A1 !important;
    color: #0D47A1 !important;
}
            
.stButton > button:active {
    background: #90CAF9 !important;
    color: #1976D2 !important;
}
            
/* 선택된 버튼 스타일 */
.selected-button, .analysis-box.selected-box {
    border: 3px solid #1E88E5 !important;
    background-color: #E3F2FD !important;
    box-shadow: 0 0 10px rgba(30, 136, 229, 0.16) !important;
}

/* 커스텀 분석 박스 */
.analysis-box {
    border: 2px solid #D1D5DB;
    border-radius: 15px;
    padding: 32px 5px 20px 5px;
    margin: 14px 0;
    text-align: center;
    cursor: pointer;
    background-color: #fff;
    font-size: 20px;
    font-weight: 500;
    transition: all 0.15s;
    min-height: 110px;
    user-select: none;
}
.analysis-box.selected-box {
    border-color: #1976D2;
    background-color: #E3F2FD;
    color: #0D47A1;
    box-shadow: 0 0 14px rgba(25, 118, 210, 0.14);
}
.analysis-box:hover {
    background-color: #F2F7FB;
}

/* analysis-box 내부 글씨 스타일 */
.analysis-title {
    font-size: 1.22em;
    font-weight: bold;
    margin-bottom: 4px;
}
.analysis-subtitle {
    font-size: 1em;
    color: #446;
}
</style>
""", unsafe_allow_html=True)


# 업로드된 파일이 없는 경우 메인 페이지로 리다이렉트
if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file is None:
    st.error("업로드된 파일이 없습니다. 메인 페이지로 돌아가세요.")
    if st.button("메인 페이지로 돌아가기"):
        st.switch_page("app.py")
    st.stop()

# 헤더
st.markdown("## 올려주신 자료를 검토해본 결과...")

# 여기서 실제로 파일을 처리하고 분석하는 대신 결과를 표시
if 'analysis_done' not in st.session_state:
    # 실제 파일 처리 (파일 유효성 검사는 이미 첫 페이지에서 완료)
    with st.spinner("검토중입니다..."):
        try:
            # 이전 페이지에서 저장한 정보 가져오기
            temp_file_path = st.session_state.temp_file_path
            epsg_code = st.session_state.epsg_code

            # 폴리곤 추출
            gdf = process_uploaded_file(temp_file_path, epsg_code)
            
            # 맵 인덱스 찾기 (도엽 번호 검색)
            import utils.map_index_finder as map_index_finder
            map_index_results = map_index_finder.find_overlapping_sheets(gdf, epsg_code)
            st.session_state.map_index_results = map_index_results
            
            # 디버깅: 찾은 도엽 정보 출력
            st.write(f"검색된 도엽 번호: {map_index_results['matched_sheets']}")

            # 폴리곤이 없는 경우
            if gdf is None or gdf.empty:
                st.error("파일에서 유효한 폴리곤을 찾을 수 없습니다.")
                if st.button("업로드"):
                    st.switch_page("app.py")
                st.stop()

            # 카카오맵 API 키 직접 입력 - 제공해주신 API 키 입력
            kakao_api_key = ""  # 사용자가 제공한 API 키

            if kakao_api_key:
                # 카카오맵 API를 이용해 위치 정보 조회
                try:
                    location_info = get_location_name(gdf, epsg_code,
                                                      kakao_api_key)

                    # 지역 정보 저장 (기존 region_info와 통합)
                    region_info = get_region_info(gdf, epsg_code)
                    region_info["address"] = location_info["address"]
                    region_info["road_address"] = location_info["road_address"]
                    region_info["region"] = location_info["region"]
                    region_info["center_point_wgs84"] = (location_info["lon"],
                                                         location_info["lat"])

                    st.session_state.region_info = region_info
                except Exception as e:
                    # API 호출 실패 시 기본 지역 정보 사용
                    region_info = get_region_info(gdf, epsg_code)
                    st.session_state.region_info = region_info
            else:
                # API 키가 없으면 기본 지역 정보 사용
                region_info = get_region_info(gdf, epsg_code)
                st.session_state.region_info = region_info

            # 성공적으로 폴리곤을 찾은 경우, 다음 단계로 진행 표시
            st.session_state.gdf = gdf
            st.session_state.analysis_done = True
            st.rerun()

        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
            if st.button("다시 시도하기"):
                st.switch_page("app.py")
            st.stop()

# 파일 처리가 완료된 경우
if 'analysis_done' in st.session_state and st.session_state.analysis_done:
    st.success(f"### 파일 분석 완료")
    
    # 지역 정보 표시
    if 'region_info' in st.session_state:
        region_info = st.session_state.region_info
        center_x, center_y = region_info["center_point_original"]
        original_epsg = region_info["original_epsg"]

        # 주소 정보 표시
        st.markdown("#### 위치 정보")
        
        # 카카오맵 API 정보가 있으면 표시
        if "road_address" in region_info:
            road_address = region_info["road_address"]
            address = region_info["address"]
            region = region_info["region"]

            st.markdown(f"""
            * 설정한 좌표계: **{st.session_state.get('selected_epsg', '알 수 없음')}**
            * 지역: **{region}**
            * 도로명 주소: {road_address}
            * 지번 주소: {address}
            * 중심 좌표 (원본 좌표계): X={center_x:.2f}, Y={center_y:.2f}
            """)

            # WGS84 좌표가 있으면 간단한 지도 링크 제공
            if "center_point_wgs84" in region_info:
                lon, lat = region_info["center_point_wgs84"]
                kakao_map_url = f"https://map.kakao.com/link/map/현재위치,{lat},{lon}"
                st.markdown(f"[카카오맵에서 위치 확인하기]({kakao_map_url})")
        else:
            # 기본 지역 정보만 표시
            region_name = region_info["region_name"]
            st.markdown(f"""
            * 좌표계: **{st.session_state.get('selected_epsg', '알 수 없음')}**
            * 분석 지역: **{region_name}** (대한민국)
            * 중심 좌표 (원본 좌표계): X={center_x:.2f}, Y={center_y:.2f}
            """)
    
    # 도엽 인덱스 정보는 session_state에서 가져오며, 하드코딩하지 않음
    matched_sheets = []
    
    # 세션에서 도엽 정보 가져오기 (테스트 데이터 대신 실제 데이터가 있으면 사용)
    if 'map_index_results' in st.session_state and st.session_state.map_index_results:
        map_index_results = st.session_state.map_index_results
        if 'matched_sheets' in map_index_results and map_index_results['matched_sheets']:
            # 실제 일치하는 도엽이 있는 경우에만 사용
            if len(map_index_results['matched_sheets']) > 0:
                matched_sheets = map_index_results['matched_sheets']
    
    # 도엽 번호 표시
    st.markdown("#### 사용하는 도엽 번호")
    
    # 도엽 번호 표시 조건 처리
    if matched_sheets:
        # 대표 도엽 번호와 나머지 개수 표시
        if len(matched_sheets) > 1:
            st.write(f"{matched_sheets[0]} 외 {len(matched_sheets)-1}개")
        else:
            st.write(matched_sheets[0])
        
        # 도엽 번호 상세 보기 (확장 가능한 섹션)
        with st.expander(f"도엽 번호 상세 보기 ({len(matched_sheets)}개)"):
            # 테마에 따른 스타일 적용
            dark_mode = st.session_state.get('dark_mode', False)
            box_bg_color = "#1E1E1E" if dark_mode else "#F0F2F6"
            text_color = "#FFFFFF" if dark_mode else "#000000"
            
            # 원래 겹치는 도엽과 인근 도엽을 구분하여 표시
            if 'map_index_results' in st.session_state and 'original_matched_sheets' in st.session_state.map_index_results:
                original_sheets = st.session_state.map_index_results.get('original_matched_sheets', [])
                nearby_sheets = [sheet for sheet in matched_sheets if sheet not in original_sheets]
                
                st.markdown("**업로드한 파일과 겹치는 도엽:**")
                # 테마에 맞는 색상으로 도엽 번호 표시
                st.markdown(f"""
                <div style="background-color:{box_bg_color}; padding:10px; border-radius:5px; color:{text_color}">
                {", ".join(original_sheets)}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**인근 도엽:**")
                st.markdown(f"""
                <div style="background-color:{box_bg_color}; padding:10px; border-radius:5px; color:{text_color}">
                {", ".join(nearby_sheets)}
                </div>
                """, unsafe_allow_html=True)
            else:
                # 이전 버전 호환성 위해 유지
                st.markdown(f"""
                <div style="background-color:{box_bg_color}; padding:10px; border-radius:5px; color:{text_color}">
                {", ".join(matched_sheets)}
                </div>
                """, unsafe_allow_html=True)
        
        # 도엽 위치 시각화 (상세보기 버튼으로 접근하도록 수정)
        if 'map_index_results' in st.session_state and 'preview_image' in st.session_state.map_index_results and st.session_state.map_index_results['preview_image']:
            with st.expander("도엽 위치 상세보기", expanded=False):
                preview_img = st.session_state.map_index_results['preview_image']
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    # 이미지 크기를 절반으로 줄임(가운데 컬럼에 배치)
                    st.image(preview_img, caption="Map Sheet Matching Result", use_container_width=True)
    else:
        st.write("도엽 정보 없음")
        
    # 외부 이미지 표시하지 않음 - 상세보기 메뉴에서만 볼 수 있도록 수정

    st.markdown("## 다음 단계로 진행해볼까요?")

    # 분석 옵션 제공
    st.markdown("### 어떤 자료가 필요하세요?")

    # 선택 가능한 항목 목록
    analysis_items = [
        ("elevation", "표고 분석", "Elevation Analysis"),
        ("slope", "경사 분석", "Slope Analysis"),
        ("aspect", "경사향 분석", "Aspect Analysis"),
        ("landcover", "토지이용", "Land Cover"),
        ("soil", "토양도", "Soil Map"),
        ("hsg", "수문학적 토양군", "HSG"),
        ("result", "결과자료", "Result Data")
    ]

    # 선택 상태를 session_state로 관리
    if "selected_boxes" not in st.session_state:
        st.session_state.selected_boxes = {
            key: False
            for key, _, _ in analysis_items
        }

    # 각 항목별 가격 정의 (결과자료 제외)
    item_prices = {
        "elevation": 500,  # 표고 분석: 500원
        "slope": 500,      # 경사 분석: 500원
        "aspect": 500,     # 경사향 분석: 500원
        "landcover": 1000, # 토지 이용: 1000원
        "soil": 1000,      # 토양도: 1000원
        "hsg": 1000,       # 수문학적 토양군: 1000원
        "result": 500      # 결과자료: 동적 계산 (초기값은 의미 없음)
    }
    
    # 선택된 항목에 따른 결과자료 가격 계산 함수 정의
    def calculate_result_price():
        # 결과자료를 제외한 다른 항목들의 총합 계산
        other_items_total = sum([
            item_prices[key] for key in st.session_state.selected_boxes
            if key != "result" and st.session_state.selected_boxes[key]
        ])
        # 결과자료 가격 = 다른 항목의 총합과 동일
        return other_items_total if other_items_total > 0 else 500  # 최소한 500원

    # 클릭 함수 정의 - 선택 상태 토글
    def toggle_box(key):
        st.session_state.selected_boxes[
            key] = not st.session_state.selected_boxes[key]

    # CSS 스타일 설정
    st.markdown("""
    <style>
    .analysis-box {
        border: 2px solid;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .analysis-box:hover {
        background-color: #f8f9fa;
    }
    .selected-box {
        border-color: #3B82F6;
        background-color: #EBF5FF;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }
    .unselected-box {
        border-color: #D1D5DB;
        background-color: white;
    }
    </style>
    """,
                unsafe_allow_html=True)

    # 첫 번째 행 (3개 항목)
    col1, col2, col3 = st.columns(3)

    # 첫 번째 행 버튼들
    with col1:
        key, title, subtitle = analysis_items[0]
        box_class = "selected-box" if st.session_state.selected_boxes[
            key] else "unselected-box"
        st.markdown(f"""
            <div class="analysis-box {box_class}" id="box-{key}">
                <strong>{title}</strong><br>
                <span style='font-size: 0.9em;'>{subtitle}</span>
            </div>
            """,
                    unsafe_allow_html=True)
        # 클릭용 버튼 (숨김)
        if st.button(f"{title}", key=f"btn_{key}"):
            toggle_box(key)
            st.rerun()

    with col2:
        key, title, subtitle = analysis_items[1]
        box_class = "selected-box" if st.session_state.selected_boxes[
            key] else "unselected-box"
        st.markdown(f"""
            <div class="analysis-box {box_class}" id="box-{key}">
                <strong>{title}</strong><br>
                <span style='font-size: 0.9em;'>{subtitle}</span>
            </div>
            """,
                    unsafe_allow_html=True)
        if st.button(
                f"{title}",
                key=f"btn_{key}",
        ):
            toggle_box(key)
            st.rerun()

    with col3:
        key, title, subtitle = analysis_items[2]
        box_class = "selected-box" if st.session_state.selected_boxes[
            key] else "unselected-box"
        st.markdown(f"""
            <div class="analysis-box {box_class}" id="box-{key}">
                <strong>{title}</strong><br>
                <span style='font-size: 0.9em;'>{subtitle}</span>
            </div>
            """,
                    unsafe_allow_html=True)
        if st.button(
                f"{title}",
                key=f"btn_{key}",
        ):
            toggle_box(key)
            st.rerun()

    # 두 번째 행 (3개 항목)
    col4, col5, col6 = st.columns(3)

    with col4:
        key, title, subtitle = analysis_items[3]
        box_class = "selected-box" if st.session_state.selected_boxes[
            key] else "unselected-box"
        st.markdown(f"""
            <div class="analysis-box {box_class}" id="box-{key}">
                <strong>{title}</strong><br>
                <span style='font-size: 0.9em;'>{subtitle}</span>
            </div>
            """,
                    unsafe_allow_html=True)
        if st.button(
                f"{title}",
                key=f"btn_{key}",
        ):
            toggle_box(key)
            st.rerun()

    with col5:
        key, title, subtitle = analysis_items[4]
        box_class = "selected-box" if st.session_state.selected_boxes[
            key] else "unselected-box"
        st.markdown(f"""
            <div class="analysis-box {box_class}" id="box-{key}">
                <strong>{title}</strong><br>
                <span style='font-size: 0.9em;'>{subtitle}</span>
            </div>
            """,
                    unsafe_allow_html=True)
        if st.button(
                f"{title}",
                key=f"btn_{key}",
        ):
            toggle_box(key)
            st.rerun()

    with col6:
        key, title, subtitle = analysis_items[5]
        box_class = "selected-box" if st.session_state.selected_boxes[
            key] else "unselected-box"
        st.markdown(f"""
            <div class="analysis-box {box_class}" id="box-{key}">
                <strong>{title}</strong><br>
                <span style='font-size: 0.9em;'>{subtitle}</span>
            </div>
            """,
                    unsafe_allow_html=True)
        if st.button(
                f"{title}",
                key=f"btn_{key}",
        ):
            toggle_box(key)
            st.rerun()

    # 선택된 항목 수와 총 금액 계산
    selected_count = sum(st.session_state.selected_boxes.values())
    
    # 결과자료의 가격은 선택된 다른 항목 합계의 2배로 동적 계산
    if st.session_state.selected_boxes["result"]:
        # 결과자료가 선택된 경우, 가격 동적 계산
        item_prices["result"] = calculate_result_price()
    
    # 총 가격 계산
    total_price = sum([
        item_prices[key] for key in st.session_state.selected_boxes
        if st.session_state.selected_boxes[key]
    ])

    # 추가 정보 - 숫자 회전 효과 추가
    st.markdown("""
    <style>
    @keyframes countAnimation {
        0% { opacity: 0.3; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes numberRoll {
        0% { content: "0"; }
        10% { content: "1"; }
        20% { content: "2"; }
        30% { content: "3"; }
        40% { content: "4"; }
        50% { content: "5"; }
        60% { content: "6"; }
        70% { content: "7"; }
        80% { content: "8"; }
        90% { content: "9"; }
        100% { content: ""; }
    }
    .animated-number {
        animation: countAnimation 0.3s ease-out;
        display: inline-block;
    }
    .rolling-digit {
        display: inline-block;
        animation: numberRoll 0.5s steps(10) 3;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 결과자료가 선택된 경우와 선택되지 않은 경우에 따라 텍스트 다르게 표시
    analysis_count = sum([1 for key, selected in st.session_state.selected_boxes.items() 
                         if selected and key != "result"])
    
    if st.session_state.selected_boxes["result"]:
        # 결과자료가 선택된 경우
        message_html = f"""
        <div style="font-size: 18px; margin: 20px 0; font-weight: 500;">
            <span class="animated-number"><b>{analysis_count}가지 분석</b></span> 및 
            <span class="animated-number"><b>결과자료 다운로드</b></span>를 진행하려면 총 
            <span class="animated-number"><b><span class="rolling-digit">{total_price}</span>원</b></span>이 필요해요
        </div>
        """
    else:
        # 결과자료가 선택되지 않은 경우
        message_html = f"""
        <div style="font-size: 18px; margin: 20px 0; font-weight: 500;">
            <span class="animated-number"><b>{analysis_count}가지 분석</b></span>을 진행하려면 총 
            <span class="animated-number"><b><span class="rolling-digit">{total_price}</span>원</b></span>이 필요해요
        </div>
        """
    
    st.markdown(message_html, unsafe_allow_html=True)

    st.markdown("지금은 Beta 기간이라 무료로 진행할 거에요(속닥속닥)")

    # 다음 단계로 진행 버튼
    col7, col8 = st.columns([1, 1])

    with col8:
        if st.button("선택한 항목으로 진행하기", use_container_width=True):
            if selected_count > 0:
                # 선택된 항목들을 리스트로 변환
                st.session_state.selected_analysis_types = [
                    key for key, selected in
                    st.session_state.selected_boxes.items() if selected
                ]
                st.switch_page("pages/02_분석옵션.py")
            else:
                st.error("최소 1개 이상의 항목을 선택해주세요.")

    with col7:
        if st.button("다른 파일 업로드", use_container_width=True):
            st.switch_page("app.py")
