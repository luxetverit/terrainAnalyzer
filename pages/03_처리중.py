import streamlit as st
import time
import os
import geopandas as gpd
from utils.map_index_finder import find_overlapping_sheets
from utils.file_processor import process_uploaded_file
from utils.dem_processor import extract_dem_files, process_dem_data
from utils.theme_util import apply_theme_toggle

# 페이지 설정
st.set_page_config(
    page_title="또초자료 운사원 - 처리 중",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 테마 토글 적용
main_col = apply_theme_toggle()

# 커스텀 CSS 스타일
st.markdown("""
<style>
.big-text {
    font-size: 24px !important;
    font-weight: bold;
}
.center-text {
    text-align: center;
}
.processing-bar {
    height: 20px;
    background-color: #f0f0f0;
    border-radius: 10px;
    margin: 20px 0;
    overflow: hidden;
}
.processing-progress {
    height: 100%;
    background-color: #4CAF50;
    border-radius: 10px;
    width: 0%;
    transition: width 0.5s;
}
</style>
""", unsafe_allow_html=True)

# 업로드된 파일이 없는 경우 메인 페이지로 리다이렉트
if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file is None:
    st.error("업로드된 파일이 없습니다. 메인 페이지로 돌아가세요.")
    if st.button("메인 페이지로 돌아가기"):
        st.switch_page("app.py")
    st.stop()

# 분석 유형이 선택되지 않은 경우 이전 페이지로 리다이렉트
if 'selected_analysis_types' not in st.session_state:
    st.error("분석 유형이 선택되지 않았습니다. 이전 페이지로 돌아가세요.")
    if st.button("이전 페이지로 돌아가기"):
        st.switch_page("pages/01_기초분석.py")
    st.stop()

# 헤더
st.markdown("<h1 class='center-text'>운사원이 열심히 작업중이에요... 잠시만 기다려주세요!</h1>", unsafe_allow_html=True)

# 처리 중 화면 표시
progress_bar = st.progress(0)
status_text = st.empty()

# 분석 처리 진행
if 'processing_done' not in st.session_state:
    
    # 1단계: 파일 분석 (0-20%)
    status_text.markdown("### 파일 분석 중...")
    progress_bar.progress(10)
    
    try:
        # 업로드된 파일 정보 가져오기
        file_path = st.session_state.temp_file_path
        epsg_code = st.session_state.epsg_code
        
        # 파일 처리 (폴리곤 추출)
        progress_bar.progress(20)
        status_text.markdown("### 지리정보 추출 중...")
        gdf = process_uploaded_file(file_path, epsg_code)
        
        # 2단계: 도엽 인덱스 분석 (20-50%)
        progress_bar.progress(30)
        status_text.markdown("### 도엽 인덱스 분석 중...")
        
        # 업로드된 파일과 도엽 인덱스를 비교하여 겹치는 도엽 찾기
        map_index_results = find_overlapping_sheets(gdf, epsg_code)
        
        # 세션에 결과 저장
        st.session_state.map_index_results = map_index_results
        
        # 도엽 결과 표시
        if map_index_results['matched_sheets']:
            matched_count = len(map_index_results['matched_sheets'])
            status_text.markdown(f"### 도엽 인덱스 분석 완료! {matched_count}개 도엽 발견")
            
            # 인덱스 미리보기 이미지가 있으면 표시
            if map_index_results['preview_image']:
                st.image(map_index_results['preview_image'], caption="도엽 인덱스 일치 결과", use_column_width=True)
                
            # 일치하는 도엽 번호 표시
            sheet_col1, sheet_col2 = st.columns([1, 3])
            with sheet_col1:
                st.write("### 도엽 목록")
            with sheet_col2:
                # 최대 20개까지만 표시
                display_sheets = map_index_results['matched_sheets'][:20]
                if len(map_index_results['matched_sheets']) > 20:
                    display_sheets.append(f"...외 {len(map_index_results['matched_sheets']) - 20}개")
                st.write(", ".join(display_sheets))
            
            # map_index_results를 세션 상태에 저장하여 01기초분석.py에서도 사용할 수 있게 함
            st.session_state.map_index_results = map_index_results
        else:
            status_text.markdown("### 일치하는 도엽이 없습니다. 좌표계를 확인해주세요.")
        
        progress_bar.progress(50)
        
        # 3단계: 표고 ZIP 파일 확인 및 처리 (50-90%)
        progress_bar.progress(50)
        status_text.markdown("### 표고 데이터 파일 확인 중...")
        
        # DEM 데이터 처리를 위한 변수 초기화
        dem_results = None
        
        # 표고 자료 ZIP 파일이 있는지 확인
        dem_zip_paths = [
            "dem_data.zip",
            "attached_assets/dem_data.zip",
            "sample_data/dem_data.zip"
        ]
        
        dem_zip_path = None
        for path in dem_zip_paths:
            if os.path.exists(path):
                dem_zip_path = path
                break
        
        if dem_zip_path and map_index_results['matched_sheets']:
            try:
                status_text.markdown("### 도엽에 해당하는 표고 데이터 추출 중...")
                progress_bar.progress(60)
                
                # 도엽 번호와 일치하는 표고 데이터 추출
                extraction_result = extract_dem_files(dem_zip_path, map_index_results['matched_sheets'])
                extracted_files = extraction_result['extracted_files']
                temp_dir = extraction_result['temp_dir']
                
                if extracted_files:
                    # 추출된 파일 정보 표시
                    st.success(f"{len(extracted_files)}개의 표고 데이터 파일을 찾았습니다.")
                    
                    # 파일 목록 표시
                    with st.expander("추출된 표고 데이터 파일"):
                        for file_path in extracted_files:
                            st.write(os.path.basename(file_path))
                    
                    status_text.markdown("### 표고 데이터 처리 중...")
                    progress_bar.progress(70)
                    
                    # 표고 및 경사 분석 색상 팔레트 가져오기
                    elevation_palette = st.session_state.get('elevation_palette', 'terrain')
                    slope_palette = st.session_state.get('slope_palette', 'spectral')
                    
                    # 표고 데이터 처리
                    dem_results = process_dem_data(
                        extracted_files, 
                        gdf, 
                        epsg_code, 
                        elevation_palette, 
                        slope_palette
                    )
                    
                    status_text.markdown("### 경사 데이터 계산 중...")
                    progress_bar.progress(80)
                    
                    # 결과 이미지 표시
                    if dem_results['elevation']['image_path'] and dem_results['slope']['image_path']:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(dem_results['elevation']['image_path'], caption="표고 분석 결과", use_container_width=True)
                        with col2:
                            st.image(dem_results['slope']['image_path'], caption="경사 분석 결과", use_container_width=True)
                    
                    # 세션 상태에 결과 저장
                    st.session_state.dem_results = dem_results
                else:
                    st.warning("도엽과 일치하는 표고 데이터 파일을 찾지 못했습니다.")
                    
                    # 샘플 데이터로 대체
                    status_text.markdown("### 임시 표고 데이터로 처리 중...")
                    progress_bar.progress(75)
                    
                    # 임시 처리 로직 추가 필요
                
            except Exception as e:
                st.error(f"표고 데이터 처리 중 오류 발생: {e}")
        else:
            status_text.markdown("### 표고 데이터 파일을 찾지 못했습니다. 샘플 데이터로 처리합니다.")
            time.sleep(1)
            # 임시 처리 로직 추가 필요
        
        # 4단계: 결과 준비 (90-100%)
        for i in range(90, 101, 2):
            progress_bar.progress(i)
            status_text.markdown("### 최종 결과 준비 중...")
            time.sleep(0.1)
            
    except Exception as e:
        st.error(f"처리 중 오류가 발생했습니다: {e}")
        if st.button("다시 시도"):
            st.rerun()
        st.stop()
    
    # 분석 완료 표시 및 결과 저장
    st.session_state.processing_done = True
    
    # 색상 팔레트 정보 가져오기 (없으면 기본값 사용)
    elevation_palette = st.session_state.get('elevation_palette', 'spectral')
    slope_palette = st.session_state.get('slope_palette', 'terrain')
    
    # 분석 결과 정보 저장
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    # 선택된 분석 유형에 따라 결과 저장
    st.session_state.analysis_results = {
        "done": True,
        "message": "분석이 완료되었습니다!",
        "palettes": {
            "elevation": elevation_palette,
            "slope": slope_palette
        }
    }
    
    # 잠시 대기 후 결과 페이지로 이동
    status_text.markdown("### 분석이 완료되었습니다! 결과 페이지로 이동합니다...")
    time.sleep(1)
    st.switch_page("pages/04_결과.py")