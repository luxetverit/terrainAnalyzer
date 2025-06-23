import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.visualization import create_elevation_heatmap
from utils.color_palettes import ALL_PALETTES, get_palette_preview_html
from utils.theme_util import apply_theme_toggle

# 페이지 설정
st.set_page_config(
    page_title="또초자료 운사원 - 결과",
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
.result-text {
    font-size: 18px !important;
    font-weight: bold;
}
.stButton>button {
    width: 100%;
    border-radius: 20px !important;
    font-size: 18px !important;
    padding: 10px 24px !important;
}
</style>
""", unsafe_allow_html=True)

# 업로드된 파일이 없는 경우 메인 페이지로 리다이렉트
if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file is None:
    st.error("업로드된 파일이 없습니다. 메인 페이지로 돌아가세요.")
    if st.button("메인 페이지로 돌아가기"):
        st.switch_page("app.py")
    st.stop()

# 필요한 정보가 없으면 처리 중 페이지로 리다이렉트
if 'processing_done' not in st.session_state:
    st.error("처리가 완료되지 않았습니다. 처리 중 페이지로 돌아가세요.")
    if st.button("이전 페이지로 돌아가기"):
        st.switch_page("pages/03_처리중.py")
    st.stop()

# 분석 결과 정보 가져오기
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {
        "done": True,
        "message": "분석이 완료되었습니다!",
        "palettes": {
            "elevation": "spectral",  # 기본값
            "slope": "terrain"        # 기본값
        }
    }

# 헤더
st.markdown("## 작업이 완료되었어요")

# 분석 결과 표시
selected_types = st.session_state.get('selected_analysis_types', [])

# 선택된 색상 팔레트 정보 가져오기
elevation_palette = st.session_state.analysis_results["palettes"]["elevation"]
slope_palette = st.session_state.analysis_results["palettes"]["slope"]

# 결과 요약 (실제 분석 결과 표시)
if 'dem_results' in st.session_state and st.session_state.dem_results:
    dem_results = st.session_state.dem_results
    
    # 표고 통계 표시
    if 'elevation' in selected_types and dem_results['elevation']['stats']:
        elev_stats = dem_results['elevation']['stats']
        st.markdown(f"## 표고는 {elev_stats['min']:.1f}~{elev_stats['max']:.1f}m, 평균표고는 {elev_stats['mean']:.1f}m로 분석되었어요.")
    
    # 경사 통계 표시
    if 'slope' in selected_types and dem_results['slope']['stats']:
        slope_stats = dem_results['slope']['stats']
        st.markdown(f"## 경사는 {slope_stats['min']:.1f}~{slope_stats['max']:.1f}도, 평균경사 {slope_stats['mean']:.1f}도로 분석되었어요.")
        
        # 경사 등급별 면적 정보가 있으면 표시
        if 'area_by_class' in slope_stats:
            # 가장 높은 비율을 가진 등급 찾기
            max_area_class = max(slope_stats['area_by_class'].items(), key=lambda x: x[1])
            max_area_pct = max_area_class[1] * 100
            st.markdown(f"## 경사는 {max_area_class[0]} 지역이 {max_area_pct:.1f}%로 대부분을 차지하네요.")
    
    # 지역 정보와 같은 추가 결과가 있는 경우 표시할 수 있음
    if 'map_index_results' in st.session_state and st.session_state.map_index_results:
        matched_sheets = st.session_state.map_index_results.get('matched_sheets', [])
        if matched_sheets:
            st.markdown(f"## 총 {len(matched_sheets)}개 도엽을 기준으로 분석되었어요.")
else:
    # 분석 결과가 없는 경우
    st.markdown("## 표고는 00~00m, 평균표고는 00m로 분석되었어요.")
    st.markdown("## 경사는 0~0도, 평균경사 0도로 분석되었어요.")
    
# 선택하지 않은 분석 유형에 대한 메시지
other_types = [t for t in ['landcover', 'soil', 'hsg'] if t in selected_types]
if other_types:
    st.markdown("## 선택하신 다른 분석 결과도 아래에서 확인하실 수 있어요.")

# 결과 시각화 영역
st.markdown("### 결과사진 대표")

# 선택한 색상 팔레트 표시
st.markdown("#### 선택하신 색상 팔레트")

# 색상 팔레트 미리보기 표시
col1, col2 = st.columns(2)
with col1:
    if 'elevation' in selected_types:
        st.markdown("##### 표고 분석 색상")
        st.markdown(get_palette_preview_html(elevation_palette), unsafe_allow_html=True)
        st.markdown(f"<div class='palette-label'>{ALL_PALETTES[elevation_palette]['name']}</div>", unsafe_allow_html=True)

with col2:
    if 'slope' in selected_types:
        st.markdown("##### 경사 분석 색상")
        st.markdown(get_palette_preview_html(slope_palette), unsafe_allow_html=True)
        st.markdown(f"<div class='palette-label'>{ALL_PALETTES[slope_palette]['name']}</div>", unsafe_allow_html=True)

# 탭으로 각 분석 결과 보여주기
if selected_types:
    tabs = st.tabs([ALL_PALETTES[elevation_palette]["name"] if type_key == "elevation" else 
                     ALL_PALETTES[slope_palette]["name"] if type_key == "slope" else 
                     type_key for type_key in selected_types])
    
    for i, type_key in enumerate(selected_types):
        with tabs[i]:
            # 랜덤 데이터로 예시 이미지 생성
            np.random.seed(42 + i)  # 각 탭마다 다른 시드
            data_array = np.random.normal(100, 25, (100, 100))  # 정규분포 데이터
            data_array = np.clip(data_array, 0, 200)  # 값 범위 제한
            
            # 분석 유형에 따라 다른 시각화
            if type_key == 'elevation':
                # 통계 데이터 (샘플)
                stats = {
                    'min': data_array.min(),
                    'max': data_array.max(),
                    'mean': data_array.mean(),
                    'area': '10.5'
                }
                # 경계 (샘플)
                bounds = (0, 0, 100, 100)
                # 표고 시각화 생성
                fig = create_elevation_heatmap(data_array, bounds, stats, elevation_palette)
                st.pyplot(fig)
                
            elif type_key == 'slope':
                # 경사 시각화 (경사 팔레트 사용)
                fig, ax = plt.subplots(figsize=(10, 8))
                # 색상 팔레트 가져오기
                colors = ALL_PALETTES[slope_palette]['colors']
                # 경사 시각화
                im = ax.imshow(data_array, cmap=plt.cm.colors.ListedColormap(colors), origin='lower')
                plt.colorbar(im, ax=ax, label='경사 (도)')
                ax.set_title('경사 분석 결과', fontweight='bold', fontsize=14)
                ax.set_xlabel('X 좌표')
                ax.set_ylabel('Y 좌표')
                st.pyplot(fig)
                
            else:
                # 기타 분석 유형 (기본 시각화)
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(data_array, cmap='viridis', origin='lower')
                plt.colorbar(im, ax=ax, label=f'{type_key} 데이터')
                ax.set_title(f'{type_key} 분석 결과', fontweight='bold', fontsize=14)
                ax.set_xlabel('X 좌표')
                ax.set_ylabel('Y 좌표')
                st.pyplot(fig)
else:
    # 선택된 분석 유형이 없는 경우
    st.warning("선택된 분석 유형이 없습니다.")

# 다운로드 버튼
col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="결과 다운로드",
        data="샘플 결과 데이터",
        file_name="result.txt",
        mime="text/plain",
        key="download_result"
    )

# 마무리 텍스트 및 버튼
st.markdown("---")
st.markdown("# 다운로드 완료!")
st.markdown("## 또초자료 운사원을 찾아주셔서 감사해요!")
st.markdown("## 다음에 또 기초자료 조사가 필요하시면 언제든지 방문해주세요!")

# 자료 출처 정보
st.markdown("---")
st.markdown("""
### 자료 출처:
- DEM - 국토지리정보원
- 토지피복도 - 환경부
- 정밀토양도 - 농촌진흥청
""")

# 문의 정보
st.markdown("""
또초자료를 이용하시면서 불편한 사항이 발생하거나
개선 또는 다시 해주었으면 하는 자료가 있으면 (메일주소) 로 문의주세요!
제가 운영진님께 잘 전달해드릴게요
""")

# 푸터
st.markdown("---")
st.markdown("Published by Edward Yoon", unsafe_allow_html=True)