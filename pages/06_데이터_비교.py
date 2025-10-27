
import streamlit as st
import os
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine
import warnings
from utils.config import get_db_engine
from utils.theme_util import apply_styles

# --- 페이지 설정 ---
st.set_page_config(page_title="데이터 비교 - 지형 분석 서비스", page_icon="🔍", layout="wide")
apply_styles()

# --- 1. 사용자 설정 (스크립트 내에서 고정) ---

# 로컬 데이터가 있는 루트 폴더
LOCAL_DATA_ROOT = r'C:\dev\terrainAnalyzer_st\utils\soil'
# 비교할 DB 테이블 이름
DB_TABLE_NAME = 'kr_soil_map'
# 로컬 데이터의 예상 좌표계 (EPSG 코드)
LOCAL_DATA_CRS = 'EPSG:5174'


# --- 2. 핵심 기능 함수 ---

@st.cache_data
def read_local_data():
    """로컬 Shapefile들을 읽어 하나의 GeoDataFrame으로 합칩니다."""
    shapefiles_to_load = []
    st.write(f"'{LOCAL_DATA_ROOT}' 폴더에서 로컬 Shapefile 검색 중...")
    for dirpath, _, filenames in os.walk(LOCAL_DATA_ROOT):
        for filename in filenames:
            if filename.lower().endswith('.shp'):
                shapefiles_to_load.append(os.path.join(dirpath, filename))
    
    if not shapefiles_to_load:
        st.error("오류: 로컬 Shapefile을 찾을 수 없습니다.")
        return None

    st.write(f"총 {len(shapefiles_to_load)}개의 로컬 Shapefile을 찾았습니다. 데이터를 합치는 중...")
    
    progress_bar = st.progress(0)
    gdf_list = []
    for i, shp_path in enumerate(shapefiles_to_load):
        try:
            gdf = gpd.read_file(shp_path, encoding='euc-kr')
            gdf_list.append(gdf)
        except Exception as e:
            st.warning(f"'{os.path.basename(shp_path)}' 파일을 읽는 중 오류 발생: {e}")
        progress_bar.progress((i + 1) / len(shapefiles_to_load))

    if not gdf_list:
        st.error("오류: 유효한 Shapefile이 하나도 없습니다.")
        return None

    combined_gdf = pd.concat(gdf_list, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry=combined_gdf.geometry)
    combined_gdf.set_crs(LOCAL_DATA_CRS, allow_override=True, inplace=True)
    st.write("로컬 데이터 읽기 및 병합 완료!")
    return combined_gdf

@st.cache_data
def read_db_data(_engine):
    """데이터베이스에서 테이블을 읽어 GeoDataFrame으로 만듭니다."""
    st.write(f"데이터베이스에서 '{DB_TABLE_NAME}' 테이블을 읽는 중...")
    try:
        db_gdf = gpd.read_postgis(f"SELECT * FROM {DB_TABLE_NAME}", _engine, geom_col='geometry')
        st.write("데이터베이스 테이블 읽기 완료!")
        return db_gdf
    except Exception as e:
        st.error(f"오류: 데이터베이스에서 '{DB_TABLE_NAME}' 테이블을 읽는 데 실패했습니다.")
        st.error(f"상세 오류: {e}")
        return None

# --- 3. Streamlit UI 구성 ---

st.markdown("### 🔍 로컬 vs 데이터베이스 데이터 비교")
st.write("""
이 페이지는 로컬에 저장된 토양도 데이터와 데이터베이스에 저장된 토양도 데이터를 비교합니다.
아래 버튼을 누르면 비교를 시작합니다.
""")

if st.button("데이터 비교 실행"):
    
    with st.spinner("데이터베이스에 연결하는 중..."):
        try:
            engine = get_db_engine()
            with engine.connect() as connection:
                st.success("데이터베이스 연결 성공!")
        except Exception as e:
            st.error(f"데이터베이스 연결 실패: {e}")
            st.stop()

    st.markdown("---")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 로컬 파일 데이터")
        with st.spinner("로컬 데이터 처리 중..."):
            local_gdf = read_local_data()

    with col2:
        st.subheader("🗄️ 데이터베이스 데이터")
        with st.spinner("DB 데이터 처리 중..."):
            db_gdf = read_db_data(engine)

    st.markdown("---")

    if local_gdf is not None and db_gdf is not None:
        st.subheader("📊 비교 결과 요약")

        # 1. 데이터 개수 비교
        st.markdown("#### 1. 데이터 개수 (Features)")
        c1, c2 = st.columns(2)
        c1.metric("로컬 파일", f"{len(local_gdf):,} 개")
        c2.metric("데이터베이스", f"{len(db_gdf):,} 개")

        # 2. 좌표계 비교
        st.markdown("#### 2. 좌표계 (CRS)")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**로컬 파일**")
            st.code(local_gdf.crs, language=None)
        with c2:
            st.write("**데이터베이스**")
            st.code(db_gdf.crs, language=None)

        # 3. 전체 영역 비교
        st.markdown("#### 3. 전체 영역 (Total Bounds)")
        st.write("**로컬 파일**")
        st.code(local_gdf.total_bounds, language=None)
        st.write("**데이터베이스**")
        st.code(db_gdf.total_bounds, language=None)

        # 4. 컬럼 정보 비교
        st.markdown("#### 4. 컬럼(속성) 목록")
        local_cols = set(local_gdf.columns)
        db_cols = set(db_gdf.columns)
        c1, c2 = st.columns(2)
        c1.metric("로컬 파일 컬럼 수", len(local_cols))
        c2.metric("데이터베이스 컬럼 수", len(db_cols))
        
        if local_cols == db_cols:
            st.success("컬럼 구성이 동일합니다.")
        else:
            st.warning("컬럼 구성이 다릅니다.")
            if len(local_cols - db_cols) > 0:
                st.write("**로컬에만 있는 컬럼:**")
                st.code(sorted(list(local_cols - db_cols)), language=None)
            if len(db_cols - local_cols) > 0:
                st.write("**DB에만 있는 컬럼:**")
                st.code(sorted(list(db_cols - local_cols)), language=None)

        # 5. 샘플 데이터 출력
        st.markdown("#### 5. 샘플 데이터 (처음 5행)")
        st.write("**로컬 파일 데이터**")
        st.dataframe(local_gdf.head())
        st.write("**데이터베이스 데이터**")
        st.dataframe(db_gdf.head())

    else:
        st.error("데이터 중 하나를 불러오는 데 실패하여 비교를 완료할 수 없습니다.")
