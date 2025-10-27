import os
import warnings

# 복구 모드를 위한 라이브러리 임포트
import fiona
import geopandas as gpd
import pandas as pd
import shapely.geometry
import streamlit as st
from sqlalchemy import create_engine

from utils.config import get_db_engine
from utils.theme_util import apply_styles

# --- 페이지 설정 ---
st.set_page_config(page_title="데이터 업로드 - 지형 분석 서비스",
                   page_icon="📤", layout="wide")
apply_styles()

# --- 1. 설정 (스크립트 내에서 고정) ---

# 로컬 데이터가 있는 루트 폴더
LOCAL_DATA_ROOT = r'C:\dev\terrainAnalyzer_st\utils\soil'
# 업로드할 새 테이블 이름
TARGET_TABLE_NAME = 'kr_soil_map_new2'
# 원본 데이터의 좌표계 (EPSG 코드)
SOURCE_CRS = 'EPSG:5174'


# --- 2. Streamlit UI 구성 ---

st.markdown("### 📤 로컬 토양도 데이터 업로드")

st.write(
    "이 페이지는 로컬 폴더(`{}`)에 있는 모든 Shapefile을 읽어, "
    "데이터베이스에 **`{}`** 라는 새로운 테이블로 업로드합니다.".format(
        LOCAL_DATA_ROOT, TARGET_TABLE_NAME
    )
)
st.warning(
    "**주의:** 만약 데이터베이스에 `{}` 테이블이 이미 존재한다면, "
    "기존 테이블은 삭제되고 **새로운 데이터로 완전히 대체됩니다.**".format(
        TARGET_TABLE_NAME
    ),
    icon="⚠️"
)


if st.button("'{}' 테이블로 업로드 시작".format(TARGET_TABLE_NAME)):

    # --- 3. 업로드 로직 실행 ---

    st.markdown("---")

    # 1. 데이터베이스 연결 확인
    with st.spinner("데이터베이스에 연결하는 중..."):
        try:
            engine = get_db_engine()
            with engine.connect() as connection:
                st.success("데이터베이스 연결 성공!")
        except Exception as e:
            st.error("데이터베이스 연결 실패: {}".format(e))
            st.stop()

    # 2. 로컬 Shapefile 검색
    with st.spinner("'{}' 폴더에서 Shapefile 검색 중...".format(LOCAL_DATA_ROOT)):
        shapefiles_to_load = []
        for dirpath, _, filenames in os.walk(LOCAL_DATA_ROOT):
            for filename in filenames:
                if filename.lower().endswith('.shp'):
                    shapefiles_to_load.append(os.path.join(dirpath, filename))

        if not shapefiles_to_load:
            st.error("오류: 처리할 Shapefile을 찾을 수 없습니다. 폴더 경로를 확인해주세요.")
            st.stop()

        st.write("총 {}개의 Shapefile을 찾았습니다.".format(len(shapefiles_to_load)))

    # 3. Shapefile 읽기 및 병합 (복구 모드 추가)
    st.write("Shapefile을 읽고 하나로 합치는 중...")
    progress_bar = st.progress(0, text="파일 읽는 중...")
    gdf_list = []
    error_files = []
    recovered_files = []

    for i, shp_path in enumerate(shapefiles_to_load):
        filename = os.path.basename(shp_path)
        progress_bar.progress((i + 1) / len(shapefiles_to_load),
                              text="파일 읽는 중: {}".format(filename))

        try:
            # 1단계: 일반 모드로 읽기 시도
            gdf = gpd.read_file(shp_path, encoding='euc-kr')
            gdf_list.append(gdf)
        except Exception as e:
            st.warning(
                "'{}' 파일을 읽는 중 오류 발생: {}. 복구 모드를 시도합니다.".format(filename, e))

            # 2단계: 복구 모드 실행
            try:
                with fiona.open(shp_path, 'r', encoding='euc-kr') as collection:
                    geometries = [shapely.geometry.shape(
                        feature['geometry']) for feature in collection]
                    attributes = [dict(feature['properties'])
                                  for feature in collection]

                gdf = gpd.GeoDataFrame(
                    attributes, geometry=geometries, crs=SOURCE_CRS)
                gdf_list.append(gdf)
                st.info("✅ '{}' 파일을 성공적으로 복구했습니다.".format(filename))
                recovered_files.append(filename)
            except Exception as e2:
                st.error("❌ '{}' 파일은 복구에도 실패했습니다: {}".format(filename, e2))
                error_files.append(filename)

    if not gdf_list:
        st.error("오류: 유효한 Shapefile이 하나도 없습니다.")
        st.stop()

    st.write("모든 파일을 성공적으로 읽었습니다. 데이터를 하나로 합치는 중...")
    combined_gdf = pd.concat(gdf_list, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(
        combined_gdf, geometry=combined_gdf.geometry)

    st.write("총 {:,}개의 피처(feature)를 성공적으로 합쳤습니다.".format(len(combined_gdf)))

    # 4. 좌표계 설정
    st.write("데이터의 좌표계를 '{}'로 설정합니다.".format(SOURCE_CRS))
    combined_gdf.set_crs(SOURCE_CRS, allow_override=True, inplace=True)

    # 5. 데이터베이스에 업로드
    spinner_text = "'{}' 테이블 이름으로 데이터베이스에 업로드 중... (시간이 소요될 수 있습니다)".format(
        TARGET_TABLE_NAME)
    with st.spinner(spinner_text):
        try:
            # 컬럼 이름을 소문자로 변경 (DB 호환성)
            combined_gdf.columns = [col.lower()
                                    for col in combined_gdf.columns]

            combined_gdf.to_postgis(
                name=TARGET_TABLE_NAME,
                con=engine,
                if_exists='replace',
                index=False,
                chunksize=1000
            )
            st.success("🎉 업로드 완료!")
            st.markdown(
                "데이터베이스에 **`{}`** 테이블이 성공적으로 생성/대체되었습니다.\n\n"
                "이제 데이터베이스 관리 도구에서 기존 테이블(`kr_soil_map`)을 백업하거나 삭제하고, "
                "새로 업로드된 `{}`의 이름을 `kr_soil_map`으로 변경하여 사용하시면 됩니다.".format(
                    TARGET_TABLE_NAME, TARGET_TABLE_NAME
                )
            )
            if recovered_files:
                st.info("다음 파일들은 복구 모드를 통해 성공적으로 처리되었습니다:")
                st.json(recovered_files)
            if error_files:
                st.warning("다음 파일들은 최종적으로 처리에 실패하여 업로드에서 제외되었습니다:")
                st.json(error_files)

        except Exception as e:
            st.error("데이터베이스 업로드 중 심각한 오류 발생: {}".format(e))
            st.error("데이터 또는 테이블 이름을 확인해주세요.")
