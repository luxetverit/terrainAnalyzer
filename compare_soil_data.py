
import os
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine
import warnings

# 경고 메시지 무시 (선택 사항)
warnings.filterwarnings('ignore', message='.*pandas.concat*.'))

# --- 1. 사용자 설정 ---

# 로컬 데이터가 있는 루트 폴더
LOCAL_DATA_ROOT = r'C:\dev\terrainAnalyzer_st\utils\soil'

# 데이터베이스 접속 정보 (사용자 환경에 맞게 수정)
DB_USER = 'your_username'  # 예: postgres
DB_PASSWORD = 'your_password'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'your_database'    # 예: gisdb

# 비교할 DB 테이블 이름
# 중요: 현재 DB에 있는 토양도 테이블 이름을 정확히 입력해야 합니다.
DB_TABLE_NAME = 'kr_soil_map'

# 로컬 데이터의 예상 좌표계 (EPSG 코드)
LOCAL_DATA_CRS = 'EPSG:5174'


# --- 2. 스크립트 실행 부분 (이 아래는 수정할 필요 없음) ---

def read_local_data():
    """로컬 Shapefile들을 읽어 하나의 GeoDataFrame으로 합칩니다."""
    shapefiles_to_load = []
    print(f"'{LOCAL_DATA_ROOT}' 폴더에서 로컬 Shapefile 검색 중...")
    for dirpath, _, filenames in os.walk(LOCAL_DATA_ROOT):
        for filename in filenames:
            if filename.lower().endswith('.shp'):
                shapefiles_to_load.append(os.path.join(dirpath, filename))
    
    if not shapefiles_to_load:
        print("오류: 로컬 Shapefile을 찾을 수 없습니다.")
        return None

    print(f"총 {len(shapefiles_to_load)}개의 로컬 Shapefile을 찾았습니다. 데이터를 합치는 중...")
    gdf_list = [gpd.read_file(p, encoding='euc-kr') for p in shapefiles_to_load]
    combined_gdf = pd.concat(gdf_list, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry=combined_gdf.geometry)
    combined_gdf.set_crs(LOCAL_DATA_CRS, allow_override=True, inplace=True)
    return combined_gdf

def read_db_data(engine):
    """데이터베이스에서 테이블을 읽어 GeoDataFrame으로 만듭니다."""
    print(f"\n데이터베이스에서 '{DB_TABLE_NAME}' 테이블을 읽는 중...")
    try:
        db_gdf = gpd.read_postgis(f"SELECT * FROM {DB_TABLE_NAME}", engine, geom_col='geometry')
        return db_gdf
    except Exception as e:
        print(f"오류: 데이터베이스에서 '{DB_TABLE_NAME}' 테이블을 읽는 데 실패했습니다.")
        print(f"상세 오류: {e}")
        return None

def main():
    """로컬 데이터와 DB 데이터를 비교하는 메인 함수"""
    # 로컬 데이터 읽기
    local_gdf = read_local_data()
    if local_gdf is None:
        return

    # 데이터베이스 연결 및 데이터 읽기
    try:
        engine_str = f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        engine = create_engine(engine_str)
        db_gdf = read_db_data(engine)
        if db_gdf is None:
            return
    except Exception as e:
        print(f"데이터베이스 연결 실패: {e}")
        return

    # --- 비교 결과 출력 ---
    print("\n--- 데이터 비교 결과 ---")
    
    # 1. 데이터 개수 비교
    print("\n1. 데이터 개수 (Features)")
    print(f"  - 로컬 파일: {len(local_gdf):,} 개")
    print(f"  - 데이터베이스: {len(db_gdf):,} 개")

    # 2. 좌표계 비교
    print("\n2. 좌표계 (CRS)")
    print(f"  - 로컬 파일: {local_gdf.crs}")
    print(f"  - 데이터베이스: {db_gdf.crs}")

    # 3. 전체 영역 비교
    print("\n3. 전체 영역 (Total Bounds)")
    print(f"  - 로컬 파일: {local_gdf.total_bounds}")
    print(f"  - 데이터베이스: {db_gdf.total_bounds}")

    # 4. 컬럼 정보 비교
    print("\n4. 컬럼(속성) 목록")
    local_cols = set(local_gdf.columns)
    db_cols = set(db_gdf.columns)
    print(f"  - 로컬 파일 컬럼 수: {len(local_cols)}")
    print(f"  - 데이터베이스 컬럼 수: {len(db_cols)}")
    if local_cols == db_cols:
        print("  - 컬럼 구성이 동일합니다.")
    else:
        print("  - 컬럼 구성이 다릅니다.")
        if len(local_cols - db_cols) > 0:
            print(f"    - 로컬에만 있는 컬럼: {sorted(list(local_cols - db_cols))}")
        if len(db_cols - local_cols) > 0:
            print(f"    - DB에만 있는 컬럼: {sorted(list(db_cols - local_cols))}")

    # 5. 샘플 데이터 출력
    print("\n5. 샘플 데이터 (처음 5행)")
    print("\n  --- 로컬 파일 데이터 ---")
    print(local_gdf.head())
    print("\n  --- 데이터베이스 데이터 ---")
    print(db_gdf.head())
    print("\n-------------------------")

if __name__ == '__main__':
    main()
