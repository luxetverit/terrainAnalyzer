
import os
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine
import warnings

# 경고 메시지 무시 (선택 사항)
warnings.filterwarnings('ignore', message='.*pandas.concat*.'))

# --- 1. 사용자 설정 ---

# 데이터가 있는 루트 폴더
SOIL_DATA_ROOT = r'C:\dev\terrainAnalyzer_st\utils\soil'

# 데이터베이스 접속 정보 (사용자 환경에 맞게 수정)
DB_USER = 'your_username'  # 예: postgres
DB_PASSWORD = 'your_password'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'your_database'    # 예: gisdb

# 업로드할 테이블 이름
TABLE_NAME = 'kr_soil_map_new'

# 원본 데이터의 좌표계 (EPSG 코드)
# 중요: 원본 데이터의 좌표계를 정확히 알고 있어야 합니다.
# 이전 대화 내용을 바탕으로 'EPSG:5174'로 추정합니다. 확실하지 않으면 확인이 필요합니다.
SOURCE_CRS = 'EPSG:5174'


# --- 2. 스크립트 실행 부분 (이 아래는 수정할 필요 없음) ---

def main():
    """
    지정된 폴더 내의 모든 Shapefile을 읽어 하나로 합친 후,
    PostGIS 데이터베이스에 업로드하는 메인 함수
    """
    print(f"데이터베이스에 연결 중... (Host: {DB_HOST})")
    
    try:
        # 데이터베이스 연결 엔진 생성
        engine_str = f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        engine = create_engine(engine_str)
        
        # 연결 테스트
        with engine.connect() as connection:
            print("데이터베이스 연결 성공!")
    except Exception as e:
        print(f"데이터베이스 연결 실패: {e}")
        print("DB 접속 정보를 다시 확인해주세요.")
        return

    # 모든 Shapefile 경로 찾기
    shapefiles_to_load = []
    print(f"\n'{SOIL_DATA_ROOT}' 폴더에서 Shapefile 검색 중...")
    for dirpath, _, filenames in os.walk(SOIL_DATA_ROOT):
        for filename in filenames:
            if filename.lower().endswith('.shp'):
                full_path = os.path.join(dirpath, filename)
                shapefiles_to_load.append(full_path)

    if not shapefiles_to_load:
        print("오류: 처리할 Shapefile을 찾을 수 없습니다. 폴더 경로를 확인해주세요.")
        return
        
    print(f"총 {len(shapefiles_to_load)}개의 Shapefile을 찾았습니다.")

    # 모든 GeoDataFrame을 담을 리스트
    gdf_list = []
    
    print("\nShapefile을 읽고 하나로 합치는 중...")
    for i, shp_path in enumerate(shapefiles_to_load):
        try:
            gdf = gpd.read_file(shp_path, encoding='euc-kr') # 한글 인코딩 지정
            gdf_list.append(gdf)
            # 진행 상황 표시
            print(f"  - {i+1}/{len(shapefiles_to_load)}: '{os.path.basename(shp_path)}' 읽기 완료")
        except Exception as e:
            print(f"  - 오류: '{os.path.basename(shp_path)}' 파일을 읽는 중 오류 발생: {e}")
            continue
            
    if not gdf_list:
        print("오류: 유효한 Shapefile이 하나도 없습니다.")
        return

    # 모든 GeoDataFrame을 하나로 합치기
    combined_gdf = pd.concat(gdf_list, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry=combined_gdf.geometry)
    
    print(f"\n총 {len(combined_gdf)}개의 피처(feature)를 성공적으로 합쳤습니다.")

    # 좌표계 설정
    print(f"데이터의 좌표계를 '{SOURCE_CRS}'로 설정합니다.")
    combined_gdf.set_crs(SOURCE_CRS, allow_override=True, inplace=True)

    # 데이터베이스에 업로드
    print(f"\n'{TABLE_NAME}' 테이블 이름으로 데이터베이스에 업로드 시작...")
    print("(데이터 양에 따라 몇 분 정도 소요될 수 있습니다)")
    
    try:
        combined_gdf.to_postgis(
            name=TABLE_NAME,
            con=engine,
            if_exists='replace',  # 테이블이 있으면 대체 ('append'는 추가, 'fail'은 오류)
            index=False,
            chunksize=1000  # 대용량 데이터를 위해 청크 단위로 업로드
        )
        print("\n업로드 완료!")
        print(f"데이터베이스 '{DB_NAME}'에 '{TABLE_NAME}' 테이블이 성공적으로 생성/대체되었습니다.")

    except Exception as e:
        print(f"\n데이터베이스 업로드 중 심각한 오류 발생: {e}")
        print("데이터 또는 테이블 이름을 확인해주세요.")


if __name__ == '__main__':
    main()
