import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import glob
import pandas as pd
import re
import time
from shapely.geometry import Point
from scipy.interpolate import griddata
import platform
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_origin
import tempfile
from sqlalchemy import create_engine

import richdem as rd

from utils.color_palettes import ALL_PALETTES, get_palette_preview_html
from utils.theme_util import apply_theme_toggle

engine = create_engine("postgresql://postgres:asdfasdf12@localhost:5432/gisDB")

# ----- Streamlit 페이지/폰트 세팅 -----
pixel_size = 1
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="또초자료 운사원 - 결과",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
main_col = apply_theme_toggle()
st.markdown("""
<style>
.big-text { font-size: 24px !important; font-weight: bold; }
.result-text { font-size: 18px !important; font-weight: bold; }
.stButton>button {
    width: 100%;
    border-radius: 20px !important;
    font-size: 18px !important;
    padding: 10px 24px !important;
}
</style>
""", unsafe_allow_html=True)

# ----- 세션 상태 체크 -----
if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file is None:
    st.error("업로드된 파일이 없습니다. 메인 페이지로 돌아가세요.")
    if st.button("메인 페이지로 돌아가기"):
        st.switch_page("app.py")
    st.stop()
if 'processing_done' not in st.session_state:
    st.error("처리가 완료되지 않았습니다. 처리 중 페이지로 돌아가세요.")
    if st.button("이전 페이지로 돌아가기"):
        st.switch_page("pages/03_처리중.py")
    st.stop()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {
        "done": True,
        "message": "분석이 완료되었습니다!",
        "palettes": {
            "elevation": "terrain",
            "slope": "RdYlBu"
        }
    }

elevation_palette = st.session_state.analysis_results["palettes"]["elevation"]
slope_palette = st.session_state.analysis_results["palettes"]["slope"]

# ----- 좌표 문자열 파싱 함수 -----
def parse_coordinate_string(coord_str):
    match = re.match(r"([\d\-. ]+)[, ]+([\d\-. ]+)", coord_str)
    if not match:
        return None
    lat_str, lon_str = match.group(1), match.group(2)
    def dms_to_deg(s):
        s = s.replace(',', ' ').replace('-', ' ').replace('°', ' ').replace('\'', ' ').replace('"', ' ')
        parts = [float(x) for x in re.split(r'\s+', s.strip()) if x]
        if len(parts) == 3:
            deg, minute, sec = parts
            return deg + (minute/60) + (sec/3600)
        elif len(parts) == 2:
            deg, minute = parts
            return deg + (minute/60)
        else:
            return float(parts[0])
    lat = dms_to_deg(lat_str)
    lon = dms_to_deg(lon_str)
    return Point(lon, lat)

# ----- 모든 geometry에서 point 추출 -----
def extract_points(gdf, elevation_field):
    xs, ys, zs = [], [], []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        z = row[elevation_field]
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == 'Point':
            xs.append(geom.x)
            ys.append(geom.y)
            zs.append(z)
        elif geom.geom_type == 'MultiPoint':
            for pt in geom.geoms:
                xs.append(pt.x)
                ys.append(pt.y)
                zs.append(z)
        elif geom.geom_type == 'LineString':
            for pt in geom.coords:
                xs.append(pt[0])
                ys.append(pt[1])
                zs.append(z)
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                for pt in line.coords:
                    xs.append(pt[0])
                    ys.append(pt[1])
                    zs.append(z)
        elif geom.geom_type == 'Polygon':
            for pt in geom.exterior.coords:
                xs.append(pt[0])
                ys.append(pt[1])
                zs.append(z)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                for pt in poly.exterior.coords:
                    xs.append(pt[0])
                    ys.append(pt[1])
                    zs.append(z)
    return np.array(xs), np.array(ys), np.array(zs)

# ----- 표고 통계 -----
def calc_elevation_stats(array):
    arr = array[np.isfinite(array)]
    return {
        'min': float(np.nanmin(arr)) if arr.size > 0 else np.nan,
        'max': float(np.nanmax(arr)) if arr.size > 0 else np.nan,
        'mean': float(np.nanmean(arr)) if arr.size > 0 else np.nan,
        'area': arr.size
    }

def merge_and_standardize_gdf_list(gdf_list, altitude_cols, geometry_cols, elevation_field, target_crs):
    dfs = []
    for gdf in gdf_list:
        try:
            geom_col = next((col for col in geometry_cols if col in gdf.columns), None)
            if geom_col and geom_col != 'geometry':
                gdf = gdf.rename(columns={geom_col: 'geometry'})
            found_col = next((col for col in altitude_cols if col in gdf.columns), None)
            if found_col and found_col != elevation_field:
                gdf = gdf.rename(columns={found_col: elevation_field})
            if elevation_field not in gdf.columns:
                gdf[elevation_field] = np.nan
            gdf = gdf.set_geometry('geometry')
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            gdf = gdf.to_crs(target_crs)
            gdf = gdf[gdf['geometry'].notna() & (~gdf['geometry'].is_empty)]
            gdf = gdf[~gdf[elevation_field].isna()]
            dfs.append(gdf[['geometry', elevation_field]])
        except Exception as e:
            st.warning(f"gdf 병합 실패: {e}")
    if dfs:
        merged_gdf = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=dfs[0].crs)
        return merged_gdf
    else:
        return None

# ---- 데이터 준비 (임시파일/캐시 사용) ----
with st.expander("🔎 업로드/샘플 도엽 병합 + DEM/경사/경사향 분석 및 클리핑 결과", expanded=True):

    user_shp_path = st.session_state.get("temp_file_path", None) \
        or st.session_state.get("uploaded_file_path", None)
    if not user_shp_path or not os.path.exists(user_shp_path):
        st.error("업로드된 파일이 없습니다. (session_state['temp_file_path'])")
        st.stop()

    if 'matched_sheets' in st.session_state:
        matched_sheets = st.session_state['matched_sheets']
    else:
        st.error("기초분석 결과가 없습니다. 먼저 기초분석을 실행해주세요.")
        st.stop()

    original_sheets = st.session_state.map_index_results.get('original_matched_sheets', [])
    nearby_sheets = [sheet for sheet in matched_sheets if sheet not in original_sheets]

    user_gdf = gpd.read_file(user_shp_path)
    user_gdf = user_gdf.to_crs("EPSG:5186")

    all_sheets = original_sheets + nearby_sheets
    
    sheet_str = ",".join([f"'{x}'" for x in all_sheets])
    sql = f"""
    SELECT geometry, elevation
    FROM kr_shp
    WHERE scode IN ({sheet_str})
    """

    sample_gdf = gpd.read_postgis(sql, engine, geom_col='geometry')

    altitude_cols = ['표고', '등고수치', '수치', '고도', 'elev', 'elevation', 'height', 'Z']
    geometry_cols = ['geometry', 'Geometry', 'GEOMETRY', 'geom', 'Geom', '좌표']
    elevation_field = "elevation"
    target_crs = "EPSG:5186"

    all_gdf = [user_gdf, sample_gdf]

    merged_gdf = merge_and_standardize_gdf_list(
        all_gdf,
        altitude_cols,
        geometry_cols,
        elevation_field,
        target_crs
    )

    if merged_gdf is None or len(merged_gdf) == 0:
        st.error("도엽 병합 실패 또는 데이터 없음.")
        st.stop()
    st.success(f"업로드+도엽 병합 완료! 총 {len(merged_gdf)}개 객체")
    
    # ---- DEM 보간 및 tif 임시저장 ----
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmpfile:
        tif_path = tmpfile.name
    
    with st.spinner("DEM 보간 중..."):
        xs, ys, zs = extract_points(merged_gdf, elevation_field)
        minx, miny, maxx, maxy = merged_gdf.total_bounds
        grid_x, grid_y = np.mgrid[minx:maxx:pixel_size, miny:maxy:pixel_size]
        dem_grid = griddata((xs, ys), zs, (grid_x, grid_y), method='linear', fill_value=np.nan)
        transform = from_origin(minx, maxy, pixel_size, pixel_size)
        with rasterio.open(
            tif_path, 'w',
            driver='GTiff',
            height=dem_grid.shape[1],
            width=dem_grid.shape[0],
            count=1,
            dtype='float32',
            crs=merged_gdf.crs,
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(np.flipud(dem_grid.T), 1)
        st.success("DEM 보간 완료!")

    # ==== 분석: 경사, 경사향 등(DEM 전체) ====
    selected_types = st.session_state.get('selected_analysis_types', [])

    # 분석 결과 tif 저장 경로
    slope_tif = None
    aspect_tif = None

    # 결과 numpy 배열 dict로 저장
    analysis_arrays = {'elevation': dem_grid}

    # Slope 분석
    if 'slope' in selected_types:
        st.info("경사(Slope) 분석(DEM 전체) 수행 중...")
        dem_rd = rd.LoadGDAL(tif_path)
        slope_arr = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')
        analysis_arrays['slope'] = slope_arr
        # 결과를 tif로 저장
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as stmp:
            slope_tif = stmp.name
        with rasterio.open(
            slope_tif, 'w',
            driver='GTiff',
            height=slope_arr.shape[0],
            width=slope_arr.shape[1],
            count=1,
            dtype='float32',
            crs=merged_gdf.crs,
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(np.flipud(slope_arr.T), 1)
        st.success("경사(Slope) 분석 완료!")

    # Aspect 분석
    if 'aspect' in selected_types:
        st.info("경사향(Aspect) 분석(DEM 전체) 수행 중...")
        dem_rd = rd.LoadGDAL(tif_path)
        aspect_arr = rd.TerrainAttribute(dem_rd, attrib='aspect')
        analysis_arrays['aspect'] = aspect_arr
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as atmp:
            aspect_tif = atmp.name
        with rasterio.open(
            aspect_tif, 'w',
            driver='GTiff',
            height=aspect_arr.shape[0],
            width=aspect_arr.shape[1],
            count=1,
            dtype='float32',
            crs=merged_gdf.crs,
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(np.flipud(aspect_arr.T), 1)
        st.success("경사향(Aspect) 분석 완료!")

    # ---- 클리핑(폴리곤 적용) ----
    with st.spinner("최종 클리핑(폴리곤 적용) 중..."):
        user_gdf = gpd.read_file(user_shp_path)
        clip_geoms = [g for g in user_gdf.geometry if g.is_valid and not g.is_empty]

        if not clip_geoms:
            st.error("클리핑에 사용할 유효한 geometry(폴리곤)가 없습니다.")
            st.stop()
        # DEM bounds
        poly_bounds = user_gdf.total_bounds
        # dem_grid와 transform으로 bounds 직접 확인도 가능
        # 각 결과별 tif로 클리핑!
        dem_results = {}
        # --- DEM(표고) ---
        with rasterio.open(tif_path) as src:
            raster_bounds = src.bounds
            overlap = not (
                poly_bounds[2] < raster_bounds.left or
                poly_bounds[0] > raster_bounds.right or
                poly_bounds[3] < raster_bounds.bottom or
                poly_bounds[1] > raster_bounds.top
            )
            if not overlap:
                st.error("클리핑 폴리곤이 DEM 래스터 범위와 겹치지 않습니다. (좌표계/영역 확인 필요)")
                st.stop()
            clipped, clipped_transform = mask(src, clip_geoms, crop=True, nodata=np.nan)
            clipped_dem = clipped[0]
            dem_results['elevation'] = {
                'stats': calc_elevation_stats(clipped_dem),
                'grid': clipped_dem
            }
        # --- Slope(경사) ---
        if 'slope' in selected_types and slope_tif:
            with rasterio.open(slope_tif) as src:
                clipped, clipped_transform = mask(src, clip_geoms, crop=True, nodata=np.nan)
                clipped_slope = clipped[0]
                dem_results['slope'] = {
                    'stats': calc_elevation_stats(clipped_slope),
                    'grid': clipped_slope
                }
        # --- Aspect(경사향) ---
        if 'aspect' in selected_types and aspect_tif:
            with rasterio.open(aspect_tif) as src:
                clipped, clipped_transform = mask(src, clip_geoms, crop=True, nodata=np.nan)
                clipped_aspect = clipped[0]
                dem_results['aspect'] = {
                    'stats': calc_elevation_stats(clipped_aspect),
                    'grid': clipped_aspect
                }
        st.success("최종 클리핑(폴리곤 적용) 완료!")

        st.session_state['dem_results'] = dem_results

    st.markdown("## 작업이 완료되었어요")
    if 'dem_results' in st.session_state and st.session_state.dem_results:
        dem_results = st.session_state.dem_results
        if 'elevation' in selected_types and dem_results['elevation']['stats']:
            elev_stats = dem_results['elevation']['stats']
            st.markdown(f"## 표고는 {elev_stats['min']:.1f}~{elev_stats['max']:.1f}m, 평균표고는 {elev_stats['mean']:.1f}m로 분석되었어요.")

    st.markdown("### 결과 페이지로 이동합니다...")
    time.sleep(3)
    st.switch_page("pages/05_자료다운.py")
    # st.page_link("pages/05_자료다운.py", label="**🖼️시각화 및 다운로드 페이지로 이동**")
