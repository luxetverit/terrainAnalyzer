import os
import tempfile
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import richdem as rd
import streamlit as st
from rasterio.mask import mask
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from sqlalchemy import create_engine

from utils.theme_util import apply_styles

# --- 1. Page Configuration and Styling ---
st.set_page_config(page_title="분석 실행 - 지형 분석 서비스",
                   page_icon="⚙️",
                   layout="wide",
                   initial_sidebar_state="collapsed")
apply_styles()

# --- 2. Session State and DB Check ---
if 'gdf' not in st.session_state:
    st.warning("분석할 데이터가 없습니다. 홈 페이지로 돌아가 파일을 먼저 업로드해주세요.")
    if st.button("홈으로 돌아가기"):
        st.switch_page("app.py")
    st.stop()

try:
    engine = create_engine(
        "postgresql://postgres:asdfasdf12@localhost:5432/gisDB")
    with engine.connect() as connection:
        pass
except Exception as e:
    st.error(f"데이터베이스에 연결할 수 없습니다. PostGIS DB가 실행 중인지 확인하세요. 에러: {e}")
    st.stop()

# --- 3. Helper Functions ---


def calc_stats(array):
    arr = array[np.isfinite(array)]
    if arr.size == 0:
        return {'min': 0, 'max': 0, 'mean': 0, 'area': 0}
    return {'min': float(np.nanmin(arr)), 'max': float(np.nanmax(arr)), 'mean': float(np.nanmean(arr)), 'area': arr.size}


def calculate_binned_stats(grid, num_bins=10):
    grid_flat = grid[~np.isnan(grid)]
    if grid_flat.size == 0:
        return []
    min_val, max_val = np.min(grid_flat), np.max(grid_flat)
    bins = np.linspace(min_val, max_val, num_bins + 1)
    hist, bin_edges = np.histogram(grid_flat, bins=bins)
    binned_stats = []
    for i in range(num_bins):
        binned_stats.append(
            {"bin_range": f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}", "area": hist[i]})
    return binned_stats


def extract_points_from_geometries(gdf, elevation_col='elevation'):
    xs, ys, zs = [], [], []
    if 'geometry' not in gdf.columns or gdf.geometry.isnull().all():
        return np.array([]), np.array([]), np.array([])

    for _, row in gdf.iterrows():
        geom = row.geometry
        z = row.get(elevation_col)

        if geom is None or geom.is_empty or z is None or not np.isfinite(z):
            continue

        geoms_to_process = [geom] if not geom.geom_type.startswith(
            'Multi') else list(geom.geoms)

        for part in geoms_to_process:
            if part.geom_type == 'Point':
                xs.append(part.x)
                ys.append(part.y)
                zs.append(z)
            elif part.geom_type in ('LineString', 'LinearRing'):
                for p in part.coords:
                    xs.append(p[0])
                    ys.append(p[1])
                    zs.append(z)
            elif part.geom_type == 'Polygon':
                for p in part.exterior.coords:
                    xs.append(p[0])
                    ys.append(p[1])
                    zs.append(z)
    return np.array(xs), np.array(ys), np.array(zs)


# --- 4. Page Header ---
st.markdown('''<div class="page-header"><h1>분석 실행</h1><p>선택하신 항목에 대한 분석을 실행합니다. 이 작업은 몇 분 정도 소요될 수 있습니다.</p></div>''', unsafe_allow_html=True)

# --- 5. Main Analysis Logic ---
if 'dem_results' not in st.session_state:
    user_gdf_original = st.session_state.gdf
    selected_types = st.session_state.get('selected_analysis_types', [])
    dem_results = {}

    with st.status("지형 분석 진행 중...", expanded=True) as status:
        try:
            # --- Part 1: Data Preparation (Robust Method) ---
            status.write("1/5: 데이터 준비 및 좌표계 통일 중...")
            target_crs = "EPSG:5186"

            # 1a. Validate and prepare user's boundary file
            if 'geometry' not in user_gdf_original.columns:
                raise ValueError("업로드한 파일에 유효한 공간 정보('geometry' 열)가 없습니다.")
            user_gdf_reprojected = user_gdf_original.to_crs(target_crs)
            if user_gdf_reprojected.empty:
                raise ValueError("업로드한 파일에서 유효한 분석 영역을 찾을 수 없습니다.")

            # 1b. Fetch contour data from DB with a buffer for improved accuracy
            buffer_m = 100.0  # 100m 버퍼
            user_bounds = user_gdf_reprojected.total_bounds

            # 확장된 경계 상자 생성
            expanded_bounds = (
                user_bounds[0] - buffer_m,
                user_bounds[1] - buffer_m,
                user_bounds[2] + buffer_m,
                user_bounds[3] + buffer_m,
            )

            bbox_wkt = (
                f"POLYGON(({expanded_bounds[0]} {expanded_bounds[1]}, "
                f"{expanded_bounds[2]} {expanded_bounds[1]}, "
                f"{expanded_bounds[2]} {expanded_bounds[3]}, "
                f"{expanded_bounds[0]} {expanded_bounds[3]}, "
                f"{expanded_bounds[0]} {expanded_bounds[1]}))"
            )
            sql = f"SELECT geometry, elevation FROM kr_contour_map WHERE ST_Intersects(geometry, ST_GeomFromText('{bbox_wkt}', 5186));"
            status.write("데이터베이스에서 버퍼 영역을 포함한 등고선 데이터를 조회합니다.")
            contour_gdf = gpd.read_postgis(sql, engine, geom_col='geometry')

            # --- Part 2: DEM and Raster Analysis ---
            dem_needed = any(item in selected_types for item in [
                             'elevation', 'slope', 'aspect'])

            if dem_needed:
                status.write("2/5: DEM(수치표고모델) 생성 준비 중...")
                if contour_gdf.empty:
                    raise ValueError(
                        "표고 분석에 필요한 등고선 데이터를 데이터베이스에서 찾을 수 없습니다. 분석하려는 지역이 DB 서비스 범위를 벗어났을 수 있습니다.")

                contour_gdf = contour_gdf.to_crs(target_crs)
                xs, ys, zs = extract_points_from_geometries(
                    contour_gdf, 'elevation')

                if xs.size == 0:
                    raise ValueError("데이터베이스에서 추출한 등고선 데이터에 유효한 고도 포인트가 없습니다.")

                status.write("3/5: DEM 보간 작업 중 (버퍼 영역 포함)...")
                # 보간 그리드는 확장된 경계를 사용
                minx, miny, maxx, maxy = expanded_bounds
                pixel_size = 1.0
                grid_x, grid_y = np.mgrid[minx:maxx:pixel_size,
                                          miny:maxy:pixel_size]

                dem_grid = griddata(
                    (xs, ys), zs, (grid_x, grid_y), method='linear', fill_value=np.nan)
                transform = from_origin(
                    expanded_bounds[0], expanded_bounds[3], pixel_size, pixel_size)

                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmpfile:
                    dem_tif_path = tmpfile.name

                with rasterio.open(
                    dem_tif_path, 'w', driver='GTiff', height=dem_grid.shape[1], width=dem_grid.shape[0],
                    count=1, dtype='float32', crs=target_crs, transform=transform, nodata=np.nan
                ) as dst:
                    dst.write(np.flipud(dem_grid.T), 1)

                status.write("4/5: 경사/경사향 분석 및 클리핑 중...")
                temp_files_to_clean = {'elevation': dem_tif_path}

                if 'slope' in selected_types:
                    dem_rd = rd.LoadGDAL(dem_tif_path)
                    slope_arr = rd.TerrainAttribute(
                        dem_rd, attrib='slope_degrees')
                    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as stmp:
                        slope_tif_path = stmp.name
                        with rasterio.open(slope_tif_path, 'w', driver='GTiff', height=slope_arr.shape[0], width=slope_arr.shape[1], count=1, dtype='float32', crs=target_crs, transform=transform, nodata=np.nan) as dst:
                            dst.write(slope_arr, 1)
                        temp_files_to_clean['slope'] = slope_tif_path

                if 'aspect' in selected_types:
                    dem_rd = rd.LoadGDAL(dem_tif_path)
                    aspect_arr = rd.TerrainAttribute(dem_rd, attrib='aspect')
                    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as atmp:
                        aspect_tif_path = atmp.name
                        with rasterio.open(aspect_tif_path, 'w', driver='GTiff', height=aspect_arr.shape[0], width=aspect_arr.shape[1], count=1, dtype='float32', crs=target_crs, transform=transform, nodata=np.nan) as dst:
                            dst.write(aspect_arr, 1)
                        temp_files_to_clean['aspect'] = aspect_tif_path

                clip_geoms = [
                    g for g in user_gdf_reprojected.geometry if g.is_valid and not g.is_empty]
                if not clip_geoms:
                    raise ValueError("클리핑에 사용할 유효한 폴리곤이 업로드된 파일에 없습니다.")

                for analysis_type, tif_path in temp_files_to_clean.items():
                    if analysis_type in selected_types or (analysis_type == 'elevation' and dem_needed):
                        with rasterio.open(tif_path) as src:
                            clipped_grid, _ = mask(
                                src, clip_geoms, crop=True, nodata=np.nan)
                            clipped_grid = clipped_grid[0]
                            dem_results[analysis_type] = {
                                'grid': clipped_grid,
                                'stats': calc_stats(clipped_grid),
                                'binned_stats': calculate_binned_stats(clipped_grid)
                            }

                for path in temp_files_to_clean.values():
                    if path and os.path.exists(path):
                        os.remove(path)
            else:
                st.info("DEM 기반 분석(표고, 경사, 경사향)이 선택되지 않았습니다.")

            # --- Part 3: Vector Analysis ---
            status.write("5/5: 벡터 데이터 분석 (토양도 등) 중...")
            if any(item in selected_types for item in ['soil', 'hsg', 'landcover']):
                user_geom = user_gdf_reprojected.union_all()
                user_wkt = user_geom.wkt

                if 'soil' in selected_types:
                    sql_soil = f"SELECT ST_Intersection(t1.geom, ST_GeomFromText('{user_wkt}', 5186)) AS geometry, t1.* FROM public.kr_soil_map AS t1 WHERE ST_Intersects(t1.geom, ST_GeomFromText('{user_wkt}', 5186));"
                    dem_results['soil'] = {'gdf': gpd.read_postgis(
                        sql_soil, engine, geom_col='geometry')}

                if 'hsg' in selected_types:
                    sql_hsg = f"SELECT ST_Intersection(t1.geom, ST_GeomFromText('{user_wkt}', 5186)) AS geometry, t1.* FROM public.kr_hsg_map AS t1 WHERE ST_Intersects(t1.geom, ST_GeomFromText('{user_wkt}', 5186));"
                    dem_results['hsg'] = {'gdf': gpd.read_postgis(
                        sql_hsg, engine, geom_col='geometry')}

                if 'landcover' in selected_types:
                    sql_landcover = f"SELECT ST_Intersection(t1.geom, ST_GeomFromText('{user_wkt}', 5186)) AS geometry, t1.* FROM public.kr_landcover_map AS t1 WHERE ST_Intersects(t1.geom, ST_GeomFromText('{user_wkt}', 5186));"
                    dem_results['landcover'] = {'gdf': gpd.read_postgis(
                        sql_landcover, engine, geom_col='geometry')}

            st.session_state.dem_results = dem_results
            status.update(label="분석 완료!", state="complete", expanded=False)

        except Exception as e:
            import traceback
            st.error(f"분석에 실패했습니다: {e}")
            st.code(traceback.format_exc())
            st.stop()

    st.success("모든 분석이 완료되었습니다. 결과 페이지로 이동합니다.")
    time.sleep(3)
    st.switch_page("pages/05_자료다운.py")
