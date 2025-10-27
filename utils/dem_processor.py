import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from shapely.geometry import MultiPolygon, Polygon

from utils.color_palettes import get_palette
from utils.plot_helpers import generate_aspect_bins  # 참고: 경사향 구간만
from utils.plot_helpers import (generate_custom_intervals,
                                generate_detailed_slope_intervals,
                                generate_slope_intervals)


def calculate_binned_stats(grid, bins, labels):
    """사전 정의된 구간과 라벨을 기반으로 그리드에 대한 통계를 계산합니다."""
    grid_flat = grid[~np.isnan(grid)]
    if grid_flat.size == 0:
        return []
    hist, _ = np.histogram(grid_flat, bins=bins)
    binned_stats = []
    num_items = min(len(labels), len(hist))
    for i in range(num_items):
        binned_stats.append(
            {"bin_range": labels[i], "area": hist[i]}
        )
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


def get_pixel_size_from_db(engine, subbasin_name, default_size=1.0):
    """데이터베이스에서 주어진 소유역에 대한 픽셀 크기를 가져옵니다."""
    try:
        from sqlalchemy import text
        with engine.connect() as connection:
            query = text(
                "SELECT pixel_size_m FROM subbasin_pixel_size WHERE subbasin = :subbasin")
            result = connection.execute(
                query, {"subbasin": subbasin_name}).scalar_one_or_none()
            return float(result) if result is not None else float(default_size)
    except Exception:
        return float(default_size)


def calc_stats(array):
    arr = array[np.isfinite(array)]
    if arr.size == 0:
        return {'min': 0, 'max': 0, 'mean': 0, 'area': 0}
    return {'min': float(np.nanmin(arr)), 'max': float(np.nanmax(arr)), 'mean': float(np.nanmean(arr)), 'area': arr.size}


def run_full_analysis(user_gdf_original, selected_types, subbasin_name):
    import richdem as rd

    from utils.config import get_db_engine
    from utils.file_processor import clip_geodataframe

    engine = get_db_engine()
    dem_results = {}
    target_crs = "EPSG:5186"

    if 'geometry' not in user_gdf_original.columns:
        raise ValueError("업로드한 파일에 유효한 공간 정보('geometry' 열)가 없습니다.")
    user_gdf_reprojected = user_gdf_original.to_crs(target_crs)
    if user_gdf_reprojected.empty:
        raise ValueError("업로드한 파일에서 유효한 분석 영역을 찾을 수 없습니다.")

    buffer_m = 100.0
    user_bounds = user_gdf_reprojected.total_bounds
    expanded_bounds = (user_bounds[0] - buffer_m, user_bounds[1] -
                       buffer_m, user_bounds[2] + buffer_m, user_bounds[3] + buffer_m)
    bbox_wkt = f"POLYGON(({expanded_bounds[0]} {expanded_bounds[1]}, {expanded_bounds[2]} {expanded_bounds[1]}, {expanded_bounds[2]} {expanded_bounds[3]}, {expanded_bounds[0]} {expanded_bounds[3]}, {expanded_bounds[0]} {expanded_bounds[1]}))"
    sql = f"SELECT geometry, elevation FROM kr_contour_map WHERE ST_Intersects(geometry, ST_GeomFromText('{bbox_wkt}', 5186));"

    dem_needed = any(item in selected_types for item in [
                     'elevation', 'slope', 'aspect'])
    if dem_needed:
        all_xs, all_ys, all_zs = [], [], []
        try:
            contour_iterator = gpd.read_postgis(sql, engine, geom_col='geometry', chunksize=10000)
            is_empty = True
            for contour_chunk in contour_iterator:
                is_empty = False
                contour_chunk = contour_chunk.to_crs(target_crs)
                xs, ys, zs = extract_points_from_geometries(contour_chunk, 'elevation')
                if xs.size > 0:
                    all_xs.append(xs)
                    all_ys.append(ys)
                    all_zs.append(zs)
            if is_empty:
                raise ValueError("표고 분석에 필요한 등고선 데이터를 데이터베이스에서 찾을 수 없습니다.")
        except Exception as e:
            raise ValueError(f"데이터베이스에서 등고선 데이터를 읽는 중 오류 발생: {e}")

        if not all_xs:
            raise ValueError("데이터베이스에서 추출한 등고선 데이터에 유효한 고도 포인트가 없습니다.")

        xs = np.concatenate(all_xs)
        ys = np.concatenate(all_ys)
        zs = np.concatenate(all_zs)
        if xs.size == 0:
            raise ValueError("데이터베이스에서 추출한 등고선 데이터에 유효한 고도 포인트가 없습니다.")

        minx, miny, maxx, maxy = expanded_bounds
        pixel_size = get_pixel_size_from_db(
            engine, subbasin_name, default_size=1.0)
        grid_x, grid_y = np.mgrid[minx:maxx:pixel_size, miny:maxy:pixel_size]
        dem_grid = griddata((xs, ys), zs, (grid_x, grid_y),
                            method='linear', fill_value=np.nan)
        transform = from_origin(
            expanded_bounds[0], expanded_bounds[3], pixel_size, pixel_size)

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmpfile:
            dem_tif_path = tmpfile.name
        with rasterio.open(dem_tif_path, 'w', driver='GTiff', height=dem_grid.shape[1], width=dem_grid.shape[0], count=1, dtype='float32', crs=target_crs, transform=transform, nodata=np.nan) as dst:
            dst.write(np.flipud(dem_grid.T), 1)

        temp_files_to_clean = {'elevation': dem_tif_path}
        if 'slope' in selected_types or 'aspect' in selected_types:
            dem_rd = rd.LoadGDAL(dem_tif_path)
            slope_arr = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')

            if 'slope' in selected_types:
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as stmp:
                    slope_tif_path = stmp.name
                    with rasterio.open(slope_tif_path, 'w', driver='GTiff', height=slope_arr.shape[0], width=slope_arr.shape[1], count=1, dtype='float32', crs=target_crs, transform=transform, nodata=np.nan) as dst:
                        dst.write(slope_arr, 1)
                    temp_files_to_clean['slope'] = slope_tif_path

            if 'aspect' in selected_types:
                aspect_arr = rd.TerrainAttribute(dem_rd, attrib='aspect')
                # 경사가 0인 곳의 경사향을 -1로 설정하여 평평한 지역으로 표시
                aspect_arr[np.isclose(slope_arr, 0)] = -1
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as atmp:
                    aspect_tif_path = atmp.name
                    with rasterio.open(aspect_tif_path, 'w', driver='GTiff', height=aspect_arr.shape[0], width=aspect_arr.shape[1], count=1, dtype='float32', crs=target_crs, transform=transform, nodata=np.nan) as dst:
                        dst.write(aspect_arr, 1)
                    temp_files_to_clean['aspect'] = aspect_tif_path

        clip_geoms = [
            g for g in user_gdf_reprojected.geometry if g.is_valid and not g.is_empty]
        if not clip_geoms:
            raise ValueError("클리핑에 사용할 유효한 폴리곤이 업로드된 파일에 없습니다.")

        for analysis_type in temp_files_to_clean:
            if analysis_type in selected_types or (analysis_type == 'elevation' and dem_needed):
                with rasterio.open(temp_files_to_clean[analysis_type]) as src:
                    clipped_grid, _ = mask(
                        src, clip_geoms, crop=True, nodata=np.nan)
                    clipped_grid = clipped_grid[0]
                    grid_stats = calc_stats(clipped_grid)
                    palette_name, bins, labels = None, [], []

                    if analysis_type == 'elevation' or analysis_type == 'slope':
                        area_threshold = 1_000_000
                        num_divisions = 10 if grid_stats.get(
                            'area', 0) > area_threshold else 8
                        if analysis_type == 'elevation':
                            palette_name = f'elevation_{num_divisions}'
                            bins, labels = generate_custom_intervals(
                                grid_stats['min'], grid_stats['max'], num_divisions)
                        else:  # slope
                            palette_name = f'slope_{num_divisions}'
                            # 경사도 90% 값이 10도 미만이면 완만하다고 판단, 세부 구간 사용
                            if np.nanpercentile(clipped_grid[~np.isnan(clipped_grid)], 90) < 10:
                                bins, labels = generate_detailed_slope_intervals(
                                    num_divisions)
                            else:
                                bins, labels = generate_slope_intervals(
                                    num_divisions)
                    elif analysis_type == 'aspect':
                        palette_name = 'aspect_5'
                        palette_data = get_palette(palette_name)

                        # 'Flat' 라벨을 찾아 앞으로 이동하여 구간 순서와 일치시킴
                        # generate_aspect_bins()의 첫 번째 구간은 'Flat'용입니다.
                        flat_label_item = None
                        for item in palette_data:
                            if item['bin_label'].strip().lower() == 'flat':
                                flat_label_item = item
                                break

                        if flat_label_item:
                            palette_data.remove(flat_label_item)
                            palette_data.insert(0, flat_label_item)

                        labels = [item['bin_label'] for item in palette_data]
                        full_bins = generate_aspect_bins()

                        # 라벨 수만큼의 구간을 사용합니다 (+1은 가장자리용).
                        bins = full_bins[:len(labels) + 1]

                    binned_stats_result = calculate_binned_stats(
                        clipped_grid, bins, labels)

                    # 잘라낸 그리드를 새 임시 TIF 파일에 저장
                    with tempfile.NamedTemporaryFile(suffix=f'_{analysis_type}.tif', delete=False) as tmp_clipped_file:
                        clipped_tif_path = tmp_clipped_file.name

                    # 소스 파일에서 메타데이터를 가져와 새 TIF 작성
                    with rasterio.open(temp_files_to_clean[analysis_type]) as src:
                        profile = src.profile
                        # 잘라낸 그리드에 대한 프로필 업데이트
                        profile.update({
                            'height': clipped_grid.shape[0],
                            'width': clipped_grid.shape[1],
                            'transform': rasterio.windows.transform(src.window(*src.bounds), src.transform)
                        })

                    with rasterio.open(clipped_tif_path, 'w', **profile) as dst:
                        dst.write(clipped_grid, 1)

                    dem_results[analysis_type] = {
                        'grid': clipped_grid,
                        'stats': grid_stats,
                        'binned_stats': binned_stats_result,
                        'bins': bins,
                        'labels': labels,
                        'palette_name': palette_name,
                        'tif_path': clipped_tif_path  # 결과에 경로 추가
                    }
        for path in temp_files_to_clean.values():
            if path and os.path.exists(path):
                os.remove(path)

    # --- 벡터 분석 (토양, HSG, 토지피복) ---
    vector_analysis_types = [
        item for item in selected_types if item in ['soil', 'hsg', 'landcover']]

    if vector_analysis_types:
        analysis_configs = {
            'soil': 'public.kr_soil_map',
            'hsg': 'public.kr_hsg_map',
            'landcover': 'public.kr_landcover_map_l3'
        }

        for analysis_type in vector_analysis_types:
            table_name = analysis_configs.get(analysis_type)
            if not table_name:
                continue

            if analysis_type == 'soil':
                analysis_crs = 'EPSG:5174'
            else:
                analysis_crs = 'EPSG:5186'

            srid = analysis_crs.split(':')[1]

            user_gdf_reprojected = user_gdf_original.to_crs(analysis_crs)
            user_wkt = user_gdf_reprojected.union_all().wkt

            # geometry 컬럼 중복을 피하기 위해 필요한 컬럼만 명시적으로 선택합니다.
            with engine.connect() as connection:
                from sqlalchemy import text
                table_name_only = table_name.split('.')[-1]
                query = text("SELECT column_name FROM information_schema.columns WHERE table_name = :table_name AND column_name != 'geometry'")
                result = connection.execute(query, {"table_name": table_name_only})
                columns = [row[0] for row in result]
                formatted_columns = ', '.join([f't1."{c}"' for c in columns])

            # 최종 SQL 쿼리를 생성합니다.
            sql = f"SELECT ST_Intersection(t1.geometry, ST_GeomFromText('{user_wkt}', {srid})) AS geometry, {formatted_columns} FROM {table_name} AS t1 WHERE ST_Intersects(t1.geometry, ST_GeomFromText('{user_wkt}', {srid}));"

            try:
                # Python에서 클리핑을 수행하지 않으므로, DB 결과를 바로 사용합니다.
                clipped_gdf = gpd.read_postgis(sql, engine, geom_col='geometry')

                if not clipped_gdf.empty:
                    # CRS 정보가 누락될 수 있으므로 다시 설정
                    clipped_gdf.set_crs(analysis_crs, inplace=True)
                    # ST_Intersection 결과로 생성될 수 있는 빈 지오메트리 제거
                    clipped_gdf = clipped_gdf[~clipped_gdf.geometry.is_empty]
                    dem_results[analysis_type] = {'gdf': clipped_gdf}
                else:
                    dem_results[analysis_type] = {'gdf': gpd.GeoDataFrame(crs=analysis_crs)}

            except Exception as e:
                print(f"Error processing {analysis_type}: {e}")
                dem_results[analysis_type] = {'gdf': gpd.GeoDataFrame(crs=analysis_crs)}

    return dem_results