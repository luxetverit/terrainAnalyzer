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
from utils.plot_helpers import generate_aspect_bins  # Note: aspect bins only
from utils.plot_helpers import (generate_custom_intervals,
                                generate_slope_intervals)


def calculate_binned_stats(grid, bins, labels):
    """Calculates statistics for a grid based on predefined bins and labels."""
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
    """Fetches pixel size for a given sub-basin from the database."""
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
    contour_gdf = gpd.read_postgis(sql, engine, geom_col='geometry')

    dem_needed = any(item in selected_types for item in [
                     'elevation', 'slope', 'aspect'])
    if dem_needed:
        if contour_gdf.empty:
            raise ValueError("표고 분석에 필요한 등고선 데이터를 데이터베이스에서 찾을 수 없습니다.")
        contour_gdf = contour_gdf.to_crs(target_crs)
        xs, ys, zs = extract_points_from_geometries(contour_gdf, 'elevation')
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
                # Set aspect to -1 where slope is 0 to mark flat areas
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
                            bins, labels = generate_slope_intervals(
                                num_divisions)
                    elif analysis_type == 'aspect':
                        palette_name = 'aspect_5'
                        palette_data = get_palette(palette_name)

                        # Find and move the 'Flat' label to the front to match the bin order
                        # The first bin from generate_aspect_bins() is for 'Flat'.
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

                        # Use as many bins as there are labels (+1 for the edges).
                        bins = full_bins[:len(labels) + 1]

                        # Augment labels with degree ranges for clarity
                        new_labels = []
                        for i, label in enumerate(labels):
                            # The first label is 'Flat'
                            if i == 0 and label.strip().lower() == 'flat':
                                new_labels.append(
                                    f"{label} (-1)")
                                continue

                            lower_bound = bins[i]
                            upper_bound = bins[i+1]

                            # For North, show the full conceptual range
                            if label.strip().lower() == 'north':
                                new_labels.append(
                                    f"{label} (0°~22.5°, 337.5°~360°)")
                            else:
                                new_labels.append(
                                    f"{label} ({lower_bound}°~{upper_bound}°)")

                        labels = new_labels

                    binned_stats_result = calculate_binned_stats(
                        clipped_grid, bins, labels)
                    dem_results[analysis_type] = {
                        'grid': clipped_grid, 'stats': grid_stats, 'binned_stats': binned_stats_result,
                        'bins': bins, 'labels': labels, 'palette_name': palette_name
                    }
        for path in temp_files_to_clean.values():
            if path and os.path.exists(path):
                os.remove(path)

    if any(item in selected_types for item in ['soil', 'hsg', 'landcover']):
        user_geom = user_gdf_reprojected.union_all()
        user_wkt = user_geom.wkt
        if 'soil' in selected_types:
            sql_soil = f"SELECT ST_Intersection(t1.geometry, ST_GeomFromText('{user_wkt}', 5186)) AS geom, t1.* FROM public.kr_soil_map AS t1 WHERE ST_Intersects(t1.geometry, ST_GeomFromText('{user_wkt}', 5186));"
            dem_results['soil'] = {'gdf': gpd.read_postgis(
                sql_soil, engine, geom_col='geom')}
        if 'hsg' in selected_types:
            sql_hsg = f"SELECT ST_Intersection(t1.geometry, ST_GeomFromText('{user_wkt}', 5186)) AS geom, t1.* FROM public.kr_hsg_map AS t1 WHERE ST_Intersects(t1.geometry, ST_GeomFromText('{user_wkt}', 5186));"
            dem_results['hsg'] = {'gdf': gpd.read_postgis(
                sql_hsg, engine, geom_col='geom')}
        if 'landcover' in selected_types:
            sql_landcover = f"SELECT ST_Intersection(t1.geometry, ST_GeomFromText('{user_wkt}', 5186)) AS geom, t1.* FROM public.kr_landcover_map AS t1 WHERE ST_Intersects(t1.geometry, ST_GeomFromText('{user_wkt}', 5186));"
            dem_results['landcover'] = {'gdf': gpd.read_postgis(
                sql_landcover, engine, geom_col='geom')}
    return dem_results
