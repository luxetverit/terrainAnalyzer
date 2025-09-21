# -*- coding: utf-8 -*-
import glob
import os
import warnings
from tempfile import NamedTemporaryFile
import tempfile

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from sqlalchemy import create_engine
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')

# ========== 설정 ==========
# DB 연결 설정
try:
    engine = create_engine(
        "postgresql://postgres:asdfasdf12@localhost:5432/gisDB")
    with engine.connect() as connection:
        print("✅ 데이터베이스 연결 성공")
except Exception as e:
    print(f"❌ 데이터베이스 연결 실패: {e}")
    engine = None

input_shp_dir = r"D:\\Python\\TerrainAnalyzer\\0.Input"
base_output_dir = r"D:\\Python\\TerrainAnalyzer\\1.basinstatus"
# pixel_size_csv 는 더 이상 사용하지 않으므로 주석 처리
# pixel_size_csv = os.path.join(base_output_dir, "pixel_size_log.csv")
# pixel_df = pd.read_csv(pixel_size_csv)
os.makedirs(base_output_dir, exist_ok=True)

# 표고 전용 사용자 정의 색상
elevation_colors = {
    8: ['#66CDAA', '#FFFF00', '#008000', '#FFA500', '#8B0000', '#A52A2A', '#808080', '#FFFAFA'],
    10: ['#66CDAA', '#DDF426', '#71B800', '#558C00', '#F29300', '#981200', '#9C1C1C', '#955050', '#9C9B9B', '#FFFAFA']
}

# 기존 경사도/경사향용 컬러맵
colormaps = {
    'slope': ['YlOrRd', 'Reds', 'hot', 'copper', 'autumn'],
    'aspect': ['hsv', 'twilight', 'rainbow', 'gist_rainbow', 'nipy_spectral']
}

aspect_labels = ["North", "Northeast", "East",
                 "Southeast", "South", "Southwest", "West", "Northwest"]
aspect_bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
interval_candidates = [1, 2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200, 500]


def create_hillshade(data, azimuth=315, altitude=45):
    """고품질 음영기복도(Hillshade) 생성"""
    # NaN 값 처리
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return np.ones_like(data)

    # 경사도와 경사향 계산
    dy, dx = np.gradient(data)

    # 경사각 계산 (라디안)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))

    # 경사향 계산 (라디안, 북쪽 기준)
    aspect = np.arctan2(-dx, dy)

    # 태양 각도를 라디안으로 변환
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(altitude)

    # Hillshade 계산
    hillshade = (np.sin(altitude_rad) * np.sin(np.pi/2 - slope) +
                 np.cos(altitude_rad) * np.cos(np.pi/2 - slope) *
                 np.cos(azimuth_rad - aspect))

    # 0~1 범위로 정규화
    hillshade = np.clip(hillshade, 0, 1)

    # 유효하지 않은 영역은 0.5로 설정 (중간 밝기)
    hillshade[~valid_mask] = 0.5

    return hillshade


def create_custom_elevation_colormap(colors, data, divisions):
    """표고용 연속적인 사용자 정의 컬러맵 생성"""
    n_colors = len(colors)
    positions = np.linspace(0, 1, n_colors)

    color_dict = {
        'red': [],
        'green': [],
        'blue': []
    }

    for i, (pos, color) in enumerate(zip(positions, colors)):
        rgb = mcolors.hex2color(color)
        color_dict['red'].append((pos, rgb[0], rgb[0]))
        color_dict['green'].append((pos, rgb[1], rgb[1]))
        color_dict['blue'].append((pos, rgb[2], rgb[2]))

    cmap = mcolors.LinearSegmentedColormap(
        f'custom_elevation_{divisions}', color_dict)
    return cmap


def calculate_accurate_scalebar_params(pixel_size, img_shape, target_size_mm, fig, ax):
    """실제 axes 정보를 사용한 정확한 스케일바 계산"""
    fig_width_inch, fig_height_inch = fig.get_size_inches()
    ax_bbox = ax.get_position()

    img_width_fig = ax_bbox.width
    img_height_fig = ax_bbox.height

    img_height_pixels, img_width_pixels = img_shape

    img_width_inch = img_width_fig * fig_width_inch
    pixels_per_inch = img_width_pixels / img_width_inch

    target_size_inch = target_size_mm / 25.4
    target_pixels = target_size_inch * pixels_per_inch
    real_distance_m = target_pixels * pixel_size

    if real_distance_m < 100:
        scale_distance_m = round(real_distance_m / 50) * 50
        if scale_distance_m == 0:
            scale_distance_m = 50
        unit = 'm'
        scale_value = scale_distance_m
    elif real_distance_m < 1000:
        scale_distance_m = round(real_distance_m / 100) * 100
        unit = 'm'
        scale_value = scale_distance_m
    elif real_distance_m < 5000:
        scale_distance_km = round(real_distance_m / 500) * 0.5
        scale_distance_m = scale_distance_km * 1000
        unit = 'km'
        scale_value = scale_distance_km
    else:
        scale_distance_km = round(real_distance_m / 1000)
        scale_distance_m = scale_distance_km * 1000
        unit = 'km'
        scale_value = scale_distance_km

    actual_scalebar_width_fig = (
        scale_distance_m / pixel_size) / pixels_per_inch / fig_width_inch

    return {
        'length': scale_value,
        'units': unit,
        'segments': 2,
        'target_size_mm': target_size_mm,
        'real_distance_m': scale_distance_m,
        'scalebar_width_fig': actual_scalebar_width_fig
    }


def draw_accurate_scalebar(fig, ax, pixel_size, scale_params, img_shape):
    """정확한 축척 반영 스케일바"""
    total_length = scale_params['length']
    units = scale_params['units']
    segments = scale_params['segments']
    scalebar_width_fig = scale_params['scalebar_width_fig']

    start_x_fig = 0.1
    start_y_fig = 0.02
    bar_height_fig = 0.008

    bg_width_fig = scalebar_width_fig + 0.01
    bg_height_fig = bar_height_fig * 2 + 0.03

    bg_rect = Rectangle((start_x_fig - 0.005, start_y_fig - 0.005),
                        bg_width_fig, bg_height_fig,
                        facecolor='white', edgecolor='none', linewidth=0,
                        alpha=0.9, transform=fig.transFigure)
    fig.patches.append(bg_rect)

    segment_width_fig = scalebar_width_fig / segments
    segment_value = total_length / segments

    for i in range(segments):
        x_fig = start_x_fig + i * segment_width_fig

        color1 = 'black' if i % 2 == 0 else 'white'
        rect1 = Rectangle((x_fig, start_y_fig + bar_height_fig),
                          segment_width_fig, bar_height_fig,
                          facecolor=color1, edgecolor='black', linewidth=0.5,
                          transform=fig.transFigure)
        fig.patches.append(rect1)

        color2 = 'white' if i % 2 == 0 else 'black'
        rect2 = Rectangle((x_fig, start_y_fig),
                          segment_width_fig, bar_height_fig,
                          facecolor=color2, edgecolor='black', linewidth=0.5,
                          transform=fig.transFigure)
        fig.patches.append(rect2)

        text_x_fig = x_fig + segment_width_fig
        text_y_fig = start_y_fig + bar_height_fig * 2 + 0.005

        segment_val = (i + 1) * segment_value

        if units == 'km':
            if segment_val != int(segment_val):
                text_label = f'{segment_val:.1f}'
            else:
                text_label = f'{int(segment_val)}'
        else:
            text_label = f'{int(segment_val)}'

        if i == segments - 1:
            text_label += units

        fig.text(text_x_fig, text_y_fig, text_label,
                 ha='center', va='bottom', fontsize=9, fontweight='bold',
                 color='black', transform=fig.transFigure)

    fig.text(start_x_fig, start_y_fig + bar_height_fig * 2 + 0.005, '0',
             ha='center', va='bottom', fontsize=9, fontweight='bold',
             color='black', transform=fig.transFigure)


def generate_custom_intervals(min_val, max_val, divisions):
    """사용자 정의 구간 생성"""
    diff = max_val - min_val
    div = divisions - 2
    target_interval = diff / div
    sorted_candidates = sorted(
        interval_candidates, key=lambda x: abs(x - target_interval))
    for best_interval in sorted_candidates:
        mid_val = min_val + diff / 2
        mid_aligned = round(mid_val / best_interval) * best_interval
        start = mid_aligned - best_interval * (div // 2)
        intervals = [f"{start}미만"]
        for i in range(div):
            lo = start + i * best_interval
            hi = lo + best_interval
            intervals.append(f"{lo}~{hi}")
        intervals.append(f"{hi}초과")
        return intervals, best_interval, start
    return [], 1, 0


def calculate_area_distribution(tif_path, interval_labels, bins):
    """면적 분포 계산"""
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        data = np.where((data == src.nodata) | np.isnan(data), np.nan, data)
        transform = src.transform
        pixel_area = abs(transform.a * transform.e)
    flat = data.flatten()
    flat = flat[~np.isnan(flat)]
    if len(flat) == 0:
        return [], 0
    hist, _ = np.histogram(flat, bins=bins)
    areas = hist * pixel_area
    total_area = np.sum(areas)
    percentages = areas / total_area * 100
    return list(zip(interval_labels, areas, percentages)), total_area


def extract_points_from_geometries(gdf, elevation_col='elevation'):
    """GDF에서 고도 포인트를 추출하는 헬퍼 함수"""
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


def generate_dem_from_db(shp_path, engine, pixel_size=1.0, buffer_m=100.0):
    """DB 등고선 데이터로 DEM을 생성하고 원본 영역으로 클리핑하는 함수"""
    target_crs = "EPSG:5186"
    try:
        user_gdf = gpd.read_file(shp_path).to_crs(target_crs)
        if user_gdf.empty:
            print("  - 입력 shp 파일이 비어있습니다.")
            return None, None, None

        user_bounds = user_gdf.total_bounds
        expanded_bounds = (
            user_bounds[0] - buffer_m, user_bounds[1] - buffer_m,
            user_bounds[2] + buffer_m, user_bounds[3] + buffer_m,
        )

        bbox_wkt = (
            f"POLYGON(({expanded_bounds[0]} {expanded_bounds[1]}, "
            f"{expanded_bounds[2]} {expanded_bounds[1]}, "
            f"{expanded_bounds[2]} {expanded_bounds[3]}, "
            f"{expanded_bounds[0]} {expanded_bounds[3]}, "
            f"{expanded_bounds[0]} {expanded_bounds[1]}))"
        )
        sql = f"SELECT geometry, elevation FROM kr_contour_map WHERE ST_Intersects(geometry, ST_GeomFromText('{bbox_wkt}', 5186));"
        
        contour_gdf = gpd.read_postgis(sql, engine, geom_col='geometry')
        if contour_gdf.empty:
            print("  - 해당 영역에서 등고선 데이터를 찾을 수 없습니다.")
            return None, None, None

        xs, ys, zs = extract_points_from_geometries(contour_gdf, 'elevation')
        if xs.size == 0:
            print("  - 등고선 데이터에서 유효한 고도 포인트를 추출할 수 없습니다.")
            return None, None, None

        minx, miny, maxx, maxy = expanded_bounds
        grid_x, grid_y = np.mgrid[minx:maxx:pixel_size, miny:maxy:pixel_size]
        
        dem_grid_interpolated = griddata((xs, ys), zs, (grid_x, grid_y), method='linear', fill_value=np.nan)
        dem_grid = np.flipud(dem_grid_interpolated.T)

        transform = from_origin(expanded_bounds[0], expanded_bounds[3], pixel_size, pixel_size)
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmpfile:
            temp_dem_path = tmpfile.name
        
        with rasterio.open(
            temp_dem_path, 'w', driver='GTiff', height=dem_grid.shape[0], width=dem_grid.shape[1],
            count=1, dtype='float32', crs=target_crs, transform=transform, nodata=np.nan
        ) as dst:
            dst.write(dem_grid, 1)

        with rasterio.open(temp_dem_path) as src:
            clip_geoms = [g for g in user_gdf.geometry if g.is_valid and not g.is_empty]
            clipped_grid, clipped_transform = mask(src, clip_geoms, crop=True, nodata=np.nan)
            clipped_grid = clipped_grid[0]
        
        os.remove(temp_dem_path)
        
        return clipped_grid, clipped_transform, target_crs
    except Exception as e:
        print(f"  - DEM 생성 중 오류 발생: {e}")
        return None, None, None



def create_visualization_with_hillshade(data, pixel_size, output_path, cmap_name, data_type='elevation',
                                        size_mm=25, divisions=None, hillshade_intensity=0.5):
    """음영 효과만을 적용한 자연스러운 지형 시각화"""
    fig, ax = plt.subplots(figsize=(10, 9))

    # 1. 음영기복도 생성 (DEM 데이터 기반)
    hillshade = create_hillshade(data, azimuth=315, altitude=45)

    # 2. 컬러맵 설정
    if data_type == 'elevation' and divisions in elevation_colors:
        cmap = create_custom_elevation_colormap(
            elevation_colors[divisions], data, divisions)
    else:
        cmap = plt.cm.get_cmap(cmap_name)

    # 3. 유효한 데이터 마스크 생성
    valid_mask = ~np.isnan(data)

    if np.any(valid_mask):
        # 4. 데이터를 0-1로 정규화
        vmin = np.nanmin(data[valid_mask])
        vmax = np.nanmax(data[valid_mask])

        if vmax > vmin:
            normalized_data = (data - vmin) / (vmax - vmin)
        else:
            normalized_data = np.full_like(data, 0.5)

        # 5. 음영 효과와 색상 데이터 조합 (곱셈 블렌딩)
        # hillshade 값을 0.5~1.5 범위로 조정 (너무 어두워지지 않게)
        hillshade_adjusted = 0.5 + hillshade * hillshade_intensity

        # 색상 데이터에 음영 적용
        shaded_data = normalized_data * hillshade_adjusted
        shaded_data = np.clip(shaded_data, 0, 1)  # 0-1 범위로 제한

        # 유효 범위만 마스킹
        masked_shaded_data = np.ma.masked_where(~valid_mask, shaded_data)

        # 6. 시각화 (원래 데이터 범위로 매핑)
        im = ax.imshow(masked_shaded_data, cmap=cmap, interpolation='bilinear',
                       vmin=0, vmax=1)

    ax.axis('off')
    ax.set_position([0.05, 0.15, 0.9, 0.8])

    # 7. 스케일바 추가
    scale_params = calculate_accurate_scalebar_params(
        pixel_size, data.shape, size_mm, fig, ax)
    draw_accurate_scalebar(fig, ax, pixel_size, scale_params, data.shape)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1,
                facecolor='white', edgecolor='none')
    plt.close()


def create_simple_visualization(data, pixel_size, output_path, cmap_name, data_type='elevation',
                                size_mm=25, divisions=None):
    """기존 방식의 단순 시각화 (비교용) - 분석 범위만"""
    fig, ax = plt.subplots(figsize=(10, 9))

    # 컬러맵 설정
    if data_type == 'elevation' and divisions in elevation_colors:
        cmap = create_custom_elevation_colormap(
            elevation_colors[divisions], data, divisions)
    else:
        cmap = plt.cm.get_cmap(cmap_name)

    # 유효한 데이터만 표시
    valid_mask = ~np.isnan(data)
    masked_data = np.ma.masked_where(~valid_mask, data)

    # 시각화
    im = ax.imshow(masked_data, cmap=cmap, interpolation='bilinear')
    ax.axis('off')
    ax.set_position([0.05, 0.15, 0.9, 0.8])

    # 스케일바 추가
    scale_params = calculate_accurate_scalebar_params(
        pixel_size, data.shape, size_mm, fig, ax)
    draw_accurate_scalebar(fig, ax, pixel_size, scale_params, data.shape)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1,
                facecolor='white', edgecolor='none')
    plt.close()


# ========== 실행 ==========
all_elev_stats = {10: [], 8: []}
all_slope_stats = {10: [], 8: []}
all_aspect_stats = []

shp_files = glob.glob(os.path.join(input_shp_dir, "*.shp"))
for shp_path in shp_files:
    name = os.path.splitext(os.path.basename(shp_path))[0]
    sub_folder = os.path.join(base_output_dir, name)
    os.makedirs(sub_folder, exist_ok=True)

    print(f"\nProcessing {name}...")
    if engine is None:
        print("DB 연결이 없어 DEM 생성을 건너뜁니다.")
        continue

    # 1. DB에서 DEM 생성
    print("  1/3: DB에서 등고선 데이터를 조회하여 DEM을 생성합니다...")
    pixel_size = 1.0  # 1m 픽셀 사이즈로 고정
    dem_data, dem_transform, dem_crs = generate_dem_from_db(shp_path, engine, pixel_size=pixel_size)

    if dem_data is None:
        print(f"❌ {name}에 대한 DEM 생성 실패.")
        continue
    print("  ✅ DEM 생성 완료.")

    dem_path = os.path.join(sub_folder, f"{name}_elevation.tif")
    with rasterio.open(
        dem_path, 'w', driver='GTiff', height=dem_data.shape[0], width=dem_data.shape[1],
        count=1, dtype='float32', crs=dem_crs, transform=dem_transform, nodata=np.nan
    ) as dst:
        dst.write(dem_data, 1)

    # 2. 생성된 DEM으로 경사/경사향 분석
    print("  2/3: 생성된 DEM으로 경사도 및 경사향을 분석합니다...")
    slope_path, aspect_path = None, None
    try:
        import richdem as rd
        dem_rd = rd.LoadGDAL(dem_path)
        
        slope_arr = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')
        slope_path = os.path.join(sub_folder, f"{name}_slope.tif")
        with rasterio.open(slope_path, 'w', driver='GTiff', height=slope_arr.shape[0], width=slope_arr.shape[1], count=1, dtype='float32', crs=dem_crs, transform=dem_transform, nodata=np.nan) as dst:
            dst.write(slope_arr, 1)

        aspect_arr = rd.TerrainAttribute(dem_rd, attrib='aspect')
        aspect_path = os.path.join(sub_folder, f"{name}_aspect.tif")
        with rasterio.open(aspect_path, 'w', driver='GTiff', height=aspect_arr.shape[0], width=aspect_arr.shape[1], count=1, dtype='float32', crs=dem_crs, transform=dem_transform, nodata=np.nan) as dst:
            dst.write(aspect_arr, 1)
        print("  ✅ 경사도/경사향 분석 완료.")
    except ImportError:
        print("  ⚠️ 'richdem' 라이브러리가 설치되지 않아 경사/경사향 분석을 건너뜁니다.")
    except Exception as e:
        print(f"  ❌ 경사도/경사향 분석 실패: {e}")

    # 3. 통계 계산 및 시각화
    print("  3/3: 통계 계산 및 시각화를 진행합니다...")
    gdf = gpd.read_file(shp_path).to_crs("EPSG:5186")
    subbasin_area = gdf.geometry.area.sum()
    area_km2 = subbasin_area / 1e6


    # 표고 및 경사도 처리
    for kind, tif_path, stat_collector in [("elevation", dem_path, all_elev_stats),
                                           ("slope", slope_path, all_slope_stats)]:
        if not os.path.exists(tif_path):
            continue

        with rasterio.open(tif_path) as src:
            data = src.read(1)
            data = np.where((data == src.nodata) |
                            np.isnan(data), np.nan, data)

        flat = data.flatten()
        flat = flat[~np.isnan(flat)]
        if len(flat) == 0:
            continue

        min_val = np.min(flat)
        max_val = np.max(flat)

        # 구간별 통계 계산
        for divisions in [10, 8]:
            if kind == "slope":
                if divisions == 10:
                    edge_vals = list(range(0, 50, 5))
                    labels = [
                        "0미만"] + [f"{edge_vals[i]}~{edge_vals[i+1]}" for i in range(len(edge_vals)-1)] + ["45초과"]
                    bins = [float("-inf")] + edge_vals + [float("inf")]
                else:
                    edge_vals = list(range(0, 40, 5))
                    labels = [
                        "0미만"] + [f"{edge_vals[i]}~{edge_vals[i+1]}" for i in range(len(edge_vals)-1)] + ["35초과"]
                    bins = [float("-inf")] + edge_vals + [float("inf")]
            else:
                labels, interval_width, start = generate_custom_intervals(
                    min_val, max_val, divisions)
                bins = [start + i *
                        interval_width for i in range(divisions - 1)]
                bins = [float('-inf')] + bins + [float('inf')]

                # 0미만 구간 조정
                if labels[0].endswith("미만") and "0" in labels[0]:
                    stats, _ = calculate_area_distribution(
                        tif_path, labels, bins)
                    if stats and stats[0][1] == 0:
                        smaller_candidates = [
                            i for i in interval_candidates if i < interval_width]
                        if smaller_candidates:
                            new_interval = max(smaller_candidates)
                            mid_val = min_val + (max_val - min_val) / 2
                            mid_aligned = round(
                                mid_val / new_interval) * new_interval
                            start = mid_aligned - new_interval * \
                                ((divisions - 2) // 2)
                            labels = [f"{start}미만"] + [f"{start + i * new_interval}~{start + (i + 1) * new_interval}" for i in range(
                                divisions - 2)] + [f"{start + (divisions - 2) * new_interval}초과"]
                            bins = [start + i *
                                    new_interval for i in range(divisions - 1)]
                            bins = [float('-inf')] + bins + [float('inf')]

            # 통계 계산 및 저장
            stats, _ = calculate_area_distribution(tif_path, labels, bins)
            stats_df = pd.DataFrame(stats, columns=["구간", "면적", "백분율"])
            stats_df.insert(0, "대상", name)
            stats_df["보정면적"] = stats_df["백분율"] / 100 * subbasin_area
            stats_df["보정백분율"] = stats_df["보정면적"] / subbasin_area * 100
            stats_df = stats_df.round(
                {"면적": 2, "백분율": 2, "보정면적": 2, "보정백분율": 2})
            stat_collector[divisions].append(stats_df)

            # 시각화 생성
            preview_dir = os.path.join(sub_folder, "visualizations", kind)
            os.makedirs(preview_dir, exist_ok=True)

            if kind == 'elevation':
                # 표고는 사용자 정의 색상만 사용
                for size_mm in [25, 50]:
                    # 음영 효과 적용 버전
                    output_path = os.path.join(preview_dir,
                                               f"{name}_{kind}_{divisions}div_hillshade_{size_mm}mm.png")
                    create_visualization_with_hillshade(data, pixel_size, output_path, None,
                                                        kind, size_mm, divisions, hillshade_intensity=0.7)

                    # 기존 단순 버전 (비교용)
                    output_path_simple = os.path.join(preview_dir,
                                                      f"{name}_{kind}_{divisions}div_simple_{size_mm}mm.png")
                    create_simple_visualization(data, pixel_size, output_path_simple, None,
                                                kind, size_mm, divisions)

            else:
                # 경사도는 기존 컬러맵 사용 + DEM 기반 음영기복도
                for cmap in colormaps[kind]:
                    for size_mm in [25, 50]:
                        # 음영기복도 오버랩 버전 (DEM 데이터 사용)
                        if dem_data is not None:
                            output_path = os.path.join(preview_dir,
                                                       f"{name}_{kind}_{divisions}div_{cmap}_hillshade_{size_mm}mm.png")
                            # 경사도 데이터 + DEM 기반 음영기복도 조합
                            fig, ax = plt.subplots(figsize=(10, 9))

                            # DEM 기반 음영기복도
                            hillshade = create_hillshade(
                                dem_data, azimuth=315, altitude=45)
                            cmap_obj = plt.cm.get_cmap(cmap)

                            # 경사도 데이터 시각화
                            valid_mask = ~np.isnan(data)
                            if np.any(valid_mask):
                                vmin = np.nanmin(data[valid_mask])
                                vmax = np.nanmax(data[valid_mask])

                                # 유효 범위만 표시
                                masked_data = np.ma.masked_where(
                                    ~valid_mask, data)
                                masked_hillshade = np.ma.masked_where(
                                    ~valid_mask, hillshade)

                                im1 = ax.imshow(masked_data, cmap=cmap_obj, interpolation='bilinear',
                                                vmin=vmin, vmax=vmax, alpha=1.0)
                                im2 = ax.imshow(masked_hillshade, cmap=cmap_obj, interpolation='bilinear',
                                                vmin=0, vmax=1, alpha=0.6)

                            ax.axis('off')
                            ax.set_position([0.05, 0.15, 0.9, 0.8])

                            scale_params = calculate_accurate_scalebar_params(
                                pixel_size, data.shape, size_mm, fig, ax)
                            draw_accurate_scalebar(
                                fig, ax, pixel_size, scale_params, data.shape)

                            plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1,
                                        facecolor='white', edgecolor='none')
                            plt.close()

                        # 기존 단순 버전 (비교용)
                        output_path_simple = os.path.join(preview_dir,
                                                          f"{name}_{kind}_{divisions}div_{cmap}_simple_{size_mm}mm.png")
                        create_simple_visualization(data, pixel_size, output_path_simple, cmap,
                                                    kind, size_mm, divisions)

    # 경사향 처리
    if os.path.exists(aspect_path):
        with rasterio.open(aspect_path) as src:
            data = src.read(1)
            data = np.where((data == src.nodata) |
                            np.isnan(data), np.nan, data)

        # 북향 처리
        north_mask = ((data >= 0) & (data < 22.5)) | (data >= 337.5)
        data_adjusted = np.where(north_mask, 0.0, data)

        with NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
            with rasterio.open(aspect_path) as src:
                profile = src.profile
                profile.update(dtype=rasterio.float32, nodata=np.nan)
            with rasterio.open(tmpfile.name, 'w', **profile) as dst:
                dst.write(data_adjusted.astype(np.float32), 1)

        # 통계 계산
        stats, _ = calculate_area_distribution(
            tmpfile.name, aspect_labels, aspect_bins)
        os.remove(tmpfile.name)

        stats_df = pd.DataFrame(stats, columns=["방향", "면적", "백분율"])
        stats_df.insert(0, "대상", name)
        stats_df["보정면적"] = stats_df["백분율"] / 100 * subbasin_area
        stats_df["보정백분율"] = stats_df["보정면적"] / subbasin_area * 100
        stats_df = stats_df.round({"면적": 2, "백분율": 2, "보정면적": 2, "보정백분율": 2})
        all_aspect_stats.append(stats_df)

        # 시각화
        preview_dir = os.path.join(sub_folder, "visualizations", "aspect")
        os.makedirs(preview_dir, exist_ok=True)

        for cmap in colormaps['aspect']:
            for size_mm in [25, 50]:
                # 단순 버전만
                output_path = os.path.join(preview_dir,
                                           f"{name}_aspect_{cmap}_{size_mm}mm.png")
                create_simple_visualization(data, pixel_size, output_path, cmap,
                                            'aspect', size_mm)

# ========== 결과 저장 ==========
excel_path = os.path.join(base_output_dir, "구간별_통계결과_hillshade.xlsx")
has_data = any(all_elev_stats[div] for div in [10, 8]) or any(
    all_slope_stats[div] for div in [10, 8]) or len(all_aspect_stats) > 0

if has_data:
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for div in [10, 8]:
            if all_elev_stats[div]:
                pd.concat(all_elev_stats[div], ignore_index=True).to_excel(
                    writer, sheet_name=f"elevation_{div}구간", index=False)
            if all_slope_stats[div]:
                pd.concat(all_slope_stats[div], ignore_index=True).to_excel(
                    writer, sheet_name=f"slope_{div}구간", index=False)
        if all_aspect_stats:
            pd.concat(all_aspect_stats, ignore_index=True).to_excel(
                writer, sheet_name="aspect_방위", index=False)

    print(f"✅ 지형 분석 시각화 완료! Excel 결과: {excel_path}")
    print(f"🏔️  표고: 음영 효과 적용 (hillshade)")
    print(f"📊 경사도/경사향: 기본 시각화")
    print(f"📁 결과 위치: 'visualizations' 폴더")
else:
    print("저장할 통계 결과가 없습니다.")
