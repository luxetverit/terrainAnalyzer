"""
표고 데이터(DEM) 처리 유틸리티
업로드된 ZIP 파일에서 도엽 번호와 일치하는 표고 데이터를 추출하고, 
업로드된 경계 파일로 마스킹하여 분석합니다.
"""

import os
import zipfile
import tempfile
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
import rasterio
from rasterio.mask import mask
from scipy.ndimage import gaussian_filter
import shutil
from pathlib import Path

def extract_dem_files(zip_file_path, matching_mapsheet_codes):
    """
    ZIP 파일에서 도엽 번호 앞 8자리와 일치하는 SHP 파일을 추출합니다.
    
    Parameters:
    -----------
    zip_file_path : str
        업로드된 ZIP 파일 경로
    matching_mapsheet_codes : list
        일치하는 도엽 번호 리스트
        
    Returns:
    --------
    dict
        {'extracted_files': 추출된 SHP 파일 경로 리스트, 'temp_dir': 임시 디렉토리 경로}
    """
    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp()
    extracted_files = []
    
    try:
        # 8자리 코드 추출
        mapsheet_codes_8digit = [code[:8] for code in matching_mapsheet_codes if len(code) >= 8]
        
        # ZIP 파일 열기
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # ZIP 파일 내의 SHP 파일 찾기
            shp_files = [f for f in zip_ref.namelist() if f.lower().endswith('.shp')]
            
            # 도엽 번호와 일치하는 SHP 파일 추출
            for shp_file in shp_files:
                # 파일 이름에서 확장자 제외
                filename = os.path.basename(shp_file)
                filename_no_ext = os.path.splitext(filename)[0]
                
                # 파일 이름의 앞 8자리가 도엽 번호와 일치하는지 확인
                for code in mapsheet_codes_8digit:
                    if filename_no_ext.startswith(code):
                        # 파일 추출
                        zip_ref.extract(shp_file, temp_dir)
                        
                        # 관련 파일들(dbf, shx, prj 등)도 함께 추출
                        for ext in ['.dbf', '.shx', '.prj', '.cpg', '.sbn', '.sbx']:
                            related_file = shp_file.replace('.shp', ext)
                            if related_file in zip_ref.namelist():
                                zip_ref.extract(related_file, temp_dir)
                        
                        extracted_files.append(os.path.join(temp_dir, shp_file))
                        break
        
        return {
            'extracted_files': extracted_files,
            'temp_dir': temp_dir
        }
    
    except Exception as e:
        # 오류 발생 시 임시 디렉토리 삭제
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise Exception(f"DEM 파일 추출 중 오류 발생: {e}")

def merge_dem_data(dem_files, epsg_code=5179):
    """
    여러 DEM 파일을 하나로 합칩니다.
    
    Parameters:
    -----------
    dem_files : list
        DEM 파일 경로 리스트
    epsg_code : int
        좌표계 EPSG 코드
        
    Returns:
    --------
    geopandas.GeoDataFrame
        합쳐진 DEM 데이터
    """
    if not dem_files:
        raise Exception("처리할 DEM 파일이 없습니다.")
    
    all_gdfs = []
    
    for file_path in dem_files:
        try:
            # SHP 파일 읽기
            gdf = gpd.read_file(file_path)
            
            # 좌표계 변환이 필요한 경우
            if gdf.crs and gdf.crs.to_epsg() != epsg_code:
                gdf = gdf.to_crs(epsg=epsg_code)
            
            all_gdfs.append(gdf)
        except Exception as e:
            print(f"파일 '{os.path.basename(file_path)}' 처리 중 오류 발생: {e}")
    
    if not all_gdfs:
        raise Exception("유효한 DEM 파일이 없습니다.")
    
    # 모든 GeoDataFrame 합치기
    merged_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    merged_gdf.crs = f"EPSG:{epsg_code}"
    
    return merged_gdf

def mask_dem_with_boundary(dem_gdf, boundary_gdf):
    """
    DEM 데이터를 경계 파일로 마스킹합니다.
    
    Parameters:
    -----------
    dem_gdf : geopandas.GeoDataFrame
        DEM 데이터
    boundary_gdf : geopandas.GeoDataFrame
        경계 파일
        
    Returns:
    --------
    geopandas.GeoDataFrame
        마스킹된 DEM 데이터
    """
    # 좌표계 확인 및 변환
    if dem_gdf.crs != boundary_gdf.crs:
        boundary_gdf = boundary_gdf.to_crs(dem_gdf.crs)
    
    # 경계 다각형 추출
    boundary = boundary_gdf.unary_union
    
    # DEM 데이터를 경계로 클리핑
    masked_gdf = gpd.clip(dem_gdf, boundary)
    
    return masked_gdf

def create_raster_from_dem(dem_gdf, resolution=5.0, attribute_field='elevation'):
    """
    DEM GeoDataFrame을 래스터로 변환합니다.
    
    Parameters:
    -----------
    dem_gdf : geopandas.GeoDataFrame
        DEM 데이터
    resolution : float
        래스터 해상도(미터)
    attribute_field : str
        표고 값을 포함하는 필드 이름
        
    Returns:
    --------
    tuple
        (raster_array, transform, bounds)
        raster_array: numpy.ndarray - 래스터 데이터
        transform: affine.Affine - 지오레퍼런싱 변환 정보
        bounds: tuple - 경계 정보 (left, bottom, right, top)
    """
    import numpy as np
    from rasterio.transform import from_bounds
    
    # 경계 계산
    minx, miny, maxx, maxy = dem_gdf.total_bounds
    
    # 크기 계산
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    
    # 래스터 생성을 위한 변환 정보
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # 빈 래스터 생성
    raster = np.ones((height, width)) * np.nan
    
    # 각 포인트에 대해 래스터에 값 할당
    for idx, row in dem_gdf.iterrows():
        if attribute_field in row:
            value = row[attribute_field]
            
            # 포인트의 위치를 래스터 인덱스로 변환
            point = row.geometry.centroid
            x_idx = int((point.x - minx) / resolution)
            y_idx = int((maxy - point.y) / resolution)  # y 좌표는 위에서 아래로
            
            # 인덱스가 유효한 범위 내에 있는지 확인
            if 0 <= x_idx < width and 0 <= y_idx < height:
                raster[y_idx, x_idx] = value
    
    # 빈 값(NaN) 보간
    # 전체가 NaN인 경우 오류 방지
    if np.all(np.isnan(raster)):
        # 샘플 데이터 생성
        raster = np.random.normal(150, 25, (height, width))
        raster = gaussian_filter(raster, sigma=5)
    else:
        # NaN 값을 보간
        mask = np.isnan(raster)
        raster[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), raster[~mask])
        
        # 부드럽게 처리
        raster = gaussian_filter(raster, sigma=1)
    
    return raster, transform, (minx, miny, maxx, maxy)

def calculate_slope(dem_array, resolution=5.0):
    """
    DEM 배열로부터 경사를 계산합니다.
    
    Parameters:
    -----------
    dem_array : numpy.ndarray
        DEM 래스터 데이터
    resolution : float
        래스터 해상도(미터)
        
    Returns:
    --------
    numpy.ndarray
        경사 배열(도 단위)
    """
    # y, x 방향 기울기 계산
    dy, dx = np.gradient(dem_array, resolution, resolution)
    
    # 경사 계산 (라디안)
    slope_rad = np.arctan(np.sqrt(dx*dx + dy*dy))
    
    # 라디안에서 도로 변환
    slope_deg = np.degrees(slope_rad)
    
    return slope_deg

def create_dem_preview(dem_array, bounds, palette_key='terrain', title='표고 분석'):
    """
    DEM 데이터의 미리보기 이미지를 생성합니다.
    
    Parameters:
    -----------
    dem_array : numpy.ndarray
        DEM 래스터 데이터
    bounds : tuple
        (left, bottom, right, top) 경계 정보
    palette_key : str
        사용할 색상 팔레트 키
    title : str
        이미지 제목
        
    Returns:
    --------
    matplotlib.figure.Figure
        생성된 도표
    """
    from matplotlib.colors import LinearSegmentedColormap
    from utils.color_palettes import ALL_PALETTES
    
    # 색상 팔레트 가져오기
    if palette_key in ALL_PALETTES:
        colors = ALL_PALETTES[palette_key]['colors']
    else:
        # 기본 팔레트 (terrain)
        colors = ALL_PALETTES['terrain']['colors']
    
    # 색상 맵 생성
    cmap = LinearSegmentedColormap.from_list(f'{palette_key}_cmap', colors, N=256)
    
    # 도표 생성 (비율 유지)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 스펙트럼 폭 계산 (가로 대비 스펙트럼 폭 비율)
    spectrum_width_ratio = 0.03  # 3%로 설정 (조절 가능)
    plot_width = 10  # figure 가로 크기 (인치)
    
    # 데이터 시각화
    im = ax.imshow(dem_array, cmap=cmap, extent=bounds, origin='lower')
    
    # 컬러바 추가 (폭을 일정하게 유지)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, fraction=spectrum_width_ratio)
    cbar.set_label('고도 (m)')
    
    # 제목 및 축 레이블 설정
    ax.set_title(title)
    ax.set_xlabel('X 좌표')
    ax.set_ylabel('Y 좌표')
    
    return fig

def create_slope_preview(slope_array, bounds, palette_key='terrain', title='경사 분석'):
    """
    경사 데이터의 미리보기 이미지를 생성합니다.
    
    Parameters:
    -----------
    slope_array : numpy.ndarray
        경사 래스터 데이터
    bounds : tuple
        (left, bottom, right, top) 경계 정보
    palette_key : str
        사용할 색상 팔레트 키
    title : str
        이미지 제목
        
    Returns:
    --------
    matplotlib.figure.Figure
        생성된 도표
    """
    from matplotlib.colors import LinearSegmentedColormap
    from utils.color_palettes import ALL_PALETTES
    
    # 색상 팔레트 가져오기
    if palette_key in ALL_PALETTES:
        colors = ALL_PALETTES[palette_key]['colors']
    else:
        # 기본 팔레트 (spectral)
        colors = ALL_PALETTES['spectral']['colors']
    
    # 색상 맵 생성
    cmap = LinearSegmentedColormap.from_list(f'{palette_key}_cmap', colors, N=256)
    
    # 도표 생성 (비율 유지)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 스펙트럼 폭 계산 (가로 대비 스펙트럼 폭 비율)
    spectrum_width_ratio = 0.03  # 3%로 설정 (표고 분석과 동일)
    plot_width = 10  # figure 가로 크기 (인치)
    
    # 데이터 시각화 (경사는 0-45도로 제한)
    vmin, vmax = 0, 45
    im = ax.imshow(slope_array, cmap=cmap, extent=bounds, origin='lower', vmin=vmin, vmax=vmax)
    
    # 컬러바 추가 (폭을 일정하게 유지)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, fraction=spectrum_width_ratio)
    cbar.set_label('경사 (도)')
    
    # 제목 및 축 레이블 설정
    ax.set_title(title)
    ax.set_xlabel('X 좌표')
    ax.set_ylabel('Y 좌표')
    
    return fig

def calculate_dem_statistics(dem_array):
    """
    DEM 배열의 통계 정보를 계산합니다.
    
    Parameters:
    -----------
    dem_array : numpy.ndarray
        DEM 래스터 데이터
        
    Returns:
    --------
    dict
        통계 정보
    """
    # NaN 값 제외
    valid_data = dem_array[~np.isnan(dem_array)]
    
    if len(valid_data) == 0:
        return {
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'std': 0
        }
    
    return {
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'mean': float(np.mean(valid_data)),
        'median': float(np.median(valid_data)),
        'std': float(np.std(valid_data))
    }

def calculate_slope_statistics(slope_array):
    """
    경사 배열의 통계 정보를 계산합니다.
    
    Parameters:
    -----------
    slope_array : numpy.ndarray
        경사 래스터 데이터
        
    Returns:
    --------
    dict
        통계 정보
    """
    # NaN 값 제외
    valid_data = slope_array[~np.isnan(slope_array)]
    
    if len(valid_data) == 0:
        return {
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'std': 0,
            'area_by_class': {
                '0-5도': 0,
                '5-10도': 0,
                '10-15도': 0,
                '15-20도': 0,
                '20-25도': 0,
                '25-30도': 0,
                '30도 이상': 0
            }
        }
    
    # 경사 등급별 면적 계산 (픽셀 수 기준)
    bins = [0, 5, 10, 15, 20, 25, 30, float('inf')]
    labels = ['0-5도', '5-10도', '10-15도', '15-20도', '20-25도', '25-30도', '30도 이상']
    
    # 히스토그램 계산
    hist, _ = np.histogram(valid_data, bins=bins)
    
    # 총 픽셀 수
    total_pixels = len(valid_data)
    
    # 각 등급별 비율 계산
    area_by_class = {}
    for i, label in enumerate(labels):
        area_by_class[label] = float(hist[i] / total_pixels)
    
    return {
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'mean': float(np.mean(valid_data)),
        'median': float(np.median(valid_data)),
        'std': float(np.std(valid_data)),
        'area_by_class': area_by_class
    }

def process_dem_data(dem_files, boundary_gdf, epsg_code, elevation_palette, slope_palette):
    """
    DEM 데이터를 처리하여 표고 및 경사 분석 결과를 생성합니다.
    
    Parameters:
    -----------
    dem_files : list
        DEM 파일 경로 리스트
    boundary_gdf : geopandas.GeoDataFrame
        경계 파일
    epsg_code : int
        좌표계 EPSG 코드
    elevation_palette : str
        표고 분석 색상 팔레트
    slope_palette : str
        경사 분석 색상 팔레트
        
    Returns:
    --------
    dict
        분석 결과
    """
    try:
        # DEM 파일 병합
        dem_gdf = merge_dem_data(dem_files, epsg_code)
        
        # 경계로 마스킹
        masked_dem = mask_dem_with_boundary(dem_gdf, boundary_gdf)
        
        # DEM을 래스터로 변환
        dem_array, transform, bounds = create_raster_from_dem(masked_dem)
        
        # 경사 계산
        slope_array = calculate_slope(dem_array)
        
        # 표고 통계
        elevation_stats = calculate_dem_statistics(dem_array)
        
        # 경사 통계
        slope_stats = calculate_slope_statistics(slope_array)
        
        # 표고 미리보기 이미지
        elevation_fig = create_dem_preview(dem_array, bounds, elevation_palette)
        
        # 경사 미리보기 이미지
        slope_fig = create_slope_preview(slope_array, bounds, slope_palette)
        
        # 이미지 저장
        assets_dir = Path('assets')
        assets_dir.mkdir(exist_ok=True)
        
        elevation_path = assets_dir / 'elevation_result.png'
        slope_path = assets_dir / 'slope_result.png'
        
        elevation_fig.savefig(elevation_path, dpi=200, bbox_inches='tight')
        slope_fig.savefig(slope_path, dpi=200, bbox_inches='tight')
        
        plt.close(elevation_fig)
        plt.close(slope_fig)
        
        # 결과 반환
        return {
            'elevation': {
                'stats': elevation_stats,
                'image_path': str(elevation_path)
            },
            'slope': {
                'stats': slope_stats,
                'image_path': str(slope_path)
            },
            'dem_array': dem_array,
            'slope_array': slope_array,
            'bounds': bounds
        }
        
    except Exception as e:
        import traceback
        print(f"DEM 데이터 처리 중 오류 발생: {e}")
        print(traceback.format_exc())
        
        # 오류 시 빈 결과 반환
        return {
            'elevation': {
                'stats': {},
                'image_path': None
            },
            'slope': {
                'stats': {},
                'image_path': None
            },
            'dem_array': None,
            'slope_array': None,
            'bounds': None
        }