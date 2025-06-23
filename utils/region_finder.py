import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# 한국 지역 정보 - 좌표 기준 EPSG:5186 (추정 중앙 좌표로 작성)
# 각 지역의 대략적인 좌표 범위를 정의 (x: 동서, y: 남북)
regions = {
    "서울": {
        "center": (199000, 452000),
        "radius": 15000
    },
    "인천": {
        "center": (170000, 447000),
        "radius": 15000
    },
    "경기도 북부": {
        "center": (205000, 480000),
        "radius": 25000
    },
    "경기도 남부": {
        "center": (210000, 425000),
        "radius": 25000
    },
    "강원도 영서": {
        "center": (270000, 490000),
        "radius": 30000
    },
    "강원도 영동": {
        "center": (335000, 475000),
        "radius": 30000
    },
    "충청북도": {
        "center": (250000, 400000),
        "radius": 30000
    },
    "충청남도": {
        "center": (190000, 380000),
        "radius": 30000
    },
    "대전": {
        "center": (215000, 360000),
        "radius": 15000
    },
    "세종": {
        "center": (205000, 370000),
        "radius": 10000
    },
    "전라북도": {
        "center": (200000, 330000),
        "radius": 30000
    },
    "전라남도": {
        "center": (190000, 280000),
        "radius": 30000
    },
    "광주": {
        "center": (195000, 310000),
        "radius": 15000
    },
    "경상북도 내륙": {
        "center": (290000, 360000),
        "radius": 35000
    },
    "경상북도 해안": {
        "center": (330000, 340000),
        "radius": 25000
    },
    "대구": {
        "center": (280000, 320000),
        "radius": 15000
    },
    "경상남도": {
        "center": (270000, 290000),
        "radius": 30000
    },
    "부산": {
        "center": (300000, 270000),
        "radius": 15000
    },
    "울산": {
        "center": (325000, 295000),
        "radius": 15000
    },
    "제주도": {
        "center": (155000, 170000),
        "radius": 20000
    }
}

def find_region_by_coords(x, y):
    """
    주어진 좌표 (EPSG:5186)에 해당하는 한국 지역을 찾습니다.
    
    Parameters:
    -----------
    x : float
        EPSG:5186 좌표계의 x 좌표(동서)
    y : float
        EPSG:5186 좌표계의 y 좌표(남북)
        
    Returns:
    --------
    str
        해당 좌표가 속한 지역명, 찾지 못한 경우 "대한민국 기타 지역"
    """
    point = Point(x, y)
    
    # 가장 가까운 지역 찾기
    closest_region = None
    min_distance = float('inf')
    
    for region_name, region_info in regions.items():
        center_x, center_y = region_info["center"]
        radius = region_info["radius"]
        
        # 단순 거리 계산 (유클리드 거리)
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 반지름 내에 있거나 가장 가까운 지역 선택
        if distance <= radius:
            return region_name
        
        if distance < min_distance:
            min_distance = distance
            closest_region = region_name
    
    # 반지름 내에 들어가는 지역이 없으면 가장 가까운 지역 반환
    return closest_region if closest_region else "대한민국 기타 지역"

def get_region_info(gdf, original_epsg):
    """
    주어진 GeoDataFrame의 원본 좌표계 중심점과 해당 지역 정보를 반환합니다.
    좌표 변환 없이 한국 지역을 간단히 텍스트로 표시합니다.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        분석할 지오메트리가 포함된 GeoDataFrame
    original_epsg : int
        입력 GeoDataFrame의 원본 좌표계 EPSG 코드
        
    Returns:
    --------
    dict
        region_name: 해당 지역명
        center_point_original: 원본 좌표계의 중심점 좌표 (x, y)
        original_epsg: 원본 좌표계 EPSG 코드
    """
    # 원본 좌표계에서 중심점 계산
    original_centroid = gdf.unary_union.centroid
    original_x, original_y = original_centroid.x, original_centroid.y
    
    # 좌표 변환 없이 간단한 지역 정보 제공
    region_name = "대한민국"
    
    # 바운딩 박스 기준으로 대략적인 위치 확인
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    
    return {
        "region_name": region_name,
        "center_point_original": (original_x, original_y),
        "original_epsg": original_epsg
    }