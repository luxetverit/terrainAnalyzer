import requests
import geopandas as gpd
import streamlit as st

def get_location_name(gdf, original_epsg, api_key):
    """
    GeoDataFrame의 중심점에 해당하는 주소 정보를 카카오맵 API로 조회합니다.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        위치 정보를 찾을 GeoDataFrame
    original_epsg : int
        원본 좌표계 EPSG 코드
    api_key : str
        카카오 API 키
        
    Returns:
    --------
    dict
        위치 정보를 담은 딕셔너리 (road_address, address, lon, lat)
    """
    # 원본 좌표계의 중심점 계산
    centroid = gdf.unary_union.centroid
    original_x, original_y = centroid.x, centroid.y
    
    # WGS84 좌표계로 변환 (카카오맵 API 사용을 위해)
    if original_epsg != 4326:
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        centroid_wgs84 = gdf_wgs84.unary_union.centroid
        lon, lat = centroid_wgs84.x, centroid_wgs84.y
    else:
        lon, lat = original_x, original_y
    
    # 카카오맵 API로 주소 조회
    api_url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"x": lon, "y": lat}
    
    try:
        response = requests.get(api_url, headers=headers, params=params)
        
        if response.status_code != 200:
            return {
                "road_address": "주소 정보를 찾을 수 없습니다",
                "address": "주소 정보를 찾을 수 없습니다",
                "region": "대한민국",
                "original_x": original_x,
                "original_y": original_y,
                "lon": lon,
                "lat": lat
            }
        
        data = response.json()
        
        if not data or "documents" not in data or len(data["documents"]) == 0:
            return {
                "road_address": "주소 정보를 찾을 수 없습니다",
                "address": "주소 정보를 찾을 수 없습니다",
                "region": "대한민국",
                "original_x": original_x,
                "original_y": original_y,
                "lon": lon,
                "lat": lat
            }
        
        # 주소 정보 추출
        document = data["documents"][0]
        road_address_data = document.get("road_address")
        address_data = document.get("address", {})
        
        road_address = road_address_data.get("address_name", "주소 정보를 찾을 수 없습니다") if road_address_data else "주소 정보를 찾을 수 없습니다"
        address = address_data.get("address_name", "주소 정보를 찾을 수 없습니다")
        
        # 지역 정보 추출 (시도, 구군)
        region1 = address_data.get("region_1depth_name", "")
        region2 = address_data.get("region_2depth_name", "")
        region3 = address_data.get("region_3depth_name", "")
        
        region = f"{region1} {region2} {region3}".strip()
        if not region:
            region = "대한민국"
        
        return {
            "road_address": road_address,
            "address": address,
            "region": region,
            "original_x": original_x,
            "original_y": original_y,
            "lon": lon,
            "lat": lat
        }
        
    except Exception as e:
        return {
            "road_address": f"주소 조회 오류: {str(e)}",
            "address": f"주소 조회 오류: {str(e)}",
            "region": "대한민국",
            "original_x": original_x,
            "original_y": original_y,
            "lon": lon,
            "lat": lat
        }