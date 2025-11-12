import itertools
import os
import shutil
import tempfile
import zipfile

import ezdxf
import geopandas as gpd
import pandas as pd
import streamlit as st
from ezdxf.addons import geo
from shapely.geometry import MultiPolygon, Polygon


def validate_file(uploaded_file):
    """
    업로드된 파일의 유효성을 검사합니다. ZIP 파일의 경우 이제 압축 해제 디렉토리를 반환합니다.
    """
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    if file_ext not in [".dxf", ".zip"]:
        return (False, "잘못된 파일 형식입니다. DXF 또는 ZIP 파일을 업로드해주세요.", None)

    # 'with' 구문을 사용하여 임시 파일을 만들고 씁니다.
    # 이렇게 하면 파일 핸들이 자동으로 닫히고 모든 잠금이 해제됩니다.
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
    except Exception as e:
        return False, f"임시 파일을 생성하지 못했습니다: {str(e)}", None

    # 이제 파일은 닫혔지만 존재합니다. 해당 경로를 안전하게 사용할 수 있습니다.
    if file_ext == ".dxf":
        try:
            doc = ezdxf.readfile(temp_file_path)
            _ = doc.modelspace()
            # 성공 시 경로를 반환합니다. 호출자가 정리를 책임집니다.
            return True, "유효한 DXF 파일", temp_file_path
        except Exception as e:
            # 실패 시 임시 파일을 정리합니다.
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return False, f"잘못된 DXF 파일: {str(e)}", None

    elif file_ext == ".zip":
        extract_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(temp_file_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # 압축 해제된 디렉토리 내에서 shapefile 찾기
            shp_files = [os.path.join(root, file) for root, _, files in os.walk(
                extract_dir) for file in files if file.lower().endswith(".shp")]

            if not shp_files:
                shutil.rmtree(extract_dir)
                return False, "ZIP 아카이브에서 SHP 파일을 찾을 수 없습니다.", None

            if len(shp_files) > 1:
                shutil.rmtree(extract_dir)
                return False, "여러개의 shp파일이 검색되었습니다. 하나의 shp파일은 한개만 허용됩니다.", None

            # 성공 시 압축 해제 디렉토리를 반환합니다. 호출자가 정리를 책임집니다.
            return True, "ZIP에서 유효한 shapefile을 찾았습니다.", extract_dir
        except Exception as e:
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            return False, f"ZIP 파일 처리 오류: {str(e)}", None
        finally:
            # 임시 ZIP 파일은 더 이상 필요하지 않으므로 항상 정리합니다.
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


def process_uploaded_file(file_path, epsg_code):
    """
    업로드된 파일 또는 디렉토리를 처리하고 닫힌 폴리곤을 추출합니다.
    """
    if os.path.isdir(file_path):
        return process_shp_directory(file_path, epsg_code)

    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == ".dxf":
        return process_dxf_file(file_path, epsg_code)
    else:
        # 이 경우는 단일 .shp 파일을 지원할 경우를 위한 것입니다.
        return process_shp_file(file_path, epsg_code)


def process_shp_directory(dir_path, epsg_code):
    """
    디렉토리의 모든 SHP 파일을 처리하고 병합한 다음 폴리곤을 추출합니다.
    """
    shp_files = [os.path.join(root, file) for root, _, files in os.walk(
        dir_path) for file in files if file.lower().endswith(".shp")]
    if not shp_files:
        return None

    gdfs = []
    for f in shp_files:
        try:
            gdf = gpd.read_file(f)
            if gdf.crs is None:
                gdf.crs = f"EPSG:{epsg_code}"
            gdfs.append(gdf)
        except Exception as e:
            print(f"경고: 파일을 읽을 수 없습니다 {f}: {e}")
            continue

    if not gdfs:
        return None

    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    merged_gdf.crs = gdfs[0].crs

    return process_shp_file(merged_gdf, epsg_code)


def process_dxf_file(file_path, epsg_code):
    """
    DXF 파일을 처리하여 닫힌 폴리곤을 추출합니다.
    """
    try:
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        polygons = []

        # 견고성을 위해 파이썬에서 엔티티 유형을 쿼리하고 속성으로 필터링합니다.
        lwpolylines = msp.query('LWPOLYLINE')
        polylines = msp.query('POLYLINE')
        
        for entity in itertools.chain(lwpolylines, polylines):
            # 속성을 사용하여 엔티티가 닫혔는지 확인합니다.
            if entity.is_closed:
                vertices = [v[:2] for v in entity.vertices()]
                if len(vertices) >= 3:
                    polygons.append(Polygon(vertices))

        for hatch in msp.query('HATCH'):
            for path in hatch.paths:
                if path.PATH_TYPE == 'PolylinePath':
                    vertices = [v[:2] for v in path.vertices]
                    if len(vertices) >= 3:
                        polygons.append(Polygon(vertices))

        if not polygons:
            return None

        gdf = gpd.GeoDataFrame(geometry=polygons, crs=f"EPSG:{epsg_code}")
        return process_shp_file(gdf, epsg_code)  # 주 프로세서 재사용

    except Exception as e:
        raise ValueError(f"DXF 파일 처리 오류: {str(e)}")


def process_shp_file(data, epsg_code):
    """
    SHP 파일 경로 또는 GeoDataFrame을 처리하여 닫힌 폴리곤을 추출하고,
    데이터가 올바른 CRS에 있는지 확인합니다.
    """
    try:
        if isinstance(data, str):
            gdf = gpd.read_file(data)
        elif isinstance(data, gpd.GeoDataFrame):
            gdf = data
        else:
            raise TypeError("입력은 파일 경로 또는 GeoDataFrame이어야 합니다.")

        # 대상 CRS 형식 표준화
        target_crs = f"EPSG:{epsg_code}"

        # 1. CRS가 없으면 사용자가 선택한 CRS를 할당합니다.
        if gdf.crs is None:
            st.warning(f"원본 데이터에 좌표계 정보가 없어, 사용자가 선택한 '{target_crs}'를 적용합니다.")
            gdf.set_crs(epsg=epsg_code, inplace=True)
        # 2. CRS가 다르면 재투영합니다.
        elif gdf.crs.to_string() != target_crs:
            original_crs = gdf.crs.to_string()
            st.info(f"원본 좌표계 '{original_crs}'를 사용자가 선택한 '{target_crs}'로 변환합니다.")
            gdf = gdf.to_crs(epsg=epsg_code)

        # 유효한 폴리곤 또는 멀티폴리곤 지오메트리 필터링
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        gdf = gdf[gdf.geometry.is_valid]

        return gdf if not gdf.empty else None

    except Exception as e:
        raise ValueError(f"SHP 데이터 처리 오류: {str(e)}")


def clip_geodataframe(target_gdf, clip_gdf):
    """
    대상 GeoDataFrame을 클리핑 GeoDataFrame의 경계로 자릅니다.
    안정성을 위해 geopandas의 기본 clip 기능을 사용합니다.
    """
    if target_gdf.empty or clip_gdf.empty:
        return gpd.GeoDataFrame(columns=target_gdf.columns, crs=target_gdf.crs)

    # CRS가 일치하는지 확인하고, 다르면 변환합니다.
    if target_gdf.crs != clip_gdf.crs:
        target_gdf = target_gdf.to_crs(clip_gdf.crs)

    # 클리핑 전에 도형 유효성 검사 및 복구
    target_gdf['geometry'] = target_gdf.geometry.buffer(0)
    clip_gdf['geometry'] = clip_gdf.geometry.buffer(0)

    # 유효하지 않거나 빈 지오메트리 제거
    target_gdf = target_gdf[target_gdf.geometry.is_valid & ~target_gdf.geometry.is_empty]
    clip_gdf = clip_gdf[clip_gdf.geometry.is_valid & ~clip_gdf.geometry.is_empty]

    if target_gdf.empty or clip_gdf.empty:
        st.warning("클리핑할 유효한 피처가 남아있지 않습니다. 빈 결과를 반환합니다.")
        return gpd.GeoDataFrame(columns=target_gdf.columns, crs=target_gdf.crs)

    # gpd.clip 대신 더 강력한 gpd.overlay 사용
    try:
        clipped_gdf = gpd.overlay(target_gdf, clip_gdf, how='intersection', keep_geom_type=False)
        if not clipped_gdf.empty:
            clipped_gdf = clipped_gdf[clipped_gdf.geometry.is_valid & ~clipped_gdf.geometry.is_empty]
        
        return clipped_gdf
    except Exception as e:
        st.warning(f"클리핑 작업(overlay) 중 오류가 발생했습니다: {e}. 빈 결과를 반환합니다.")
        return gpd.GeoDataFrame(columns=target_gdf.columns, crs=target_gdf.crs)