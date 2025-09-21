import logging
import os
import shutil
import tempfile
import zipfile

import ezdxf
import geopandas as gpd
import streamlit as st
from ezdxf.addons import geo
from shapely.geometry import MultiPolygon, Polygon


def validate_file(uploaded_file):
    """
    Validate the uploaded file. For ZIPs, it now returns the extraction directory.
    """
    logging.info("validate_file 함수 내부 실행 시작")
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    logging.info(f"파일 확장자 확인: {file_ext}")

    if file_ext not in [".dxf", ".zip"]:
        message = "Invalid file type. Please upload a DXF file or ZIP file containing SHP files."
        logging.warning(message)
        return (False, message, None)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    temp_file_path = temp_file.name
    logging.info(f"임시 파일 생성: {temp_file_path}")

    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logging.info(f"업로드된 파일을 임시 파일에 저장: {temp_file_path}")

        if file_ext == ".dxf":
            logging.info("DXF 파일 유효성 검사 시작")
            try:
                doc = ezdxf.readfile(temp_file_path)
                _ = doc.modelspace()
                logging.info("DXF 파일 유효성 검사 성공")
                return True, "Valid DXF file", temp_file_path
            except Exception as e:
                logging.exception("DXF 파일 처리 중 오류 발생")
                return False, f"Invalid DXF file: {str(e)}", None

        elif file_ext == ".zip":
            logging.info("ZIP 파일 유효성 검사 시작")
            extract_dir = tempfile.mkdtemp()
            logging.info(f"ZIP 압축 해제를 위한 임시 디렉토리 생성: {extract_dir}")
            try:
                with zipfile.ZipFile(temp_file_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                logging.info("ZIP 파일 압축 해제 완료")

                shp_files = [os.path.join(root, file) for root, _, files in os.walk(
                    extract_dir) for file in files if file.lower().endswith(".shp")]
                logging.info(f"발견된 SHP 파일: {shp_files}")

                if not shp_files:
                    shutil.rmtree(extract_dir)
                    message = "No SHP files found in the ZIP archive"
                    logging.warning(message)
                    return False, message, None

                logging.info("ZIP 파일 내 SHP 유효성 검사 성공")
                # Return the path to the extraction DIRECTORY
                return True, "Valid shapefile found in ZIP", extract_dir

            except Exception as e:
                logging.exception("ZIP 파일 처리 중 오류 발생")
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
                return False, f"Error processing ZIP file: {str(e)}", None

    except Exception as e:
        logging.exception("파일 처리 중 예상치 못한 오류 발생")
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return False, f"An unexpected error occurred: {str(e)}", None


def process_uploaded_file(file_path, epsg_code):
    """
    Process the uploaded file or directory and extract closed polygons.
    """
    if os.path.isdir(file_path):
        logging.info(f"디렉토리 내 모든 SHP 파일 병합 시작: {file_path}")
        return process_shp_directory(file_path, epsg_code)

    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == ".dxf":
        logging.info(f"DXF 파일 처리 시작: {file_path}")
        return process_dxf_file(file_path, epsg_code)
    else:
        raise ValueError(f"Unsupported file path or extension: {file_path}")


def process_shp_directory(dir_path, epsg_code):
    """
    Process a directory of SHP files to extract and merge polygons.
    """
    try:
        shp_files = [os.path.join(root, file) for root, _, files in os.walk(
            dir_path) for file in files if file.lower().endswith(".shp")]
        if not shp_files:
            logging.warning(f"No SHP files found in directory: {dir_path}")
            return None

        logging.info(f"병합할 SHP 파일 {len(shp_files)}개 발견.")

        gdfs = []
        for shp_file in shp_files:
            try:
                gdf = gpd.read_file(shp_file)
                gdfs.append(gdf)
            except Exception as e:
                logging.warning(f"파일 읽기 실패 {shp_file}: {e}")
                continue

        if not gdfs:
            logging.error("SHP 파일들을 읽을 수 없습니다.")
            return None

        merged_gdf = gpd.pd.concat(gdfs, ignore_index=True)
        logging.info("모든 SHP 파일을 하나의 GeoDataFrame으로 병합 완료")

        # Process the merged GeoDataFrame
        return process_gdf(merged_gdf, epsg_code)

    except Exception as e:
        raise ValueError(f"SHP 디렉토리 처리 중 오류 발생: {str(e)}")


def process_dxf_file(file_path, epsg_code):
    """
    Process a DXF file to extract closed polygons.
    """
    try:
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        polygons = []

        for entity in msp.query('LWPOLYLINE[closed==True], POLYLINE[closed==True]'):
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
            logging.warning("DXF 파일에서 유효한 폴리곤을 찾지 못했습니다.")
            return None

        gdf = gpd.GeoDataFrame(geometry=polygons, crs=f"EPSG:{epsg_code}")
        return process_gdf(gdf, epsg_code)

    except Exception as e:
        raise ValueError(f"DXF 파일 처리 중 오류 발생: {str(e)}")


def process_shp_file(file_path, epsg_code):
    """
    Process a single SHP file to extract closed polygons.
    """
    try:
        gdf = gpd.read_file(file_path)
        return process_gdf(gdf, epsg_code)
    except Exception as e:
        raise ValueError(f"SHP 파일 처리 중 오류 발생: {str(e)}")


def process_gdf(gdf, epsg_code):
    """
    Helper function to filter, clean, and transform a GeoDataFrame.
    """
    if gdf.crs is None:
        gdf.set_crs(epsg=epsg_code, inplace=True)

    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        logging.warning("필터링 후 GeoDataFrame이 비어있습니다 (폴리곤 없음).")
        return None

    gdf = gdf[gdf.geometry.is_valid]
    if gdf.empty:
        logging.warning("필터링 후 GeoDataFrame이 비어있습니다 (유효한 지오메트리 없음).")
        return None

    gdf = gdf.to_crs("EPSG:5179")
    logging.info("GeoDataFrame 처리 및 좌표계 변환 완료 (EPSG:5179)")

    return gdf
