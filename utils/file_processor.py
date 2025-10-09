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
    Validate the uploaded file. For ZIPs, it now returns the extraction directory.
    """
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    if file_ext not in [".dxf", ".zip"]:
        return (False, "Invalid file type. Please upload a DXF file or ZIP file.", None)

    # Use a 'with' statement to create and write to the temp file.
    # This ensures the file handle is closed automatically, releasing any locks.
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
    except Exception as e:
        return False, f"Failed to create a temporary file: {str(e)}", None

    # Now, the file is closed but exists. We can safely use its path.
    if file_ext == ".dxf":
        try:
            doc = ezdxf.readfile(temp_file_path)
            _ = doc.modelspace()
            # On success, return the path. The caller is responsible for cleanup.
            return True, "Valid DXF file", temp_file_path
        except Exception as e:
            # On failure, clean up the temp file.
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return False, f"Invalid DXF file: {str(e)}", None

    elif file_ext == ".zip":
        extract_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(temp_file_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # Find shapefiles within the extracted directory
            shp_files = [os.path.join(root, file) for root, _, files in os.walk(
                extract_dir) for file in files if file.lower().endswith(".shp")]

            if not shp_files:
                shutil.rmtree(extract_dir)
                return False, "No SHP files found in the ZIP archive.", None

            if len(shp_files) > 1:
                shutil.rmtree(extract_dir)
                return False, "여러개의 shp파일이 검색되었습니다. 하나의 shp파일은 한개만 허용됩니다.", None

            # On success, return the extraction dir. The caller is responsible for its cleanup.
            return True, "Valid shapefile found in ZIP", extract_dir
        except Exception as e:
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            return False, f"Error processing ZIP file: {str(e)}", None
        finally:
            # The temporary ZIP file is no longer needed, so always clean it up.
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


def process_uploaded_file(file_path, epsg_code):
    """
    Process the uploaded file or directory and extract closed polygons.
    """
    if os.path.isdir(file_path):
        return process_shp_directory(file_path, epsg_code)

    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == ".dxf":
        return process_dxf_file(file_path, epsg_code)
    else:
        # This case might be for a single .shp file if you ever support that
        return process_shp_file(file_path, epsg_code)


def process_shp_directory(dir_path, epsg_code):
    """
    Process all SHP files in a directory, merge them, and extract polygons.
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
            print(f"Warning: Could not read file {f}: {e}")
            continue

    if not gdfs:
        return None

    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    merged_gdf.crs = gdfs[0].crs

    return process_shp_file(merged_gdf, epsg_code)


def process_dxf_file(file_path, epsg_code):
    """
    Process a DXF file to extract closed polygons.
    """
    try:
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        polygons = []

        # Query for entity types and filter by attribute in Python for robustness
        lwpolylines = msp.query('LWPOLYLINE')
        polylines = msp.query('POLYLINE')
        
        for entity in itertools.chain(lwpolylines, polylines):
            # Check if the entity is closed using its attribute
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
        return process_shp_file(gdf, epsg_code)  # Re-use the main processor

    except Exception as e:
        raise ValueError(f"Error processing DXF file: {str(e)}")


def process_shp_file(data, epsg_code):
    """
    Process a SHP file path or a GeoDataFrame to extract closed polygons,
    ensuring the data is in the correct CRS.
    """
    try:
        if isinstance(data, str):
            gdf = gpd.read_file(data)
        elif isinstance(data, gpd.GeoDataFrame):
            gdf = data
        else:
            raise TypeError("Input must be a file path or a GeoDataFrame.")

        # Standardize the target CRS format
        target_crs = f"EPSG:{epsg_code}"

        # 1. If CRS is missing, assign the user-selected one.
        if gdf.crs is None:
            st.warning(f"원본 데이터에 좌표계 정보가 없어, 사용자가 선택한 '{target_crs}'를 적용합니다.")
            gdf.set_crs(epsg=epsg_code, inplace=True)
        # 2. If CRS is different, reproject.
        elif gdf.crs.to_string() != target_crs:
            original_crs = gdf.crs.to_string()
            st.info(f"원본 좌표계 '{original_crs}'를 사용자가 선택한 '{target_crs}'로 변환합니다.")
            gdf = gdf.to_crs(epsg=epsg_code)

        # Filter for valid Polygon or MultiPolygon geometries
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        gdf = gdf[gdf.geometry.is_valid]

        return gdf if not gdf.empty else None

    except Exception as e:
        raise ValueError(f"Error processing SHP data: {str(e)}")


def clip_geodataframe(target_gdf, clip_gdf):
    """
    Clips a target GeoDataFrame to the boundaries of a clipping GeoDataFrame.

    Args:
        target_gdf (gpd.GeoDataFrame): The GeoDataFrame to be clipped.
        clip_gdf (gpd.GeoDataFrame): The GeoDataFrame defining the clipping boundaries.

    Returns:
        gpd.GeoDataFrame: The clipped GeoDataFrame.
    """
    if target_gdf.empty or clip_gdf.empty:
        return gpd.GeoDataFrame(columns=target_gdf.columns, crs=target_gdf.crs)
        
    # Ensure both GeoDataFrames have the same CRS
    if target_gdf.crs != clip_gdf.crs:
        target_gdf = target_gdf.to_crs(clip_gdf.crs)

    # Perform the clip
    clipped_gdf = gpd.clip(target_gdf, clip_gdf, keep_geom_type=True)

    # Filter out any invalid or empty geometries that might result from clipping
    clipped_gdf = clipped_gdf[clipped_gdf.geometry.is_valid & ~
                              clipped_gdf.geometry.is_empty]

    return clipped_gdf

