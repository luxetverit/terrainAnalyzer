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

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    temp_file_path = temp_file.name

    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if file_ext == ".dxf":
            try:
                doc = ezdxf.readfile(temp_file_path)
                _ = doc.modelspace()
                return True, "Valid DXF file", temp_file_path
            except Exception as e:
                return False, f"Invalid DXF file: {str(e)}", None

        elif file_ext == ".zip":
            extract_dir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(temp_file_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)

                shp_files = [os.path.join(root, file) for root, _, files in os.walk(
                    extract_dir) if file.lower().endswith(".shp")]
                if not shp_files:
                    shutil.rmtree(extract_dir)
                    return False, "No SHP files found in the ZIP archive", None

                return True, "Valid shapefile found in ZIP", extract_dir
            except Exception as e:
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
                return False, f"Error processing ZIP file: {str(e)}", None
    finally:
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
            return None

        gdf = gpd.GeoDataFrame(geometry=polygons, crs=f"EPSG:{epsg_code}")
        return process_shp_file(gdf, epsg_code)  # Re-use the main processor

    except Exception as e:
        raise ValueError(f"Error processing DXF file: {str(e)}")


def process_shp_file(data, epsg_code):
    """
    Process a SHP file path or a GeoDataFrame to extract closed polygons.
    """
    try:
        if isinstance(data, str):
            gdf = gpd.read_file(data)
        elif isinstance(data, gpd.GeoDataFrame):
            gdf = data
        else:
            raise TypeError("Input must be a file path or a GeoDataFrame.")

        if gdf.crs is None:
            gdf.set_crs(epsg=epsg_code, inplace=True)

        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        gdf = gdf[gdf.geometry.is_valid]
        gdf = gdf.to_crs("EPSG:5179")

        return gdf if not gdf.empty else None

    except Exception as e:
        raise ValueError(f"Error processing SHP data: {str(e)}")
        raise ValueError(f"Error processing SHP data: {str(e)}")
