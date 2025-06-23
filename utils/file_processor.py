import streamlit as st
import os
import tempfile
import geopandas as gpd
import ezdxf
from ezdxf.addons import geo
from shapely.geometry import Polygon, MultiPolygon
import tempfile
import zipfile
import shutil

def validate_file(uploaded_file):
    """
    Validate the uploaded file to ensure it is a valid DXF or ZIP (containing SHP) file.
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        The file uploaded by the user through Streamlit's file_uploader.
        
    Returns:
    --------
    tuple
        (is_valid, message, temp_file_path)
        is_valid: bool - Whether the file is valid
        message: str - A message explaining the validation result
        temp_file_path: str - Path to the temporary file created
    """
    # Check file extension
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_ext not in ['.dxf', '.zip']:
        return False, "Invalid file type. Please upload a DXF file or ZIP file containing SHP files.", None
    
    # Create a temporary file to save the uploaded content
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    temp_file_path = temp_file.name
    
    try:
        # Write uploaded file to the temporary file
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Validate based on file type
        if file_ext == '.dxf':
            try:
                # Try to load the DXF file
                doc = ezdxf.readfile(temp_file_path)
                # Basic check: ensure it has a modelspace
                _ = doc.modelspace()
                return True, "Valid DXF file", temp_file_path
            except Exception as e:
                return False, f"Invalid DXF file: {str(e)}", None
        
        elif file_ext == '.zip':
            try:
                # Create a temporary directory to extract the ZIP contents
                extract_dir = tempfile.mkdtemp()
                
                # Extract the ZIP file
                with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Look for SHP files in the extracted directory
                shp_files = []
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        if file.lower().endswith('.shp'):
                            shp_files.append(os.path.join(root, file))
                
                if not shp_files:
                    # Clean up
                    shutil.rmtree(extract_dir)
                    return False, "No SHP files found in the ZIP archive", None
                
                # Validate the first SHP file found
                try:
                    gdf = gpd.read_file(shp_files[0])
                    if gdf.empty:
                        shutil.rmtree(extract_dir)
                        return False, "Empty shapefile found in ZIP", None
                    
                    # Return the path to the SHP file, not the ZIP file
                    return True, "Valid shapefile found in ZIP", shp_files[0]
                except Exception as e:
                    shutil.rmtree(extract_dir)
                    return False, f"Invalid shapefile in ZIP: {str(e)}", None
                    
            except zipfile.BadZipFile:
                return False, "Invalid ZIP file", None
            except Exception as e:
                return False, f"Error processing ZIP file: {str(e)}", None
    
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return False, f"Error processing file: {str(e)}", None
    
    return True, "Valid file", temp_file_path

def process_uploaded_file(file_path, epsg_code):
    """
    Process the uploaded file and extract closed polygons.
    
    Parameters:
    -----------
    file_path : str
        Path to the temporary file.
    epsg_code : int
        The EPSG code of the input file's coordinate system.
        
    Returns:
    --------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the closed polygons.
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.dxf':
        return process_dxf_file(file_path, epsg_code)
    elif file_ext == '.shp':
        return process_shp_file(file_path, epsg_code)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

def process_dxf_file(file_path, epsg_code):
    """
    Process a DXF file to extract closed polygons.
    
    Parameters:
    -----------
    file_path : str
        Path to the DXF file.
    epsg_code : int
        The EPSG code of the input file's coordinate system.
        
    Returns:
    --------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the closed polygons.
    """
    try:
        # Read DXF file
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        
        # List to store polygons
        polygons = []
        
        # Extract closed polylines and convert to polygons
        for entity in msp:
            if entity.dxftype() == 'LWPOLYLINE' or entity.dxftype() == 'POLYLINE':
                if entity.closed:
                    # Extract vertices
                    vertices = []
                    if entity.dxftype() == 'LWPOLYLINE':
                        for vertex in entity.vertices():
                            vertices.append((vertex[0], vertex[1]))
                    else:  # POLYLINE
                        for vertex in entity.vertices:
                            vertices.append((vertex.dxf.location[0], vertex.dxf.location[1]))
                    
                    # Create polygon if we have at least 3 vertices
                    if len(vertices) >= 3:
                        # Make sure the polygon is closed
                        if vertices[0] != vertices[-1]:
                            vertices.append(vertices[0])
                        
                        # Create Shapely polygon
                        polygon = Polygon(vertices)
                        if polygon.is_valid:
                            polygons.append(polygon)
                            
            elif entity.dxftype() == 'HATCH':
                # Extract boundary paths
                for path in entity.paths:
                    vertices = []
                    for vertex in path.vertices:
                        vertices.append((vertex[0], vertex[1]))
                    
                    # Create polygon if we have at least 3 vertices
                    if len(vertices) >= 3:
                        # Make sure the polygon is closed
                        if vertices[0] != vertices[-1]:
                            vertices.append(vertices[0])
                        
                        # Create Shapely polygon
                        polygon = Polygon(vertices)
                        if polygon.is_valid:
                            polygons.append(polygon)
        
        # Create GeoDataFrame
        if polygons:
            gdf = gpd.GeoDataFrame(geometry=polygons, crs=f"EPSG:{epsg_code}")
            
            # Transform to EPSG:5179 for DEM analysis
            gdf = gdf.to_crs("EPSG:5179")
            
            return gdf
        else:
            return None
            
    except Exception as e:
        raise ValueError(f"Error processing DXF file: {str(e)}")

def process_shp_file(file_path, epsg_code):
    """
    Process a SHP file to extract closed polygons.
    
    Parameters:
    -----------
    file_path : str
        Path to the SHP file.
    epsg_code : int
        The EPSG code of the input file's coordinate system.
        
    Returns:
    --------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the closed polygons.
    """
    try:
        # Read shapefile
        gdf = gpd.read_file(file_path)
        
        # Set CRS if not already set
        if gdf.crs is None:
            gdf.set_crs(epsg=epsg_code, inplace=True)
        
        # Filter for polygon geometries
        gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
        # Filter for valid geometries
        gdf = gdf[gdf.geometry.is_valid]
        
        # Transform to EPSG:5179 for DEM analysis
        gdf = gdf.to_crs("EPSG:5179")
        
        if gdf.empty:
            return None
            
        return gdf
            
    except Exception as e:
        raise ValueError(f"Error processing SHP file: {str(e)}")
