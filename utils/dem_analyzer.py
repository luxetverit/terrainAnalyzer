import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import os

def analyze_elevation(gdf, dem_path, epsg_code):
    """
    Analyze elevation data for the given geometries using the DEM file.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries.
    dem_path : str
        Path to the DEM GeoTIFF file.
    epsg_code : int
        The EPSG code of the input file's coordinate system.
        
    Returns:
    --------
    tuple
        (stats, elevation_array, bounds)
        stats: dict - Statistics of the elevation data
        elevation_array: numpy.ndarray - The masked elevation data
        bounds: tuple - The bounds of the analysis area (left, bottom, right, top)
    """
    # If the DEM file doesn't exist (for demo purposes), generate sample data
    if not os.path.exists(dem_path):
        return simulate_elevation_analysis(gdf)
        
    try:
        # Open the DEM file
        with rasterio.open(dem_path) as src:
            # Check if the GeoDataFrame needs to be reprojected
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            
            # Get geometries as list for masking
            geometries = gdf.geometry.values.tolist()
            
            # Mask the raster with the geometries
            out_image, out_transform = mask(src, geometries, crop=True, filled=True)
            
            # Get the masked data (first band)
            elevation_data = out_image[0]
            
            # Create a mask for no-data values
            no_data_mask = elevation_data == src.nodata
            
            # Get masked array for statistics calculation (ignoring no-data values)
            masked_elevations = np.ma.masked_array(elevation_data, mask=no_data_mask)
            
            # Calculate statistics
            stats = {
                'min': float(masked_elevations.min()),
                'max': float(masked_elevations.max()),
                'mean': float(masked_elevations.mean()),
                'std': float(masked_elevations.std()),
                'count': int((~masked_elevations.mask).sum())
            }
            
            # Calculate the bounds of the masked area
            # Get window bounds from the output transform
            window_bounds = rasterio.windows.bounds(
                rasterio.windows.Window(
                    col_off=0,
                    row_off=0,
                    width=elevation_data.shape[1],
                    height=elevation_data.shape[0]
                ),
                out_transform
            )
            
            # Calculate area in square kilometers
            # Get pixel size in meters
            pixel_size_x = out_transform[0]
            pixel_size_y = abs(out_transform[4])
            # Calculate area in square meters
            pixel_area = pixel_size_x * pixel_size_y
            total_area = pixel_area * stats['count']
            # Convert to square kilometers
            stats['area'] = round(total_area / 1000000, 2)
            
            return stats, elevation_data, window_bounds
    
    except Exception as e:
        raise ValueError(f"Error analyzing elevation data: {str(e)}")

def simulate_elevation_analysis(gdf):
    """
    Simulate elevation analysis for demonstration purposes when the DEM file is not available.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries.
        
    Returns:
    --------
    tuple
        (stats, elevation_array, bounds)
        stats: dict - Statistics of the simulated elevation data
        elevation_array: numpy.ndarray - The simulated elevation data
        bounds: tuple - The bounds of the analysis area (left, bottom, right, top)
    """
    # Get the total bounds of all geometries
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Create a simulated elevation grid
    width = 200
    height = 200
    elevation_array = np.zeros((height, width))
    
    # Generate a sample elevation pattern (a simple gradient with some noise)
    y, x = np.mgrid[0:height, 0:width]
    elevation_array = 100 + 0.5 * x + 0.3 * y + np.random.normal(0, 10, (height, width))
    
    # Create a central feature (mountain or valley)
    center_x, center_y = width // 2, height // 2
    for i in range(height):
        for j in range(width):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            elevation_array[i, j] += 200 * np.exp(-0.005 * dist**2)
    
    # Calculate statistics
    stats = {
        'min': float(np.min(elevation_array)),
        'max': float(np.max(elevation_array)),
        'mean': float(np.mean(elevation_array)),
        'std': float(np.std(elevation_array)),
        'count': width * height,
        'area': round((maxx - minx) * (maxy - miny) / 1000000, 2)  # Approximate area in sq km
    }
    
    # Use the GeoDataFrame bounds for the visualization
    bounds = (minx, miny, maxx, maxy)
    
    return stats, elevation_array, bounds
