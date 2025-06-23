import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from utils.color_palettes import ALL_PALETTES

def create_elevation_heatmap(elevation_array, bounds, stats, palette_key='spectral'):
    """
    Create a heatmap visualization of the elevation data.
    
    Parameters:
    -----------
    elevation_array : numpy.ndarray
        The elevation data array.
    bounds : tuple
        The bounds of the analysis area (left, bottom, right, top).
    stats : dict
        Statistics of the elevation data.
    palette_key : str, optional
        The key for the color palette to use (default: 'spectral').
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the visualization.
    """
    # Get the appropriate color palette
    if palette_key not in ALL_PALETTES:
        palette_key = 'spectral'  # 기본값
    
    colors = ALL_PALETTES[palette_key]['colors']
    cmap = LinearSegmentedColormap.from_list(f'{palette_key}_cmap', colors, N=256)
    
    # Create the figure with a larger size
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the elevation data as a heatmap
    im = ax.imshow(
        elevation_array,
        cmap=cmap,
        aspect='auto',
        origin='lower',
        extent=bounds,
        vmin=stats['min'],
        vmax=stats['max']
    )
    
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Elevation (m)', rotation=270, labelpad=20)
    
    # Add contour lines for better elevation visualization
    levels = np.linspace(stats['min'], stats['max'], 15)
    contour = ax.contour(
        elevation_array,
        levels=levels,
        colors='k',
        alpha=0.3,
        extent=bounds
    )
    
    # Set labels and title
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title('Elevation Analysis', fontweight='bold', fontsize=14)
    
    # Add statistics annotation
    stats_text = (
        f"Min: {stats['min']:.2f} m\n"
        f"Max: {stats['max']:.2f} m\n"
        f"Mean: {stats['mean']:.2f} m\n"
        f"Area: {stats.get('area', 'N/A')} sq km"
    )
    
    # Create a box for the statistics
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=props,
        fontsize=10
    )
    
    # Add grid lines for reference
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add elevation range legend
    handles = [
        mpatches.Patch(color=colors[0], label=f'Min ({stats["min"]:.1f} m)'),
        mpatches.Patch(color=colors[5], label=f'Mid ({stats["min"] + (stats["max"]-stats["min"])/2:.1f} m)'),
        mpatches.Patch(color=colors[-1], label=f'Max ({stats["max"]:.1f} m)')
    ]
    ax.legend(handles=handles, loc='lower right', title="Elevation Range")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig
