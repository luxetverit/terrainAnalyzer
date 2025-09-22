import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Rectangle

# --- Constants for Plotting ---
INTERVAL_CANDIDATES = [1, 2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200, 500]
ELEVATION_COLORS = {
    10: ['#66CDAA', '#DDF426', '#71B800', '#558C00', '#F29300', '#981200', '#9C1C1C', '#955050', '#9C9B9B', '#FFFAFA']
}

# --- Plotting Helper Functions ---

def add_north_arrow(ax, x=0.95, y=0.95, size=0.05):
    """Adds a compass-style north arrow to the top-right of the Axes."""
    north_part = Polygon([[x, y], [x - size * 0.4, y - size], [x, y - size*0.8]],
                         facecolor='black', edgecolor='black', transform=ax.transAxes)
    south_part = Polygon([[x, y], [x + size * 0.4, y - size], [x, y - size*0.8]],
                         facecolor='#666666', edgecolor='black', transform=ax.transAxes)
    ax.add_patch(north_part)
    ax.add_patch(south_part)
    ax.text(x, y + size*0.2, 'N', ha='center', va='center',
            fontsize='large', fontweight='bold', transform=ax.transAxes)

def add_scalebar_vector(ax, dx=1.0):
    """Adds a scale bar and ratio to the bottom-left of the Axes (for vector plots)."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    map_width_m = (x_max - x_min) * dx
    target_length = map_width_m * 0.5
    powers = 10**np.floor(np.log10(target_length))
    base = np.ceil(target_length / powers)
    if base > 5: base = 5
    elif base > 2: base = 2
    nice_len_m = base * powers

    num_segments = 4
    segment_m = nice_len_m / num_segments
    segment_data = segment_m / dx
    bar_height = (y_max - y_min) * 0.01

    x_pos = x_min + (x_max - x_min) * 0.05
    y_pos = y_min + (y_max - y_min) * 0.05

    for i in range(num_segments):
        color = 'black' if i % 2 == 0 else 'white'
        rect = Rectangle((x_pos + i * segment_data, y_pos), segment_data, bar_height,
                         edgecolor='black', facecolor=color, lw=1, zorder=10)
        ax.add_patch(rect)
        ax.text(x_pos + i * segment_data, y_pos - bar_height*0.5, f'{int(i * segment_m)}',
                ha='center', va='top', fontsize='small', zorder=10)

    ax.text(x_pos + nice_len_m / dx, y_pos - bar_height*0.5, f'{int(nice_len_m)} m',
            ha='center', va='top', fontsize='small', zorder=10)

def calculate_accurate_scalebar_params(pixel_size, img_shape, target_size_mm, fig, ax):
    """Calculates accurate scale bar parameters using actual axes information."""
    fig_width_inch, _ = fig.get_size_inches()
    ax_bbox = ax.get_position()
    _, img_width_pixels = img_shape
    img_width_inch = ax_bbox.width * fig_width_inch
    pixels_per_inch = img_width_pixels / img_width_inch
    target_pixels = (target_size_mm / 25.4) * pixels_per_inch
    real_distance_m = target_pixels * pixel_size

    if real_distance_m < 100:
        scale_distance_m = round(real_distance_m / 50) * 50 or 50
        unit, scale_value = 'm', scale_distance_m
    elif real_distance_m < 1000:
        scale_distance_m = round(real_distance_m / 100) * 100
        unit, scale_value = 'm', scale_distance_m
    elif real_distance_m < 5000:
        scale_distance_km = round(real_distance_m / 500) * 0.5
        scale_distance_m = scale_distance_km * 1000
        unit, scale_value = 'km', scale_distance_km
    else:
        scale_distance_km = round(real_distance_m / 1000)
        scale_distance_m = scale_distance_km * 1000
        unit, scale_value = 'km', scale_distance_km

    return {
        'length': scale_value, 'units': unit, 'segments': 2,
        'scalebar_width_fig': (scale_distance_m / pixel_size) / pixels_per_inch / fig_width_inch
    }

def draw_accurate_scalebar(fig, ax, pixel_size, scale_params, img_shape):
    """Draws an accurate scale bar."""
    total_length, units, segments = scale_params['length'], scale_params['units'], scale_params['segments']
    scalebar_width_fig = scale_params['scalebar_width_fig']
    start_x_fig, start_y_fig, bar_height_fig = 0.1, 0.02, 0.008

    bg_rect = Rectangle((start_x_fig - 0.005, start_y_fig - 0.005), scalebar_width_fig + 0.01,
                        bar_height_fig * 2 + 0.03, facecolor='white', edgecolor='none',
                        alpha=0.9, transform=fig.transFigure)
    fig.patches.append(bg_rect)

    segment_width_fig = scalebar_width_fig / segments
    for i in range(segments):
        x_fig = start_x_fig + i * segment_width_fig
        for j in range(2):
            color = 'black' if (i + j) % 2 == 0 else 'white'
            rect = Rectangle((x_fig, start_y_fig + j * bar_height_fig), segment_width_fig,
                             bar_height_fig, facecolor=color, edgecolor='black',
                             linewidth=0.5, transform=fig.transFigure)
            fig.patches.append(rect)

    for i in range(segments + 1):
        text_x_fig = start_x_fig + i * segment_width_fig
        text_y_fig = start_y_fig + bar_height_fig * 2 + 0.005
        segment_val = i * (total_length / segments)
        text_label = f'{segment_val:.1f}' if units == 'km' and segment_val != int(segment_val) else f'{int(segment_val)}'
        if i == segments: text_label += units
        fig.text(text_x_fig, text_y_fig, text_label, ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='black', transform=fig.transFigure)

def create_hillshade(data, azimuth=315, altitude=45):
    """Creates a high-quality hillshade from elevation data."""
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask): return np.ones_like(data) * 0.5
    dy, dx = np.gradient(data)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    azimuth_rad, altitude_rad = np.radians(azimuth), np.radians(altitude)
    hillshade = (np.sin(altitude_rad) * np.sin(np.pi/2 - slope) +
                 np.cos(altitude_rad) * np.cos(np.pi/2 - slope) *
                 np.cos(azimuth_rad - aspect))
    hillshade = np.clip(hillshade, 0, 1)
    hillshade[~valid_mask] = 0.5
    return hillshade

def create_padded_fig_ax(figsize=(10, 10)):
    """Creates a Figure and Axes with padding."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def generate_custom_intervals(min_val, max_val, divisions):
    """Generates custom intervals for legends."""
    if max_val - min_val == 0:
        return [f"{min_val:.1f}"] * divisions, 0, min_val
    diff = max_val - min_val
    div = divisions - 2 or 1
    target_interval = diff / div
    best_interval = min(INTERVAL_CANDIDATES, key=lambda x: abs(x - target_interval))
    for candidate in sorted(INTERVAL_CANDIDATES, key=lambda x: abs(x - target_interval)):
        if candidate * div > diff * 0.5:
            best_interval = candidate
            break
    start = round((min_val + diff / 2) / best_interval) * best_interval - best_interval * (div // 2)
    labels = [f"{start} 미만"] + [f"{start + i * best_interval}~{start + (i + 1) * best_interval}" for i in range(div)] + [f"{start + div * best_interval} 초과"]
    return labels, best_interval, start

def adjust_ax_limits(ax, y_pad_fraction=0.15, x_pad_fraction=0.05):
    """Expands the Axes limits to create padding."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range, y_range = x_max - x_min, y_max - y_min
    ax.set_xlim(x_min - x_range * x_pad_fraction, x_max + x_range * x_pad_fraction)
    ax.set_ylim(y_min - y_range * y_pad_fraction, y_max + y_range * y_pad_fraction)