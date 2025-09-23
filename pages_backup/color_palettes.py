"""
Color palettes collection for visualization of analysis results
"""

# Spectral palette (default)
SPECTRAL = [
    "#5E4FA2",
    "#3288BD",
    "#66C2A5",
    "#ABDDA4",
    "#E6F598",
    "#FFFFBF",
    "#FEE08B",
    "#FDAE61",
    "#F46D43",
    "#D53E4F",
]

# Red-Yellow-Green palette
RDYLGN = [
    "#A50026",
    "#D73027",
    "#F46D43",
    "#FDAE61",
    "#FEE08B",
    "#FFFFBF",
    "#D9EF8B",
    "#A6D96A",
    "#66BD63",
    "#1A9850",
]

# Terrain palette
TERRAIN = [
    "#333366",
    "#336699",
    "#339966",
    "#66CC66",
    "#99CC66",
    "#CCCC66",
    "#FFCC66",
    "#FF9966",
    "#FF6666",
    "#FF3333",
]

# Viridis palette
VIRIDIS = [
    "#440154",
    "#482777",
    "#3F4A8A",
    "#31688E",
    "#26828E",
    "#1F9E89",
    "#35B779",
    "#6CCE59",
    "#B4DE2C",
    "#FDE725",
]

# Inferno palette
INFERNO = [
    "#000004",
    "#1B0C41",
    "#4A0C6B",
    "#781C6D",
    "#A52C60",
    "#CF4446",
    "#ED6925",
    "#FB9A06",
    "#F7D13D",
    "#FCFFA4",
]

# Grayscale palette
GREYS = [
    "#000000",
    "#1C1C1C",
    "#383838",
    "#545454",
    "#707070",
    "#8C8C8C",
    "#A8A8A8",
    "#C4C4C4",
    "#E0E0E0",
    "#FFFFFF",
]

# Brown palette (new)
BROWN = [
    "#f7f1e1",
    "#e6d8b9",
    "#d6b88c",
    "#c69a6d",
    "#a66d4e",
    "#844c2e",
    "#5c2f1a",
    "#3e1f10",
    "#2a1308",
    "#1a0a04",
]

# All palettes in a dictionary
ALL_PALETTES = {
    "spectral": {"name": "Spectral", "colors": SPECTRAL},
    "rdylgn": {"name": "Red-Yellow-Green", "colors": RDYLGN},
    "terrain": {"name": "Terrain", "colors": TERRAIN},
    "viridis": {"name": "Viridis", "colors": VIRIDIS},
    "inferno": {"name": "Inferno", "colors": INFERNO},
    "greys": {"name": "Grayscale", "colors": GREYS},
    "brown": {"name": "Brown", "colors": BROWN},
}


def get_palette_preview_html(palette_key):
    """
    Generate HTML preview for a color palette.

    Parameters:
    -----------
    palette_key : str
        Palette key (e.g., 'spectral', 'viridis')

    Returns:
    --------
    str
        HTML code for palette preview
    """
    if palette_key not in ALL_PALETTES:
        palette_key = "spectral"  # Default value

    colors = ALL_PALETTES[palette_key]["colors"]
    width = 100 / len(colors)

    html = '<div style="display: flex; width: 100%; height: 20px; border-radius: 4px; overflow: hidden;">'
    for color in colors:
        html += f'<div style="width: {width}%; background-color: {color};"></div>'
    html += "</div>"

    return html

import pandas as pd
import os
from sqlalchemy import text
from utils.config import get_db_engine

def get_landcover_colormap():
    """
    Reads the landcover color map from the database.

    Returns:
    --------
    dict
        A dictionary mapping L2_CODE (str) to a hex color string (str).
        Returns an empty dictionary if the connection fails.
    """
    try:
        engine = get_db_engine()
        with engine.connect() as connection:
            query = text("SELECT l2_code, r, g, b FROM landcover_codes")
            result = connection.execute(query)
            color_map = {}
            for row in result:
                l2_code = str(row.l2_code)
                r, g, b = int(row.r), int(row.g), int(row.b)
                hex_color = f'#{r:02x}{g:02x}{b:02x}'
                color_map[l2_code] = hex_color
            return color_map
    except Exception as e:
        print(f"Warning: Could not fetch landcover color map from DB: {e}")
        return {}
