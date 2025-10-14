"""
분석 결과 시각화를 위한 색상 팔레트 모음
"""

# 스펙트럼 팔레트 (기본값)
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

# 빨강-노랑-초록 팔레트
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

# 지형 팔레트
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

# Viridis 팔레트
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

# Inferno 팔레트
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

# 회색조 팔레트
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

# 갈색 팔레트 (신규)
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

# 모든 팔레트를 딕셔너리에 저장
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
    색상 팔레트의 HTML 미리보기를 생성합니다.

    매개변수:
    -----------
    palette_key : str
        팔레트 키 (예: 'spectral', 'viridis')

    반환값:
    --------
    str
        팔레트 미리보기를 위한 HTML 코드
    """
    if palette_key not in ALL_PALETTES:
        palette_key = "spectral"  # 기본값

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
    데이터베이스에서 토지피복 색상 맵을 읽어옵니다.

    반환값:
    --------
    dict
        L2_CODE(str)를 16진수 색상 문자열(str)에 매핑하는 딕셔너리.
        연결에 실패하면 빈 딕셔너리를 반환합니다.
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

import streamlit as st

@st.cache_data
def get_palette(palette_name: str):
    """
    데이터베이스에서 특정 색상 팔레트를 가져옵니다.

    인자:
        palette_name (str): 가져올 팔레트의 이름 (예: 'elevation_10').

    반환값:
        list: 각 딕셔너리에 'bin_label'과 'hex_color'가 포함된 딕셔너리 리스트.
              실패 시 빈 리스트를 반환합니다.
    """
    try:
        engine = get_db_engine()
        with engine.connect() as connection:
            query = text("""
                SELECT bin_label, hex_color 
                FROM color_palettes 
                WHERE palette_name = :name 
                ORDER BY sequence
            """)
            result = connection.execute(query, {"name": palette_name})
            palette = [{'bin_label': row.bin_label, 'hex_color': row.hex_color} for row in result]
            return palette
    except Exception as e:
        st.error(f"DB에서 '{palette_name}' 팔레트를 불러오는 데 실패했습니다: {e}")
        return []