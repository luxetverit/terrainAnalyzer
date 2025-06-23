"""
토지피복도 시각화 유틸리티
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from matplotlib.colors import ListedColormap
import os
import json

# 샘플 L2_CODE에 대한 RGB 값 매핑 (환경부 토지피복도 분류체계 기반)
L2_CODE_COLORS = {
    "110": [255, 0, 0],       # 주거지역
    "120": [255, 150, 0],     # 공업지역
    "130": [255, 195, 0],     # 상업지역
    "140": [220, 220, 0],     # 문화·체육·휴양시설
    "150": [0, 255, 0],       # 교통지역
    "210": [0, 230, 130],     # 논
    "220": [0, 200, 255],     # 밭
    "230": [0, 0, 255],       # 시설재배지
    "240": [170, 150, 240],   # 과수원
    "250": [115, 0, 200],     # 기타재배지
    "310": [0, 115, 0],       # 활엽수림
    "320": [0, 85, 0],        # 침엽수림
    "330": [50, 115, 0],      # 혼효림
    "410": [150, 150, 150],   # 자연초지
    "420": [170, 170, 170],   # 인공초지
    "510": [90, 170, 200],    # 내륙습지
    "520": [0, 180, 220],     # 연안습지
    "610": [180, 180, 255],   # 자연나지
    "620": [100, 100, 100],   # 기타나지
    "710": [0, 70, 200],      # 내륙수
    "720": [0, 90, 255],      # 해양수
}

def normalize_rgb(rgb_list):
    """RGB 값을 0-1 범위로 정규화"""
    return [r/255 for r in rgb_list]

def create_landcover_colormap():
    """토지피복 코드에 대한 컬러맵 생성"""
    colors = [normalize_rgb(L2_CODE_COLORS[code]) for code in L2_CODE_COLORS]
    return ListedColormap(colors)

def generate_sample_landcover_image(size=(400, 400)):
    """
    토지피복도 샘플 이미지 생성
    
    Parameters:
    -----------
    size : tuple
        이미지 크기 (가로, 세로)
        
    Returns:
    --------
    tuple
        (raster_array, colormap, codes)
    """
    # 샘플 토지피복 코드 목록
    codes = list(L2_CODE_COLORS.keys())
    
    # 0부터 (코드 수 - 1) 범위의 랜덤 데이터 생성
    landcover_data = np.random.randint(0, len(codes), size=size)
    
    # 가우시안 필터로 부드럽게 (토지피복 경계를 좀 더 자연스럽게)
    from scipy.ndimage import gaussian_filter
    landcover_data = gaussian_filter(landcover_data.astype(float), sigma=5)
    landcover_data = np.round(landcover_data).astype(int) % len(codes)
    
    # 컬러맵 생성
    cmap = create_landcover_colormap()
    
    return landcover_data, cmap, codes

def create_landcover_preview(landcover_data=None, colormap=None, codes=None, title="토지피복도 분석"):
    """
    토지피복도 미리보기 이미지 생성
    
    Parameters:
    -----------
    landcover_data : numpy.ndarray
        토지피복 래스터 데이터
    colormap : matplotlib.colors.ListedColormap
        토지피복 코드 컬러맵
    codes : list
        토지피복 코드 목록
    title : str
        이미지 제목
        
    Returns:
    --------
    matplotlib.figure.Figure
        생성된 도표
    """
    if landcover_data is None or colormap is None or codes is None:
        landcover_data, colormap, codes = generate_sample_landcover_image()
    
    # 도표 생성
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 데이터 시각화
    im = ax.imshow(landcover_data, cmap=colormap, interpolation='nearest')
    
    # 컬러바 생성 (너무 많아서 컬러바에는 일부만 표시)
    sample_indices = np.linspace(0, len(codes)-1, min(10, len(codes))).astype(int)
    sample_codes = [codes[i] for i in sample_indices]
    
    cbar = fig.colorbar(im, ax=ax, ticks=sample_indices)
    cbar.set_label('토지피복 유형')
    cbar.ax.set_yticklabels(sample_codes)
    
    # 제목 및 축 레이블 설정
    ax.set_title(title)
    ax.set_xlabel('X 좌표')
    ax.set_ylabel('Y 좌표')
    
    # 격자 없애기
    ax.grid(False)
    
    return fig

def save_sample_landcover_image(palette_key="landcover", output_dir="assets/palette_samples"):
    """
    토지피복도 샘플 이미지 저장
    
    Parameters:
    -----------
    palette_key : str
        팔레트 키
    output_dir : str
        출력 디렉토리
        
    Returns:
    --------
    str
        저장된 파일 경로
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 랜덤 시드 고정
    np.random.seed(42)
    
    # 샘플 이미지 생성
    landcover_data, colormap, codes = generate_sample_landcover_image()
    
    # 미리보기 생성
    fig = create_landcover_preview(landcover_data, colormap, codes)
    
    # 저장 경로
    file_path = os.path.join(output_dir, f"landcover_{palette_key}.png")
    
    # 저장
    plt.savefig(file_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return file_path

def get_landcover_legend_html():
    """
    토지피복도 범례 HTML 생성
    
    Returns:
    --------
    str
        HTML 코드
    """
    # 토지피복 분류 설명 (코드별 이름)
    code_names = {
        "110": "주거지역",
        "120": "공업지역",
        "130": "상업지역",
        "140": "문화·체육·휴양시설",
        "150": "교통지역",
        "210": "논",
        "220": "밭",
        "230": "시설재배지",
        "240": "과수원",
        "250": "기타재배지",
        "310": "활엽수림",
        "320": "침엽수림",
        "330": "혼효림",
        "410": "자연초지",
        "420": "인공초지",
        "510": "내륙습지",
        "520": "연안습지",
        "610": "자연나지",
        "620": "기타나지",
        "710": "내륙수",
        "720": "해양수",
    }
    
    html = '<div style="display: flex; flex-direction: column; gap: 5px; max-height: 300px; overflow-y: auto; width: 100%;">'
    
    for code, rgb in L2_CODE_COLORS.items():
        color = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
        name = code_names.get(code, code)
        html += f'<div style="display: flex; align-items: center; gap: 5px;">'
        html += f'<div style="width: 20px; height: 20px; background-color: {color}; border: 1px solid #ccc;"></div>'
        html += f'<span>{code} - {name}</span>'
        html += '</div>'
    
    html += '</div>'
    
    return html

# 토지피복 컬러맵 생성을 위한 함수
def get_landcover_palette_preview_html():
    """
    토지피복도 팔레트 미리보기 HTML 생성
    
    Returns:
    --------
    str
        HTML 코드
    """
    # 색상 리스트 추출
    colors = [f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})" for rgb in L2_CODE_COLORS.values()]
    width = 100 / len(colors)
    
    html = '<div style="display: flex; width: 100%; height: 20px; border-radius: 4px; overflow: hidden;">'
    for color in colors:
        html += f'<div style="width: {width}%; background-color: {color};"></div>'
    html += '</div>'
    
    return html

# 메인 실행 (단독 실행 시)
if __name__ == "__main__":
    # 샘플 이미지 생성 및 저장
    output_path = save_sample_landcover_image()
    print(f"토지피복도 샘플 이미지가 저장되었습니다: {output_path}")