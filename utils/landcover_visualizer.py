"""
토지피복도 시각화 유틸리티
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from sqlalchemy import create_engine, text
from utils.config import get_db_engine

def get_landcover_data_from_db():
    """
    Fetches landcover data (code, name, color) from the database.
    """
    try:
        engine = get_db_engine()
        with engine.connect() as connection:
            query = text("SELECT l2_code, l2_name, r, g, b FROM landcover_codes ORDER BY l2_code")
            result = connection.execute(query)
            landcover_data = []
            for row in result:
                landcover_data.append({
                    "l2_code": str(row.l2_code),
                    "l2_name": str(row.l2_name),
                    "rgb": [int(row.r), int(row.g), int(row.b)]
                })
            return landcover_data
    except Exception as e:
        print(f"Warning: Could not fetch landcover data from DB: {e}")
        return []

def normalize_rgb(rgb_list):
    """RGB 값을 0-1 범위로 정규화"""
    return [r / 255 for r in rgb_list]

def create_landcover_colormap():
    """토지피복 코드에 대한 컬러맵 생성"""
    landcover_data = get_landcover_data_from_db()
    colors = [normalize_rgb(item['rgb']) for item in landcover_data]
    return ListedColormap(colors)

def generate_sample_landcover_image(size=(400, 400)):
    """
    토지피복도 샘플 이미지 생성
    """
    landcover_data = get_landcover_data_from_db()
    codes = [item['l2_code'] for item in landcover_data]

    # 0부터 (코드 수 - 1) 범위의 랜덤 데이터 생성
    random_data = np.random.randint(0, len(codes), size=size)

    # 가우시안 필터로 부드럽게 (토지피복 경계를 좀 더 자연스럽게)
    from scipy.ndimage import gaussian_filter

    random_data = gaussian_filter(random_data.astype(float), sigma=5)
    random_data = np.round(random_data).astype(int) % len(codes)

    # 컬러맵 생성
    cmap = create_landcover_colormap()

    return random_data, cmap, codes

def create_landcover_preview(
    landcover_data=None, colormap=None, codes=None, title="토지피복도 분석"
):
    """
    토지피복도 미리보기 이미지 생성
    """
    if landcover_data is None or colormap is None or codes is None:
        landcover_data, colormap, codes = generate_sample_landcover_image()

    # 도표 생성
    fig, ax = plt.subplots(figsize=(10, 8))

    # 데이터 시각화
    im = ax.imshow(landcover_data, cmap=colormap, interpolation="nearest")

    # 컬러바 생성 (너무 많아서 컬러바에는 일부만 표시)
    sample_indices = np.linspace(0, len(codes) - 1, min(10, len(codes))).astype(int)
    sample_codes = [codes[i] for i in sample_indices]

    cbar = fig.colorbar(im, ax=ax, ticks=sample_indices)
    cbar.set_label("토지피복 유형")
    cbar.ax.set_yticklabels(sample_codes)

    # 제목 및 축 레이블 설정
    ax.set_title(title)
    ax.set_xlabel("X 좌표")
    ax.set_ylabel("Y 좌표")

    # 격자 없애기
    ax.grid(False)

    return fig

def save_sample_landcover_image(
    palette_key="landcover", output_dir="assets/palette_samples"
):
    """
    토지피복도 샘플 이미지 저장
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
    plt.savefig(file_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return file_path

def get_landcover_legend_html():
    """
    토지피복도 범례 HTML 생성
    """
    landcover_data = get_landcover_data_from_db()
    html = '<div style="display: flex; flex-direction: column; gap: 5px; max-height: 300px; overflow-y: auto; width: 100%;">'

    for item in landcover_data:
        color = f"rgb({item['rgb'][0]}, {item['rgb'][1]}, {item['rgb'][2]})"
        name = item['l2_name']
        code = item['l2_code']
        html += f'<div style="display: flex; align-items: center; gap: 5px;">'
        html += f'<div style="width: 20px; height: 20px; background-color: {color}; border: 1px solid #ccc;"></div>'
        html += f"<span>{code} - {name}</span>"
        html += "</div>"

    html += "</div>"

    return html

def get_landcover_palette_preview_html():
    """
    토지피복도 팔레트 미리보기 HTML 생성
    """
    landcover_data = get_landcover_data_from_db()
    colors = [f"rgb({item['rgb'][0]}, {item['rgb'][1]}, {item['rgb'][2]})" for item in landcover_data]
    width = 100 / len(colors) if colors else 100

    html = '<div style="display: flex; width: 100%; height: 20px; border-radius: 4px; overflow: hidden;">'
    for color in colors:
        html += f'<div style="width: {width}%; background-color: {color};"></div>'
    html += "</div>"

    return html

# 메인 실행 (단독 실행 시)
if __name__ == "__main__":
    # 샘플 이미지 생성 및 저장
    output_path = save_sample_landcover_image()
    print(f"토지피복도 샘플 이미지가 저장되었습니다: {output_path}")