import sys
from pathlib import Path
import platform
import pyproj

# --- PROJ Data Directory Configuration (v2 - More Robust) ---
# This block tries multiple strategies to find the PROJ data directory,
# which is essential for accurate CRS transformations.
try:
    found_path = None
    # Strategy 1: Use pyproj's own mechanism
    try:
        pyproj_data_dir = Path(pyproj.datadir.get_data_dir())
        if pyproj_data_dir.exists():
            found_path = pyproj_data_dir
            print(f"pyproj.datadir.get_data_dir() found path: {found_path}")
    except Exception:
        print("pyproj.datadir.get_data_dir() failed, trying other methods.")

    # Strategy 2: Check standard Conda path if first failed
    if not found_path:
        conda_prefix = Path(sys.prefix)
        if platform.system() == "Windows":
            conda_path = conda_prefix / "Library" / "share" / "proj"
        else:
            conda_path = conda_prefix / "share" / "proj"
        if conda_path.exists():
            found_path = conda_path
            print(f"Found PROJ data in Conda path: {found_path}")

    # Strategy 3: Check standard venv/pip site-packages path
    if not found_path:
        for path in sys.path:
            if "site-packages" in path:
                venv_path = Path(path) / "pyproj" / "proj_dir" / "share" / "proj"
                if venv_path.exists():
                    found_path = venv_path
                    print(f"Found PROJ data in venv path: {found_path}")
                    break

    if found_path:
        pyproj.datadir.set_data_dir(str(found_path))
        print(f"SUCCESS: pyproj data directory set to: {found_path}")
    else:
        print("WARNING: Could not find pyproj data directory. CRS transformations may be incorrect.")

except Exception as e:
    print(f"CRITICAL: An unexpected error occurred while setting pyproj data directory: {e}")
# --- End of Configuration ---

"""
인덱스 파일(GPKG)을 사용하여 업로드된 경계 파일과 겹치는 도엽 번호를 찾는 유틸리티
"""

import os
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import streamlit as st


INDEX_FILE = 'attached_assets/TN_MAPINDX_5K.gpkg'


def find_overlapping_sheets(gdf, epsg_code):
    try:
        print("\n--- 도엽 인덱스 검색 디버깅 시작 ---")
        print(f"입력된 EPSG 코드: {epsg_code}")
        print(f"입력된 GDF 폴리곤 개수: {len(gdf)}")
        print(f"입력된 GDF의 초기 CRS: {gdf.crs}")
        print(f"입력된 GDF의 초기 경계 (total_bounds): {gdf.total_bounds}")

        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError(f"도엽 인덱스 파일({INDEX_FILE})을 찾을 수 없습니다.")

        map_index = gpd.read_file(INDEX_FILE)
        print(f"\n[1] 도엽 인덱스 파일 로드")
        print(f"  - 파일 경로: {INDEX_FILE}")
        print(f"  - 총 도엽 수: {len(map_index)}")
        print(f"  - 도엽 좌표계 (CRS): {map_index.crs}")
        print(f"  - 도엽 전체 경계 (total_bounds): {map_index.total_bounds}")

        # --- CRS 처리 ---
        print(f"\n[2] 입력 GDF 좌표계 설정 및 변환")
        gdf_transformed = gdf.set_crs(f"EPSG:{epsg_code}", allow_override=True).to_crs(map_index.crs)
        print(f"  - 단계: 입력 GDF에 원본 좌표계(EPSG:{epsg_code}) 설정 후 도엽 인덱스 좌표계({map_index.crs})로 변환 완료.")
        print(f"  - 변환 후 CRS: {gdf_transformed.crs}")
        print(f"  - 변환 후 경계: {gdf_transformed.total_bounds}")

        # --- 경계 좌표 비교 ---
        source_bounds = gdf.total_bounds # Original bounds before any CRS operation
        transformed_bounds = gdf_transformed.total_bounds
        # Compare string representations of CRS to avoid issues with object comparison
        if np.allclose(source_bounds, transformed_bounds) and str(gdf.set_crs(f"EPSG:{epsg_code}").crs) != str(map_index.crs):
            print("\n[경고] 좌표계 변환이 일어났으나, 경계 좌표가 거의 동일합니다.")
            print("       입력하신 EPSG 코드가 파일의 실제 좌표계와 일치하는지 확인해주세요.")
            print("       (또는 pyproj 데이터 디렉토리 설정이 잘못되었을 수 있습니다.)")

        # --- 공간 조인 (Overlap Detection) ---
        print("\n[3] 공간 조인(sjoin)으로 겹치는 도엽 검색")
        overlapping_sheets = gpd.sjoin(map_index, gdf_transformed, how="inner", predicate="intersects")

        if overlapping_sheets.empty:
            print("  - 결과: 겹치는 도엽을 찾지 못했습니다.")
            print("--- 도엽 인덱스 검색 디버깅 종료 ---\n")
            return {'matched_sheets': [], 'mapsheet_info': [], 'preview_image': None}

        print(f"  - 결과: {len(overlapping_sheets)}개의 겹치는 항목을 찾았습니다.")
        unique_overlaps = sorted(overlapping_sheets['MAPIDCD_NO'].unique())
        print(f"  - 고유 도엽 ID: {unique_overlaps}")
        
        info = []
        for _, row in overlapping_sheets.iterrows():
             info.append({
                'sheet_id': row['MAPIDCD_NO'],
                'overlap_area_sqm': 'N/A (sjoin used)'
            })

        print(f"최종 겹치는 도엽 수: {len(unique_overlaps)}")
        print("--- 도엽 인덱스 검색 디버깅 종료 ---\n")

        extended_overlaps = set(unique_overlaps)
        for sheet_id in unique_overlaps:
            try:
                if len(sheet_id) >= 8:
                    base = sheet_id[:-2]
                    row_col = int(sheet_id[-2:])
                    row, col = row_col // 10, row_col % 10
                    for r_off in [-1, 0, 1]:
                        for c_off in [-1, 0, 1]:
                            if r_off == 0 and c_off == 0: continue
                            adj_row, adj_col = row + r_off, col + c_off
                            if 0 <= adj_row <= 9 and 0 <= adj_col <= 9:
                                adj_row_col = adj_row * 10 + adj_col
                                adjacent_sheet_id = f"{base}{adj_row_col:02d}"
                                extended_overlaps.add(adjacent_sheet_id)
            except Exception as e:
                print(f"인근 도엽 계산 중 오류 (도엽: {sheet_id}): {e}")
        
        all_sheet_ids = sorted(list(extended_overlaps))
        print(f"인근 도엽 포함 후 총 도엽 수: {len(all_sheet_ids)}")

        # Pass the transformed GDF to the preview function
        preview_image = create_preview_image(gdf_transformed, map_index, all_sheet_ids, unique_overlaps)

        return {
            'matched_sheets': all_sheet_ids,
            'mapsheet_info': info,
            'preview_image': preview_image,
        }

    except Exception as e:
        import traceback
        print(f"도엽 찾기 중 심각한 오류 발생: {e}")
        print(traceback.format_exc())
        return {'matched_sheets': [], 'mapsheet_info': [], 'preview_image': None}


def create_preview_image(gdf, map_index, matched_sheets, original_matched_sheets=None):
    assets_dir = Path('assets')
    assets_dir.mkdir(exist_ok=True)
    image_path = assets_dir / 'map_index_preview.png'
    
    dark_mode = st.session_state.get('dark_mode', False)
    
    background_color = '#0E1117' if dark_mode else 'white'
    text_color = 'white' if dark_mode else 'black'
    # ... (rest of the function is styling and can be omitted for brevity) ...
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    matching_index = map_index[map_index['MAPIDCD_NO'].isin(matched_sheets)]
    if not matching_index.empty:
        bounds = matching_index.total_bounds
        x_margin = (bounds[2] - bounds[0]) * 0.1
        y_margin = (bounds[3] - bounds[1]) * 0.1
        plot_bounds = [bounds[0] - x_margin, bounds[1] - y_margin, bounds[2] + x_margin, bounds[3] + y_margin]
        ax.set_xlim(plot_bounds[0], plot_bounds[2])
        ax.set_ylim(plot_bounds[1], plot_bounds[3])

    # Plotting logic...
    gdf.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2)
    matching_index.plot(ax=ax, edgecolor='blue', facecolor='lightblue', alpha=0.5)

    plt.savefig(image_path, dpi=200, bbox_inches='tight', facecolor=background_color)
    plt.close(fig)

    return str(image_path)