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
        print(f"도엽 인덱스 검색 시작: 입력 좌표계 EPSG:{epsg_code}, 폴리곤 개수: {len(gdf)}")

        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError(f"도엽 인덱스 파일이 존재하지 않습니다: {INDEX_FILE}")

        map_index = gpd.read_file(INDEX_FILE)
        print(f"도엽 인덱스 파일 로드 완료: {len(map_index)}개 도엽, 좌표계: {map_index.crs}")

        # --- CRS Transformation (Robust Version) ---
        gdf.crs = f"EPSG:{epsg_code}"
        print(f"입력 GDF의 좌표계를 {gdf.crs}로 설정했습니다.")
        print(f"변환 전 원본 경계 (EPSG:{epsg_code}): {gdf.total_bounds}")

        gdf_converted = gdf.to_crs(map_index.crs)
        print(f"도엽 인덱스 좌표계({map_index.crs})로 변환을 시도합니다.")
        print(f"변환 후 경계 ({map_index.crs}): {gdf_converted.total_bounds}")
        
        if np.allclose(gdf.total_bounds, gdf_converted.total_bounds) and gdf.crs != gdf_converted.crs:
            print("경고: 좌표계 변환 후에도 경계 좌표가 거의 동일합니다. 입력 좌표계가 올바른지 확인해주세요.")

        gdf = gdf_converted
        # --- End of CRS Transformation ---

        total_bounds = gdf.total_bounds
        overlaps = []
        info = []

        for idx, geom in enumerate(gdf.geometry):
            if geom is None or geom.is_empty:
                continue
            
            potential_matches = map_index.cx[geom.bounds[0]:geom.bounds[2], geom.bounds[1]:geom.bounds[3]]
            if potential_matches.empty:
                continue

            actual_matches = potential_matches[potential_matches.intersects(geom)]
            
            for _, row in actual_matches.iterrows():
                sheet_id = row['MAPIDCD_NO']
                if sheet_id not in overlaps:
                    overlaps.append(sheet_id)
                
                overlap_geom = geom.intersection(row.geometry)
                info.append({
                    'sheet_id': sheet_id,
                    'overlap_area_sqm': overlap_geom.area
                })

        unique_overlaps = sorted(list(set(overlaps)))
        print(f"최종 겹치는 도엽 수: {len(unique_overlaps)}")

        if not unique_overlaps:
            return {'matched_sheets': [], 'mapsheet_info': [], 'preview_image': None}

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

        preview_image = create_preview_image(gdf, map_index, all_sheet_ids, unique_overlaps)

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