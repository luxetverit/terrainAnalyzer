import datetime
import gc
import io
import platform
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import zoom
from shapely import wkb
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from utils.color_palettes import get_landcover_colormap, get_palette
from utils.plot_helpers import (add_north_arrow, add_scalebar_vector,
                                adjust_ax_limits,
                                calculate_accurate_scalebar_params,
                                create_hillshade, create_padded_fig_ax,
                                draw_accurate_scalebar)
from utils.theme_util import apply_styles

# --- 최종 해결책: 모든 오류 수정을 집대성한 '완전판' Shapefile 생성 함수 ---
def create_shapefile_zip(gdf: gpd.GeoDataFrame, base_filename: str) -> io.BytesIO | None:
    """
    모든 오류 방지 로직을 통합하여 가장 안정적인 방식으로 Shapefile을 생성합니다.
    """
    if gdf.empty:
        return None

    gdf = gdf.copy()
    
    # 1. 분석 유형에 따른 핵심 컬럼만 선택
    analysis_type = base_filename.split('_')[0]
    essential_cols = {
        'soil': ['soilsy'], 'hsg': ['hg'],
        'landcover': ['l1_code', 'l1_name', 'l2_code', 'l2_name', 'l3_code', 'l3_name']
    }
    cols_to_keep = ['geometry']
    gdf_lower_columns = {col.lower(): col for col in gdf.columns}
    if analysis_type in essential_cols:
        for col in essential_cols[analysis_type]:
            if col in gdf_lower_columns:
                cols_to_keep.append(gdf_lower_columns[col])
        gdf = gdf[cols_to_keep]

    # 2. 데이터 및 지오메트리 정제
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid]
    if gdf.empty: return None

    # 3. 도형 분할 및 단순화 (오류 방지)
    gdf = gdf.explode(index_parts=True)
    try:
        gdf['geometry'] = gdf.simplify(tolerance=1.0, preserve_topology=True)
    except Exception:
        pass # 단순화 실패 시에도 계속 진행

    # 4. 가장 강력한 컬럼 이름 정리 (10자 제한, 소문자, 중복 제거)
    new_columns_map = {}
    processed_names = []
    for col in gdf.columns:
        if col.lower() == 'geometry': continue
        
        safe_name = col.lower()[:10]
        final_name = safe_name
        count = 1
        while final_name in processed_names:
            suffix = f"_{count}"
            base_name = safe_name[:10-len(suffix)]
            final_name = f"{base_name}{suffix}"
            count += 1
        
        processed_names.append(final_name)
        new_columns_map[col] = final_name
    
    gdf = gdf.rename(columns=new_columns_map)

    # 5. GeoPandas.to_file을 안전한 임시 파일 이름으로 사용
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_basename = "output"
        shp_path = str(Path(tmpdir) / f"{temp_basename}.shp")
        
        try:
            gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='cp949')

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in Path(tmpdir).glob(f'{temp_basename}.*'):
                    # .cpg와 .prj 파일은 제외하고 나머지 파일만 압축에 포함
                    if file_path.suffix.lower() not in ['.cpg', '.prj']:
                        arcname = file_path.name.replace(temp_basename, base_filename)
                        zip_file.write(file_path, arcname=arcname)
            zip_buffer.seek(0)
            return zip_buffer

        except Exception as e:
            st.error(f"GeoPandas로 SHP 파일 생성 중 최종 오류가 발생했습니다: {e}")
            return None


# --- 다운로드를 위한 기본 파일 이름 준비 ---
uploaded_file_name = st.session_state.get('uploaded_file_name', 'untitled')
base_filename = Path(uploaded_file_name).stem
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

plot_figures = {}
shp_buffers = {}

# --- 0. Matplotlib 글꼴 구성 ---
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 페이지 설정 및 스타일링 ---
st.set_page_config(page_title="분석 결과 - 지형 분석 서비스",
                   page_icon="📊",
                   layout="wide",
                   initial_sidebar_state="collapsed")
apply_styles()

# --- 2. 세션 상태 확인 ---
if 'dem_results' not in st.session_state:
    st.warning("분석 결과가 없습니다. 홈 페이지로 돌아가 분석을 먼저 실행해주세요.")
    if st.button("홈으로 돌아가기"):
        st.switch_page("app.py")
    st.stop()

# --- 3. 세션에서 데이터 로드 ---
dem_results = st.session_state.dem_results
selected_types = st.session_state.get('selected_analysis_types', [])
matched_sheets = st.session_state.get('matched_sheets', [])
analysis_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# --- 4. 페이지 헤더 ---
cols = st.columns([0.95, 0.05])
with cols[0]:
    st.markdown('''<div class="page-header" style="margin-top: -1.5rem;"><h1>분석 결과</h1><p>선택하신 항목에 대한 지형 분석 결과입니다.</p></div>''',
                unsafe_allow_html=True)
with cols[1]:
    if st.button("🏠", help="홈 화면으로 돌아갑니다.", use_container_width=True):
        if 'dem_results' in st.session_state:
            for analysis_type in st.session_state.dem_results:
                results = st.session_state.dem_results.get(analysis_type, {})
                tif_path = results.get('tif_path')
                if tif_path and Path(tif_path).exists():
                    try: Path(tif_path).unlink()
                    except OSError: pass
        for key in list(st.session_state.keys()):
            if key != 'upload_counter':
                del st.session_state[key]
        st.switch_page("app.py")

# --- 6. 2D 분석 결과 ---
st.markdown("### 📈 2D 상세 분석 결과")
analysis_map = {
    'elevation': {'title': "표고 분석", 'unit': "m", 'binned_label': "표고 구간별 면적", 'legend_title': '표고(Elevation)'},
    'slope': {'title': "경사 분석", 'unit': "°", 'binned_label': "경사 구간별 면적", 'legend_title': '경사(Slope)'},
    'aspect': {'title': "경사향 분석", 'unit': "°", 'binned_label': "경사향 구간별 면적", 'legend_title': '경사향(Aspect)'},
    'soil': {'title': "토양도 분석", 'unit': "m²", 'binned_label': "토양 종류별 면적", 'class_col': 'soilsy', 'legend_title': '토양도(Soilsy)'},
    'hsg': {'title': "수문학적 토양군", 'unit': "m²", 'binned_label': "HSG 등급별 면적", 'class_col': 'hg', 'legend_title': '토양군(HSG)'},
    'landcover': {'title': "토지피복도", 'unit': "m²", 'binned_label': "토지피복별 면적", 'class_col': 'l2_name', 'legend_title': '토지피복도(Landcover)'},
}
valid_selected_types = [t for t in selected_types if t in dem_results]

if not valid_selected_types:
    st.info("표시할 2D 분석 결과가 없습니다.")
else:
    for analysis_type in valid_selected_types:
        with st.container(border=True):
            st.markdown(f"### 📈 {analysis_map.get(analysis_type, {}).get('title', analysis_type)}")
            results = dem_results[analysis_type]
            stats = results.get('stats')
            grid = results.get('grid')
            gdf = results.get('gdf')

            if grid is not None:
                grid = grid.astype(np.float32)
                effective_pixel_size = st.session_state.get('pixel_size', 1.0)
                PIXEL_THRESHOLD = 5_000_000
                if grid.size > PIXEL_THRESHOLD:
                    downsample_factor = (PIXEL_THRESHOLD / grid.size) ** 0.5
                    st.info(f"ℹ️ 시각화 해상도를 원본의 {downsample_factor:.1%}로 자동 조정합니다.")
                    grid = zoom(grid, downsample_factor, order=1)
                    effective_pixel_size = effective_pixel_size / downsample_factor

            if stats: # RASTER DATA PLOTTING
                st.markdown("#### 요약 통계")
                cols = st.columns(3)
                cols[0].metric("최소값", f"{stats.get('min', 0):.2f} {analysis_map.get(analysis_type, {}).get('unit', '')}")
                cols[1].metric("최대값", f"{stats.get('max', 0):.2f} {analysis_map.get(analysis_type, {}).get('unit', '')}")
                cols[2].metric("평균값", f"{stats.get('mean', 0):.2f} {analysis_map.get(analysis_type, {}).get('unit', '')}")

                title = analysis_map.get(analysis_type, {}).get('title', analysis_type)
                with st.spinner(f"'{title}' 분석도 생성 중..."):
                    bins, labels, palette_name = results.get('bins'), results.get('labels'), results.get('palette_name')
                    if not all([bins, labels, palette_name]):
                        st.warning(f"'{title}'에 대한 시각화 정보를 생성할 수 없습니다.")
                        continue
                    
                    palette_data = get_palette(palette_name)
                    if not palette_data:
                        st.warning(f"'{palette_name}' 팔레트를 DB에서 찾을 수 없습니다.")
                        continue

                    if analysis_type == 'aspect':
                        flat_item = next((item for item in palette_data if item['bin_label'].strip().lower() == 'flat'), None)
                        if flat_item:
                            palette_data.remove(flat_item)
                            palette_data.insert(0, flat_item)

                    colors = [item['hex_color'] for item in palette_data]
                    cmap = ListedColormap(colors)
                    norm = BoundaryNorm(bins, cmap.N)

                    fig, ax = create_padded_fig_ax(figsize=(10, 8))
                    ax.set_title(title, fontsize=16, pad=20)

                    use_hillshade = st.toggle("음영기복도 중첩", value=True, key=f"hillshade_{analysis_type}") if analysis_type == 'elevation' else False
                    if use_hillshade:
                        hillshade = create_hillshade(grid)
                        rgba_data = cmap(norm(grid))
                        intensity = 0.7
                        hillshade_adjusted = 0.5 + hillshade * intensity
                        for i in range(3):
                            rgba_data[:, :, i] *= hillshade_adjusted
                        valid_mask = ~np.isnan(grid)
                        rgba_data[~valid_mask, 3] = 0
                        ax.imshow(np.clip(rgba_data, 0, 1), origin='upper')
                    else:
                        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm)

                    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
                    legend = ax.legend(handles=patches, title=analysis_map.get(analysis_type, {}).get('legend_title', '범례'),
                                       bbox_to_anchor=(0.1, 0.1), loc='lower left', bbox_transform=fig.transFigure,
                                       fontsize='small', frameon=True, framealpha=1, edgecolor='black')
                    legend.get_title().set_fontweight('bold')
                    for patch in legend.get_patches():
                        patch.set_edgecolor('black'); patch.set_linewidth(0.7)

                    adjust_ax_limits(ax); add_north_arrow(ax)
                    scale_params = calculate_accurate_scalebar_params(effective_pixel_size, grid.shape, 50, fig, ax)
                    draw_accurate_scalebar(fig, ax, effective_pixel_size, scale_params, grid.shape)
                    ax.axis('off')

                    st.pyplot(fig)
                    plot_figures[analysis_type] = fig

            elif gdf is not None and not gdf.empty: # VECTOR DATA PLOTTING
                title = analysis_map.get(analysis_type, {}).get('title', analysis_type)
                with st.spinner(f"'{title}' 분석도 생성 중..."):
                    fig, ax = create_padded_fig_ax(figsize=(10, 10))
                    type_info = analysis_map.get(analysis_type, {})
                    class_col = type_info.get('class_col')

                    if analysis_type == 'landcover':
                        color_map = get_landcover_colormap()
                        if color_map and 'l2_code' in gdf.columns and 'l2_name' in gdf.columns:
                            gdf['plot_color'] = gdf['l2_code'].map(color_map)
                            gdf.plot(ax=ax, color=gdf['plot_color'].fillna('#FFFFFF'), linewidth=0.5, edgecolor='k')
                            unique_cats = gdf[['l2_code', 'l2_name']].drop_duplicates().sort_values(by='l2_code')
                            patches = [mpatches.Patch(color=color_map.get(code, '#FFFFFF'), label=f'{name}') for _, (code, name) in unique_cats.iterrows()]
                            if patches:
                                n_cols = (len(patches) + 19) // 20
                                legend = ax.legend(handles=patches, title=type_info.get('legend_title', '분류'),
                                                   bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=n_cols)
                                for patch in legend.get_patches():
                                    patch.set_edgecolor('black'); patch.set_linewidth(0.7)
                        else:
                            gdf.plot(column=class_col, ax=ax, legend=True, categorical=True,
                                     legend_kwds={'title': type_info.get('legend_title', '분류'), 'bbox_to_anchor': (1.05, 1), 'loc': 'upper left'})
                    else:
                        if class_col and class_col in gdf.columns:
                            gdf_plot = gdf[gdf[class_col].notna()].copy()
                            unique_cats = sorted(gdf_plot[class_col].unique())
                            num_cats = len(unique_cats)
                            cmap = plt.cm.get_cmap('tab20' if num_cats <= 20 else 'viridis', num_cats)
                            colors = cmap.colors if num_cats <= 20 else cmap(np.linspace(0, 1, num_cats))
                            color_map = {cat: color for cat, color in zip(unique_cats, colors)}
                            gdf_plot['plot_color'] = gdf_plot[class_col].map(color_map)
                            gdf_plot.plot(ax=ax, color=gdf_plot['plot_color'], linewidth=0.5, edgecolor='k')
                            patches = [mpatches.Patch(color=color, label=cat) for cat, color in color_map.items()]
                            if patches:
                                n_cols = (len(patches) + 19) // 20
                                legend = ax.legend(handles=patches, title=type_info.get('legend_title', '분류'),
                                                   bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=n_cols)
                                for patch in legend.get_patches():
                                    patch.set_edgecolor('black'); patch.set_linewidth(0.7)
                        else:
                            gdf.plot(ax=ax)

                    adjust_ax_limits(ax); add_north_arrow(ax)
                    x_min, x_max = ax.get_xlim()
                    map_width_m = x_max - x_min
                    proxy_img_width_px = int(10 * 150)
                    effective_pixel_size = map_width_m / proxy_img_width_px if proxy_img_width_px > 0 else 1.0
                    proxy_img_shape = (int(10 * 150), proxy_img_width_px)
                    scale_params = calculate_accurate_scalebar_params(effective_pixel_size, proxy_img_shape, 50, fig, ax)
                    draw_accurate_scalebar(fig, ax, effective_pixel_size, scale_params, proxy_img_shape)
                    ax.axis('off')

                    st.pyplot(fig)
                    plot_figures[analysis_type] = fig

                    shp_zip_buffer = create_shapefile_zip(
                        gdf, f"{analysis_type}_{base_filename}"
                    )
                    if shp_zip_buffer:
                        shp_buffers[analysis_type] = {
                            "buffer": shp_zip_buffer,
                            "title": type_info.get('title', analysis_type)
                        }
            else:
                st.info("시각화할 2D 데이터가 없습니다.")
            st.markdown("---")

# --- 7. 요약 및 최종 다운로드 ---
st.markdown("### 📋 상세 분석 보고서")

with st.spinner("보고서 생성중..."):
    # --- TXT 요약(표시용) 및 CSV(다운로드용) 데이터 생성 ---
    summary_lines = []
    csv_data = []

    pixel_size = st.session_state.get('pixel_size', 1.0)
    area_per_pixel = pixel_size * pixel_size

    # --- 보고서 헤더의 총 면적 계산 ---
    report_total_area_m2 = 0
    if 'gdf' in st.session_state and not st.session_state.gdf.empty:
        if 'area' not in st.session_state.gdf.columns:
            st.session_state.gdf['area'] = st.session_state.gdf.geometry.area
        report_total_area_m2 = st.session_state.gdf.area.sum()
    else:
        area_source_type = next((t for t in valid_selected_types if dem_results.get(t)), None)
        if area_source_type:
            results = dem_results[area_source_type]
            stats = results.get('stats')
            gdf = results.get('gdf')
            if stats:
                report_total_area_m2 = stats.get('area', 0) * area_per_pixel
            elif gdf is not None and not gdf.empty:
                if 'area' not in gdf.columns:
                    gdf['area'] = gdf.geometry.area
                report_total_area_m2 = gdf.area.sum()

    # 일반 정보
    summary_lines.append(f"분석 일시: {analysis_date}")
    summary_lines.append(f"분석 대상: {st.session_state.get('uploaded_file_name', 'N/A')}")
    csv_data.append({'분석 구분': '기본 정보', '항목': '분석 일시', '값': analysis_date, '단위': '', '면적(m²)': '', '비율(%)': ''})
    csv_data.append({'분석 구분': '기본 정보', '항목': '분석 대상', '값': st.session_state.get('uploaded_file_name', 'N/A'), '단위': '', '면적(m²)': '', '비율(%)': ''})

    summary_lines.append(f"사용된 도엽: {len(matched_sheets)}개 ({', '.join(matched_sheets)})" if len(matched_sheets) <= 20 else f"사용된 도엽: {len(matched_sheets)}개")
    csv_data.append({'분석 구분': '기본 정보', '항목': '사용된 도엽', '값': f"{len(matched_sheets)}개 ({', '.join(matched_sheets)})" if len(matched_sheets) <= 20 else f"{len(matched_sheets)}개", '단위': '', '면적(m²)': '', '비율(%)': ''})

    summary_lines.append(f"총 분석 면적: {int(report_total_area_m2):,} m²")
    csv_data.append({'분석 구분': '기본 정보', '항목': '총 분석 면적', '값': '', '단위': 'm²', '면적(m²)': f"{int(report_total_area_m2):,}", '비율(%)': ''})
    summary_lines.append("")

    # 분석별 정보
    for analysis_type in valid_selected_types:
        stats = dem_results[analysis_type].get('stats')
        binned_stats = dem_results[analysis_type].get('binned_stats')
        gdf = dem_results[analysis_type].get('gdf')
        title_info = analysis_map.get(analysis_type, {})
        title = title_info.get('title', analysis_type)

        summary_lines.append(f"--- {title} ---")
        total_area_m2 = 0
        if stats:
            total_area_m2 = stats.get('area', 0) * area_per_pixel
            unit = title_info.get('unit', '')
            summary_lines.extend([f"- 최소값: {stats.get('min', 0):.2f} {unit}", f"- 최대값: {stats.get('max', 0):.2f} {unit}", f"- 평균값: {stats.get('mean', 0):.2f} {unit}"])
            csv_data.extend([
                {'분석 구분': title, '항목': '최소값', '값': f"{stats.get('min', 0):.2f}", '단위': unit, '면적(m²)': '', '비율(%)': ''},
                {'분석 구분': title, '항목': '최대값', '값': f"{stats.get('max', 0):.2f}", '단위': unit, '면적(m²)': '', '비율(%)': ''},
                {'분석 구분': title, '항목': '평균값', '값': f"{stats.get('mean', 0):.2f}", '단위': unit, '면적(m²)': '', '비율(%)': ''}
            ])
        elif gdf is not None and not gdf.empty:
            if 'area' not in gdf.columns: gdf['area'] = gdf.geometry.area
            total_area_m2 = gdf.area.sum()

        if binned_stats:
            summary_lines.append(f"\n[{title_info.get('binned_label', '구간별 통계')}]")
            for row in binned_stats:
                binned_area_m2 = row['area'] * area_per_pixel
                percentage = (binned_area_m2 / total_area_m2 * 100) if total_area_m2 > 0 else 0
                summary_lines.append(f"- {row['bin_range']}: {int(binned_area_m2):,} m² ({percentage:.1f} %)")
                csv_data.append({'분석 구분': title, '항목': row['bin_range'], '값': '', '단위': '', '면적(m²)': f"{int(binned_area_m2):,}", '비율(%)': f"{percentage:.1f}"})

        if gdf is not None and not gdf.empty:
            class_col = title_info.get('class_col')
            if class_col and class_col in gdf.columns:
                dissolved_gdf = gdf.dissolve(by=class_col, aggfunc={'area': 'sum'}) if 'area' in gdf.columns else gdf.dissolve(by=class_col)
                if 'area' not in dissolved_gdf.columns: dissolved_gdf['area'] = dissolved_gdf.geometry.area
                dissolved_gdf = dissolved_gdf.sort_values(by='area', ascending=False)
                summary_lines.append(f"\n[{title_info.get('binned_label', '종류별 통계')}]")
                for item, row in dissolved_gdf.iterrows():
                    area = row['area']
                    percentage = (area / total_area_m2 * 100) if total_area_m2 > 0 else 0
                    summary_lines.append(f"- {item}: {int(area):,} m² ({percentage:.1f} %)")
                    csv_data.append({'분석 구분': title, '항목': item, '값': '', '단위': '', '면적(m²)': f"{int(area):,}", '비율(%)': f"{percentage:.1f}"})

        summary_lines.append("")

    # --- 표시 텍스트 및 CSV 문자열 생성 ---
    summary_text = "\n".join(summary_lines)
    report_df = pd.DataFrame(csv_data)
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    csv_string = csv_buffer.getvalue()

# 요약 텍스트 영역 표시
st.text_area("", summary_text, height=400)

# --- 최종 ZIP 생성 및 다운로드 버튼 ---
with st.spinner("다운로드 파일 생성중..."):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(f"analysis_summary_{timestamp}.csv", csv_string)
        for analysis_type, fig in plot_figures.items():
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            zip_file.writestr(f"{analysis_type}_analysis.png", img_buffer.getvalue())
            plt.close(fig)
        for analysis_type in valid_selected_types:
            results = dem_results.get(analysis_type, {})
            tif_path = results.get('tif_path')
            if tif_path and Path(tif_path).exists():
                zip_file.write(tif_path, arcname=f"{analysis_type}.tif")
        for analysis_type, data in shp_buffers.items():
            with zipfile.ZipFile(data["buffer"], 'r') as inner_zip:
                for file_info in inner_zip.infolist():
                    zip_file.writestr(file_info.filename, inner_zip.read(file_info.filename))
    zip_buffer.seek(0)

# --- 8. 최종 다운로드 섹션 ---
st.markdown("### 📥 다운로드")
download_items = []
main_zip_label = "📥 시각화자료+분석보고서+SHP (ZIP)"
main_zip_help = "분석 리포트, 모든 분석도(PNG), 모든 원본 분석 파일(TIF), 벡터 데이터(SHP)를 한번에 다운로드합니다."

download_items.append({
    "label": main_zip_label, "data": zip_buffer,
    "file_name": f"analysis_results_{base_filename}_{timestamp}.zip",
    "mime": "application/zip", "key": "main_zip_download", "help": main_zip_help
})

if download_items:
    cols = st.columns(len(download_items))
    for i, item in enumerate(download_items):
        with cols[i]:
            st.download_button(
                label=item["label"],
                data=item["data"],
                file_name=item["file_name"],
                mime=item["mime"],
                key=item["key"],
                use_container_width=True,
                help=item.get("help", "")
            )

st.markdown("")
if st.button("새로운 분석 시작하기", use_container_width=True):
    # ... (Session state cleanup) ...
    st.switch_page("app.py")