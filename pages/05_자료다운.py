import datetime
import gc
import io
import platform

import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Polygon, Rectangle
from scipy.ndimage import zoom

from utils.color_palettes import get_landcover_colormap, get_palette
from utils.plot_helpers import (add_north_arrow, add_scalebar_vector,
                                adjust_ax_limits,
                                calculate_accurate_scalebar_params,
                                create_hillshade, create_padded_fig_ax,
                                draw_accurate_scalebar, generate_aspect_bins,
                                generate_custom_intervals,
                                generate_slope_intervals)
from utils.theme_util import apply_styles

# --- 0. Matplotlib Font Configuration ---
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    # For Linux, ensure 'NanumGothic' is installed.
    # sudo apt-get install -y fonts-nanum*
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# --- 1. Page Configuration and Styling ---
st.set_page_config(page_title="분석 결과 - 지형 분석 서비스",
                   page_icon="📊",
                   layout="wide",
                   initial_sidebar_state="collapsed")
apply_styles()

# --- 2. Session State Check ---
if 'dem_results' not in st.session_state:
    st.warning("분석 결과가 없습니다. 홈 페이지로 돌아가 분석을 먼저 실행해주세요.")
    if st.button("홈으로 돌아가기"):
        st.switch_page("app.py")
    st.stop()

# --- 3. Data Loading from Session ---
dem_results = st.session_state.dem_results
selected_types = st.session_state.get('selected_analysis_types', [])
matched_sheets = st.session_state.get('matched_sheets', [])
analysis_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# --- 4. Page Header ---
st.markdown('''<div class="page-header"><h1>분석 결과</h1><p>선택하신 항목에 대한 지형 분석 결과입니다.</p></div>''',
            unsafe_allow_html=True)

# --- 6. 2D Analysis Results (in Tabs) ---
st.markdown("### 📈 2D 상세 분석 결과")
analysis_map = {
    'elevation': {'title': "표고 분석", 'unit': "m", 'binned_label': "표고 구간별 면적"},
    'slope': {'title': "경사 분석", 'unit': "°", 'binned_label': "경사 구간별 면적"},
    'aspect': {'title': "경사향 분석", 'unit': "°", 'binned_label': "경사향 구간별 면적"},
    'soil': {'title': "토양도 분석", 'unit': "m²", 'binned_label': "토양 종류별 면적", 'class_col': 'soilsy', 'legend_title': '토성'},
    'hsg': {'title': "수문학적 토양군", 'unit': "m²", 'binned_label': "HSG 등급별 면적", 'class_col': 'hg', 'legend_title': 'HSG 등급'},
    'landcover': {'title': "토지피복도", 'unit': "m²", 'binned_label': "토지피복별 면적", 'class_col': 'l2_name', 'legend_title': '토지피복'},
}
valid_selected_types = [t for t in selected_types if t in dem_results]

if not valid_selected_types:
    st.info("표시할 2D 분석 결과가 없습니다.")
else:
    # Loop through each analysis type and display its results sequentially
    for analysis_type in valid_selected_types:
        with st.container(border=True):
            st.markdown(
                f"### 📈 {analysis_map.get(analysis_type, {}).get('title', analysis_type)}")

            results = dem_results[analysis_type]
            stats = results.get('stats')
            grid = results.get('grid')
            gdf = results.get('gdf')

            # [OPTIMIZATION 1] Halve memory usage of the grid by changing data type
            if grid is not None:
                grid = grid.astype(np.float32)

                # [OPTIMIZATION 2 & Scalebar Fix] Dynamic Downsampling and Pixel Size Adjustment
                effective_pixel_size = st.session_state.get('pixel_size', 1.0)
                PIXEL_THRESHOLD = 5_000_000  # Approx 2500x2000 image
                if grid.size > PIXEL_THRESHOLD:
                    downsample_factor = (PIXEL_THRESHOLD / grid.size) ** 0.5
                    st.info(
                        f"ℹ️ 분석 영역이 매우 커서 시각화 해상도를 원본의 {downsample_factor:.1%}로 자동 조정합니다.")
                    # order=1 for bilinear interpolation
                    grid = zoom(grid, downsample_factor, order=1)
                    # Adjust pixel size to match the new resolution for accurate scale bar
                    effective_pixel_size = effective_pixel_size / downsample_factor

            if stats:
                st.markdown("#### 요약 통계")
                cols = st.columns(3)
                cols[0].metric(
                    label="최소값", value=f"{stats.get('min', 0):.2f} {analysis_map.get(analysis_type, {}).get('unit', '')}")
                cols[1].metric(
                    label="최대값", value=f"{stats.get('max', 0):.2f} {analysis_map.get(analysis_type, {}).get('unit', '')}")
                cols[2].metric(
                    label="평균값", value=f"{stats.get('mean', 0):.2f} {analysis_map.get(analysis_type, {}).get('unit', '')}")
                st.markdown("---_---")

                title = analysis_map.get(analysis_type, {}).get(
                    'title', analysis_type)
                with st.spinner(f"'{title}' 분석도를 생성하는 중입니다..."):
                    # --- Unified DEM Analysis Plotting ---

                    # Retrieve all visualization info from the results dictionary
                    bins = results.get('bins')
                    labels = results.get('labels')
                    palette_name = results.get('palette_name')

                    if not all([bins, labels, palette_name]):
                        st.warning(f"'{title}'에 대한 시각화 정보를 생성할 수 없습니다.")
                        continue

                    palette_data = get_palette(palette_name)
                    if not palette_data:
                        st.warning(f"'{palette_name}' 팔레트를 DB에서 찾을 수 없습니다.")
                        continue

                    # For aspect, reorder the palette to put 'Flat' first, matching the data processing.
                    if analysis_type == 'aspect':
                        flat_label_item = None
                        for item in palette_data:
                            if item['bin_label'].strip().lower() == 'flat':
                                flat_label_item = item
                                break

                        if flat_label_item:
                            palette_data.remove(flat_label_item)
                            palette_data.insert(0, flat_label_item)

                    colors = [item['hex_color'] for item in palette_data]

                    # Create colormap and normalization
                    cmap = ListedColormap(colors)
                    norm = BoundaryNorm(bins, cmap.N)

                    # --- Plotting ---
                    fig, ax = create_padded_fig_ax(figsize=(10, 8))
                    ax.set_title(title, fontsize=16, pad=20)

                    use_hillshade = False
                    if analysis_type == 'elevation':
                        use_hillshade = st.toggle(
                            "음영기복도 중첩", value=True, key=f"hillshade_{analysis_type}")
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
                            del hillshade, hillshade_adjusted, rgba_data
                            gc.collect()
                        else:
                            ax.imshow(np.ma.masked_invalid(
                                grid), cmap=cmap, norm=norm)
                    else:
                        ax.imshow(np.ma.masked_invalid(
                            grid), cmap=cmap, norm=norm)

                    # 등고선 추가 (50m 간격)
                    if analysis_type == 'elevation':
                        min_val, max_val = stats.get(
                            'min', 0), stats.get('max', 1)
                        level_interval = 100
                        start_level = np.ceil(
                            min_val / level_interval) * level_interval
                        end_level = np.floor(
                            max_val / level_interval) * level_interval

                        if start_level < end_level:
                            contour_levels = np.arange(
                                start_level, end_level + 1, level_interval)
                        else:
                            contour_levels = np.linspace(min_val, max_val, 10)

                        contour = ax.contour(
                            grid, levels=contour_levels, colors='k', alpha=0.3, linewidths=0.7)

                        # 등고선 라벨 추가 (50m 간격)
                        label_interval = 100
                        start_label_level = np.ceil(
                            min_val / label_interval) * label_interval
                        end_label_level = np.floor(
                            max_val / label_interval) * label_interval

                        if start_label_level < end_label_level:
                            label_levels = np.arange(
                                start_label_level, end_label_level + 1, label_interval)
                        else:
                            # 100m 간격 라벨을 표시할 수 없으면 모든 등고선에 라벨 표시
                            label_levels = contour_levels

                        clabels = ax.clabel(
                            contour, levels=label_levels, inline=True, fontsize=8, fmt='%.0f')
                        # [Readability Improvement] Add a white stroke to contour labels
                        plt.setp(clabels, fontweight='bold', path_effects=[
                                 path_effects.withStroke(linewidth=3, foreground='w')])

                    # --- Legend and Map Elements ---
                    patches = [mpatches.Patch(color=color, label=label)
                               for color, label in zip(colors, labels)]
                    # legend_title = analysis_map[analysis_type].get('binned_label', '범례')
                    legend_title = '범  례'
                    legend = ax.legend(handles=patches, title=legend_title,
                                       bbox_to_anchor=(0.1, 0.1), loc='lower left',
                                       bbox_transform=fig.transFigure,
                                       fontsize='small', frameon=True, framealpha=1,
                                       edgecolor='black')
                    legend.get_title().set_fontweight('bold')

                    adjust_ax_limits(ax)
                    add_north_arrow(ax)
                    scale_params = calculate_accurate_scalebar_params(
                        effective_pixel_size, grid.shape, 25, fig, ax)
                    draw_accurate_scalebar(
                        fig, ax, effective_pixel_size, scale_params, grid.shape)
                    ax.axis('off')

                    # --- Display and Download ---
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png',
                                bbox_inches='tight', dpi=150)
                    img_buffer.seek(0)
                    st.pyplot(fig)
                    plt.close(fig)

                    file_name_suffix = ""
                    if analysis_type == 'elevation' and use_hillshade:
                        file_name_suffix = "_hillshade"

                    st.download_button(
                        label="PNG 이미지로 다운로드",
                        data=img_buffer,
                        file_name=f"{analysis_type}_analysis{file_name_suffix}_{datetime.datetime.now().strftime('%Y%m%d')}.png",
                        mime="image/png",
                        key=f"download_{analysis_type}"
                    )

            elif gdf is not None and not gdf.empty:
                title = analysis_map.get(
                    analysis_type, {}).get('title', analysis_type)
                with st.spinner(f"'{title}' 분석도를 생성하는 중입니다..."):
                    fig, ax = create_padded_fig_ax(figsize=(10, 10))
                    type_info = analysis_map.get(analysis_type, {})
                    class_col = type_info.get('class_col')

                    # Landcover custom color logic
                    if analysis_type == 'landcover':
                        # Check if color map is loaded and required columns exist
                        color_map = get_landcover_colormap()
                        if color_map and 'l2_code' in gdf.columns and 'l2_name' in gdf.columns:
                            gdf['plot_color'] = gdf['l2_code'].map(color_map)
                            # Plot with the specified colors, handling potential missing colors
                            gdf.plot(ax=ax, color=gdf['plot_color'].fillna(
                                '#FFFFFF'), linewidth=0.5, edgecolor='k')

                            # Create a custom legend
                            unique_cats = gdf[['l2_code', 'l2_name']].drop_duplicates(
                            ).sort_values(by='l2_code')
                            patches = []
                            for _, row in unique_cats.iterrows():
                                code = row['l2_code']
                                name = row['l2_name']
                                # Default to white if not in map
                                color = color_map.get(code, '#FFFFFF')
                                patch = mpatches.Patch(
                                    color=color, label=f'{name} ({code})')
                                patches.append(patch)

                            if patches:
                                n_items = len(patches)
                                n_cols = (n_items + 19) // 20
                                ax.legend(handles=patches, title=type_info.get('legend_title', '분류'),
                                          bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small',
                                          ncol=n_cols)
                        else:
                            # Fallback for landcover if custom colors fail
                            gdf.plot(column=class_col, ax=ax, legend=True, categorical=True,
                                     legend_kwds={'title': type_info.get('legend_title', '분류'), 'bbox_to_anchor': (1.05, 1), 'loc': 'upper left'})

                    # Logic for other vector types (soil, hsg)
                    else:
                        if class_col and class_col in gdf.columns:
                            gdf_plot = gdf[gdf[class_col].notna()].copy()
                            unique_cats = sorted(gdf_plot[class_col].unique())

                            num_cats = len(unique_cats)
                            # Use 'tab20' for up to 20 categories, then a sampled colormap for more
                            if num_cats <= 20:
                                cmap = plt.cm.get_cmap('tab20', num_cats)
                                colors = cmap.colors
                            else:
                                cmap = plt.cm.get_cmap('viridis', num_cats)
                                colors = cmap(np.linspace(0, 1, num_cats))

                            color_map = {cat: color for cat,
                                         color in zip(unique_cats, colors)}
                            gdf_plot['plot_color'] = gdf_plot[class_col].map(
                                color_map)

                            gdf_plot.plot(
                                ax=ax, color=gdf_plot['plot_color'], linewidth=0.5, edgecolor='k')

                            patches = [mpatches.Patch(
                                color=color, label=cat) for cat, color in color_map.items()]

                            if patches:
                                n_items = len(patches)
                                n_cols = (n_items + 19) // 20
                                ax.legend(handles=patches, title=type_info.get('legend_title', '분류'),
                                          bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small',
                                          ncol=n_cols)
                        else:
                            gdf.plot(ax=ax)
                            if class_col:
                                st.warning(
                                    f"주의: 분석에 필요한 분류 컬럼 '{class_col}'을(를) 찾을 수 없습니다. DB 테이블 스키마를 확인해주세요. 사용 가능한 컬럼: {list(gdf.columns)}")
                            else:
                                st.warning("분석 유형에 대한 분류 컬럼이 지정되지 않았습니다.")

                    adjust_ax_limits(ax)
                    add_north_arrow(ax)
                    add_scalebar_vector(ax, dx=1.0)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)  # Close figure to save memory
            else:
                st.info("시각화할 2D 데이터가 없습니다.")
            st.markdown("---")  # Add a separator between analyses
# --- 7. Summary and Downloads ---
st.markdown("### 📋 요약 및 다운로드")
st.markdown("#### 상세 분석 보고서")

summary_lines = []
summary_lines.append(f"분석 일시: {analysis_date}")
summary_lines.append(
    f"분석 대상: {st.session_state.get('uploaded_file_name', 'N/A')}")
if len(matched_sheets) > 20:
    summary_lines.append(
        f"사용된 도엽: {len(matched_sheets)}개 ({', '.join(matched_sheets[:20])}...)")
else:
    summary_lines.append(
        f"사용된 도엽: {len(matched_sheets)}개 ({', '.join(matched_sheets)})")
summary_lines.append("")

pixel_size = st.session_state.get('pixel_size', 1.0)
area_per_pixel = pixel_size * pixel_size

for analysis_type in valid_selected_types:
    stats = dem_results[analysis_type].get('stats')
    binned_stats = dem_results[analysis_type].get('binned_stats')
    gdf = dem_results[analysis_type].get('gdf')
    title_info = analysis_map.get(analysis_type, {})

    summary_lines.append(f"--- {title_info.get('title', analysis_type)} ---")

    # Calculate total area first to use for percentages
    total_area_m2 = 0
    if stats:
        total_area_m2 = stats.get('area', 0) * area_per_pixel
        unit = title_info.get('unit', '')
        summary_lines.append(f"- 최소값: {stats.get('min', 0):.2f} {unit}")
        summary_lines.append(f"- 최대값: {stats.get('max', 0):.2f} {unit}")
        summary_lines.append(f"- 평균값: {stats.get('mean', 0):.2f} {unit}")
        summary_lines.append(f"- 분석 면적: {int(total_area_m2):,} m²")
    elif gdf is not None and not gdf.empty:
        gdf['area'] = gdf.geometry.area
        total_area_m2 = gdf.area.sum()

    if binned_stats:
        summary_lines.append(f"\n[{title_info.get('binned_label', '구간별 통계')}]")
        for row in binned_stats:
            binned_area_m2 = row['area'] * area_per_pixel
            percentage = (binned_area_m2 / total_area_m2 *
                          100) if total_area_m2 > 0 else 0
            summary_lines.append(
                f"- {row['bin_range']} {title_info.get('unit', '')}: {int(binned_area_m2):,} m² ({percentage:.1f} %)")

    if gdf is not None and not gdf.empty:
        class_col = title_info.get('class_col')
        if class_col and class_col in gdf.columns:
            summary = gdf.groupby(class_col)[
                'area'].sum().sort_values(ascending=False)
            summary_lines.append(
                f"\n[{title_info.get('binned_label', '종류별 통계')}]")
            for item, area in summary.items():
                percentage = (area / total_area_m2 *
                              100) if total_area_m2 > 0 else 0
                summary_lines.append(
                    f"- {item}: {int(area):,} m² ({percentage:.1f} %)")
        else:
            summary_lines.append(f"- 총 분석 면적: {int(total_area_m2):,} m²")
            if class_col:
                summary_lines.append(
                    f"- (상세 면적 통계를 계산하려면 '{class_col}' 컬럼이 필요합니다.)")

    summary_lines.append("")

summary_text = "\n".join(summary_lines)
st.text_area("분석 결과 요약", summary_text, height=300)
st.download_button("분석 결과(.txt) 다운로드", summary_text, "analysis_summary.txt")


# --- 8. Footer ---
st.markdown("---_---")
if st.button("새로운 분석 시작하기"):
    for key in list(st.session_state.keys()):
        if key not in ['upload_counter']:
            del st.session_state[key]
    st.switch_page("app.py")
    st.switch_page("app.py")
    st.switch_page("app.py")
