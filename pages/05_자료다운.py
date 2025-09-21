import datetime
import io
import platform

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import streamlit as st
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap
from matplotlib.patches import Polygon, Rectangle
from scipy.interpolate import griddata
from stpyvista import stpyvista

from utils.color_palettes import get_landcover_colormap
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

# --- 3. Helper functions (v7 Styling - Final Layout) ---


def add_north_arrow(ax, x=0.95, y=0.95, size=0.05):
    """Axes의 우측 상단에 나침반형 방위를 추가합니다. """
    north_part = Polygon([[x, y], [x - size * 0.4, y - size], [x, y - size*0.8]],
                         facecolor='black', edgecolor='black', transform=ax.transAxes)
    south_part = Polygon([[x, y], [x + size * 0.4, y - size], [x, y - size*0.8]],
                         facecolor='#666666', edgecolor='black', transform=ax.transAxes)
    ax.add_patch(north_part)
    ax.add_patch(south_part)
    ax.text(x, y + size*0.2, 'N', ha='center', va='center',
            fontsize='large', fontweight='bold', transform=ax.transAxes)


def add_scalebar_vector(ax, dx=1.0):
    """Axes의 좌측 하단에 축척과 비율을 추가합니다. (벡터용)"""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # 1. 그래픽 축척 계산 및 그리기
    map_width_m = (x_max - x_min) * dx
    target_length = map_width_m * 0.5
    powers = 10**np.floor(np.log10(target_length))
    base = np.ceil(target_length / powers)
    if base > 5:
        base = 5
    elif base > 2:
        base = 2
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

    # 2. 비율 축척 계산 및 표시
    fig = ax.get_figure()
    ax_pos = ax.get_position()
    fig_width_inch = fig.get_size_inches()[0]
    ax_width_inch = ax_pos.width * fig_width_inch
    m_per_inch = map_width_m / ax_width_inch
    # ratio_scale = m_per_inch / 0.0254
    ax.text(x_pos + (nice_len_m / dx) / 2, y_pos + bar_height * 2.5, '',
            ha='center', va='bottom', fontsize='medium', zorder=10)

def calculate_accurate_scalebar_params(pixel_size, img_shape, target_size_mm, fig, ax):
    """실제 axes 정보를 사용한 정확한 스케일바 계산"""
    fig_width_inch, fig_height_inch = fig.get_size_inches()
    ax_bbox = ax.get_position()

    img_width_fig = ax_bbox.width
    img_height_fig = ax_bbox.height

    img_height_pixels, img_width_pixels = img_shape

    img_width_inch = img_width_fig * fig_width_inch
    pixels_per_inch = img_width_pixels / img_width_inch

    target_size_inch = target_size_mm / 25.4
    target_pixels = target_size_inch * pixels_per_inch
    real_distance_m = target_pixels * pixel_size

    if real_distance_m < 100:
        scale_distance_m = round(real_distance_m / 50) * 50
        if scale_distance_m == 0:
            scale_distance_m = 50
        unit = 'm'
        scale_value = scale_distance_m
    elif real_distance_m < 1000:
        scale_distance_m = round(real_distance_m / 100) * 100
        unit = 'm'
        scale_value = scale_distance_m
    elif real_distance_m < 5000:
        scale_distance_km = round(real_distance_m / 500) * 0.5
        scale_distance_m = scale_distance_km * 1000
        unit = 'km'
        scale_value = scale_distance_km
    else:
        scale_distance_km = round(real_distance_m / 1000)
        scale_distance_m = scale_distance_km * 1000
        unit = 'km'
        scale_value = scale_distance_km

    actual_scalebar_width_fig = (
        scale_distance_m / pixel_size) / pixels_per_inch / fig_width_inch

    return {
        'length': scale_value,
        'units': unit,
        'segments': 2,
        'target_size_mm': target_size_mm,
        'real_distance_m': scale_distance_m,
        'scalebar_width_fig': actual_scalebar_width_fig
    }


def draw_accurate_scalebar(fig, ax, pixel_size, scale_params, img_shape):
    """정확한 축척 반영 스케일바"""
    total_length = scale_params['length']
    units = scale_params['units']
    segments = scale_params['segments']
    scalebar_width_fig = scale_params['scalebar_width_fig']

    start_x_fig = 0.1
    start_y_fig = 0.02
    bar_height_fig = 0.008

    bg_width_fig = scalebar_width_fig + 0.01
    bg_height_fig = bar_height_fig * 2 + 0.03

    bg_rect = Rectangle((start_x_fig - 0.005, start_y_fig - 0.005),
                        bg_width_fig, bg_height_fig,
                        facecolor='white', edgecolor='none', linewidth=0,
                        alpha=0.9, transform=fig.transFigure)
    fig.patches.append(bg_rect)

    segment_width_fig = scalebar_width_fig / segments
    segment_value = total_length / segments

    for i in range(segments):
        x_fig = start_x_fig + i * segment_width_fig

        color1 = 'black' if i % 2 == 0 else 'white'
        rect1 = Rectangle((x_fig, start_y_fig + bar_height_fig),
                          segment_width_fig, bar_height_fig,
                          facecolor=color1, edgecolor='black', linewidth=0.5,
                          transform=fig.transFigure)
        fig.patches.append(rect1)

        color2 = 'white' if i % 2 == 0 else 'black'
        rect2 = Rectangle((x_fig, start_y_fig),
                          segment_width_fig, bar_height_fig,
                          facecolor=color2, edgecolor='black', linewidth=0.5,
                          transform=fig.transFigure)
        fig.patches.append(rect2)

        text_x_fig = x_fig + segment_width_fig
        text_y_fig = start_y_fig + bar_height_fig * 2 + 0.005

        segment_val = (i + 1) * segment_value

        if units == 'km':
            if segment_val != int(segment_val):
                text_label = f'{segment_val:.1f}'
            else:
                text_label = f'{int(segment_val)}'
        else:
            text_label = f'{int(segment_val)}'

        if i == segments - 1:
            text_label += units

        fig.text(text_x_fig, text_y_fig, text_label,
                 ha='center', va='bottom', fontsize=9, fontweight='bold',
                 color='black', transform=fig.transFigure)

    fig.text(start_x_fig, start_y_fig + bar_height_fig * 2 + 0.005, '0',
             ha='center', va='bottom', fontsize=9, fontweight='bold',
             color='black', transform=fig.transFigure)


def create_hillshade(data, azimuth=315, altitude=45):
    """고품질 음영기복도(Hillshade) 생성 (numpy.gradient 사용)"""
    # NaN 값 처리: 계산에 영향을 주지 않도록 유효한 마스크 생성
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return np.ones_like(data) * 0.5  # 데이터가 모두 NaN이면 중간 밝기 반환

    # 경사도와 경사향 계산 (y, x 순서 주의)
    dy, dx = np.gradient(data)

    # 경사각 계산 (라디안)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))

    # 경사향 계산 (라디안, 북쪽 0도 기준)
    aspect = np.arctan2(-dx, dy)

    # 태양 각도를 라디안으로 변환
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(altitude)

    # Hillshade 계산 공식
    hillshade = (np.sin(altitude_rad) * np.sin(np.pi/2 - slope) +
                 np.cos(altitude_rad) * np.cos(np.pi/2 - slope) *
                 np.cos(azimuth_rad - aspect))

    # 0~1 범위로 정규화
    hillshade = np.clip(hillshade, 0, 1)

    # 원래 NaN이었던 위치는 중간 밝기로 설정
    hillshade[~valid_mask] = 0.5

    return hillshade


def create_padded_fig_ax(figsize=(10, 10)):
    """상하에 여백이 있는 Figure와 Axes를 생성합니다. """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


interval_candidates = [1, 2, 3, 5, 10, 15, 20, 25, 50, 100, 150, 200, 500]

# 표고 전용 사용자 정의 색상 (from hillsade.py)
elevation_colors = {
    10: ['#66CDAA', '#DDF426', '#71B800', '#558C00', '#F29300', '#981200', '#9C1C1C', '#955050', '#9C9B9B', '#FFFAFA']
}

def generate_custom_intervals(min_val, max_val, divisions):
    """사용자 정의 구간 생성 (hillsade.py 로직)"""
    # divisions는 라벨의 개수를 의미
    if max_val - min_val == 0:
        # 데이터 값이 모두 같을 경우
        labels = [f"{min_val:.1f}"] * divisions
        interval = 0
        start = min_val
        return labels, interval, start

    diff = max_val - min_val
    div = divisions - 2  # '미만', '초과'를 제외한 중간 구간의 수
    if div <= 0:  # divisions가 2 이하일 경우
        div = 1

    target_interval = diff / div
    # 목표 구간과 가장 가까운 'nice'한 구간 후보를 찾음
    sorted_candidates = sorted(
        interval_candidates, key=lambda x: abs(x - target_interval))

    # 가장 적합한 후보를 선택 (단, diff보다 큰 후보는 제외)
    best_interval = sorted_candidates[0]
    for candidate in sorted_candidates:
        if candidate * div > diff * 0.5:  # 너무 간격이 넓어지지 않도록 필터링
            best_interval = candidate
            break

    mid_val = min_val + diff / 2
    mid_aligned = round(mid_val / best_interval) * best_interval
    start = mid_aligned - best_interval * (div // 2)

    # 라벨 생성
    labels = [f"{start} 미만"]
    for i in range(div):
        lo = start + i * best_interval
        hi = lo + best_interval
        labels.append(f"{lo}~{hi}")
    labels.append(f"{hi} 초과")

    return labels, best_interval, start


def adjust_ax_limits(ax, y_pad_fraction=0.15, x_pad_fraction=0.05):
    """Axes의 Y축 범위를 확장하여 여백을 만듭니다. """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - x_range * x_pad_fraction,
                x_max + x_range * x_pad_fraction)
    ax.set_ylim(y_min - y_range * y_pad_fraction,
                y_max + y_range * y_pad_fraction)


# --- 3. Data Loading from Session ---
dem_results = st.session_state.dem_results
selected_types = st.session_state.get('selected_analysis_types', [])
matched_sheets = st.session_state.get('matched_sheets', [])
analysis_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# --- 4. Page Header ---
st.markdown('''<div class="page-header"><h1>분석 결과</h1><p>선택하신 항목에 대한 지형 분석 결과입니다.</p></div>''',
            unsafe_allow_html=True)

# --- 5. 3D Visualization (주석 처리) ---


# """
# st.markdown("### 🧊 3D 통합 분석 결과")
# dem_analysis_types = [t for t in selected_types if t in [
#     'elevation', 'slope', 'aspect']]
# if not dem_analysis_types:
#     st.info("3D 시각화를 위한 분석 항목(표고, 경사, 경사향)이 선택되지 않았습니다.")
# else:
#     dem_grid = dem_results.get('elevation', {}).get('grid')
#     if dem_grid is None or np.all(np.isnan(dem_grid)):
#         st.warning("3D 시각화를 위한 표고 데이터가 없습니다.")
#     else:
#         with st.spinner("3D 모델을 렌더링하는 중입니다..."):
#             try:
#                 x = np.arange(dem_grid.shape[1])
#                 y = np.arange(dem_grid.shape[0])
#                 xv, yv = np.meshgrid(x, y)

#                 grid_pv = pv.StructuredGrid(xv, yv, dem_grid)
#                 grid_pv["Elevation"] = dem_grid.ravel(order="F")
#                 if 'slope' in dem_results:
#                     grid_pv["Slope (°)"] = dem_results['slope']['grid'].ravel(
#                         order="F")
#                 if 'aspect' in dem_results:
#                     grid_pv["Aspect (°)"] = dem_results['aspect']['grid'].ravel(
#                         order="F")

#                 grid_pv.warp_by_scalar(
#                     scalars="Elevation", factor=0.5, inplace=True)
#                 plotter = pv.Plotter(
#                     shape=(1, len(dem_analysis_types)), border=False, window_size=[1200, 400])
#                 scalar_bar_args = {
#                     'title_font_size': 16, 'label_font_size': 12, 'vertical': True,
#                     'position_x': 0.75, 'position_y': 0.1, 'height': 0.8, 'width': 0.08,
#                     'n_labels': 7, 'fmt': '%.2f', 'shadow': True, 'font_family': 'arial',
#                 }

#                 for i, analysis_type in enumerate(dem_analysis_types):
#                     plotter.subplot(0, i)
#                     var_name = {"elevation": "Elevation", "slope": "Slope (°)", "aspect": "Aspect (°)"}[
#                         analysis_type]
#                     cmap = {"elevation": "terrain", "slope": "viridis",
#                             "aspect": "hsv"}[analysis_type]

#                     plotter.add_text(var_name, font_size=12)
#                     mesh = grid_pv.copy()
#                     mesh.set_active_scalars(var_name)
#                     specific_args = scalar_bar_args.copy()
#                     specific_args['title'] = var_name
#                     plotter.add_mesh(mesh, scalars=var_name, cmap=cmap, show_scalar_bar=True,
#                                      scalar_bar_args=specific_args, nan_opacity=0.0)
#                     plotter.camera_position = 'iso'

#                 plotter.background_color = 'white'
#                 stpyvista(plotter, key="pv_3d_view")
#             except Exception as e:
#                 st.error(f"3D 모델 생성 중 오류가 발생했습니다: {e}")
# """

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
    tab_titles = [analysis_map.get(t, {}).get('title', t)
                  for t in valid_selected_types]
    tabs = st.tabs(tab_titles)
    for i, analysis_type in enumerate(valid_selected_types):
        with tabs[i]:
            results = dem_results[analysis_type]
            stats = results.get('stats')
            grid = results.get('grid')
            gdf = results.get('gdf')

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

                title = analysis_map.get(analysis_type, {}).get('title', analysis_type)
                with st.spinner(f"'{title}' 분석도를 생성하는 중입니다..."):
                    # 표고 분석: QGIS 스타일 등급 범례 + 음영기복도 오버레이 옵션
                    if analysis_type == 'elevation':

                        # --- 시각화 옵션 ---
                        use_hillshade = st.toggle("음영기복도 중첩", value=True,
                                                  help="지형의 입체감을 표현하기 위해 음영기복도를 표고 분석도에 겹쳐서 표시합니다.")

                        fig, ax = create_padded_fig_ax(figsize=(10, 8))

                        min_val, max_val = stats.get('min', 0), stats.get('max', 1)

                        # hillsade.py 스타일 범례 및 구간 생성
                        divisions = 10
                        legend_labels, interval, start = generate_custom_intervals(
                            min_val, max_val, divisions)

                        # BoundaryNorm을 위한 레벨(구간 경계) 설정
                        # N개의 구간은 N+1개의 경계가 필요함
                        div = divisions - 2
                        levels = [start + i * interval for i in range(div + 1)]
                        levels = [float('-inf')] + levels + [float('inf')]

                        # 사용자 정의 색상 적용
                        cmap = ListedColormap(elevation_colors[10])
                        norm = BoundaryNorm(levels, cmap.N)

                        # 음영기복도 중첩 로직 (hillsade.py 스타일)
                        if use_hillshade:
                            # 1. DEM 데이터로부터 음영기복도 생성
                            hillshade = create_hillshade(grid)

                            # 2. 컬러맵 적용하여 RGBA 데이터 생성
                            # norm은 각 픽셀이 어떤 색상 구간에 속하는지 인덱스를 반환
                            rgba_data = cmap(norm(grid))

                            # 3. 음영 효과와 색상 데이터 조합 (RGB 채널에만 적용)
                            # hillshade 값을 조정하여 너무 어두워지지 않게 함 (0.5 ~ 1.2)
                            intensity = 0.7
                            hillshade_adjusted = 0.5 + hillshade * intensity

                            # RGB 채널(0,1,2)에 음영 효과를 곱함
                            for i in range(3):
                                rgba_data[:, :, i] *= hillshade_adjusted

                            # 4. 유효하지 않은 데이터(NaN)는 투명하게 처리
                            valid_mask = ~np.isnan(grid)
                            rgba_data[~valid_mask, 3] = 0  # Alpha channel to 0

                            # 5. 최종 이미지 표시
                            im = ax.imshow(
                                np.clip(rgba_data, 0, 1), origin='upper')
                        else:
                            # 음영기복도를 사용하지 않을 경우, 기존처럼 표고도만 그리기
                            im = ax.imshow(np.ma.masked_invalid(
                                grid), cmap=cmap, norm=norm)

                        # 등고선 추가 (50m 간격)
                        level_interval = 50
                        start_level = np.ceil(min_val / level_interval) * level_interval
                        end_level = np.floor(max_val / level_interval) * level_interval

                        if start_level < end_level:
                            contour_levels = np.arange(start_level, end_level + 1, level_interval)
                        else:
                            contour_levels = np.linspace(min_val, max_val, 10)

                        contour = ax.contour(
                            grid, levels=contour_levels, colors='k', alpha=0.3, linewidths=0.7)

                        # 등고선 라벨 추가 (100m 간격)
                        label_interval = 100
                        start_label_level = np.ceil(min_val / label_interval) * label_interval
                        end_label_level = np.floor(max_val / label_interval) * label_interval

                        if start_label_level < end_label_level:
                            label_levels = np.arange(start_label_level, end_label_level + 1, label_interval)
                        else:
                            # 100m 간격 라벨을 표시할 수 없으면 모든 등고선에 라벨 표시
                            label_levels = contour_levels

                        clabels = ax.clabel(contour, levels=label_levels, inline=True, fontsize=8, fmt='%.0f')
                        plt.setp(clabels, fontweight='bold')

                        ax.set_title(analysis_map[analysis_type].get(
                            'title', ''), fontsize=16, pad=20)

                        # hillsade.py 스타일 범례 생성
                        patches = []
                        for i in range(len(legend_labels)):
                            color = cmap(i)
                            label = f"{legend_labels[i]} m"
                            patches.append(mpatches.Patch(
                                color=color, label=label))

                        legend = ax.legend(handles=patches, title="표고 범위 (m)",
                                           bbox_to_anchor=(1.05, 1), loc='upper left',
                                           fontsize='small', frameon=True, framealpha=1,
                                           edgecolor='black')
                        legend.get_title().set_fontweight('bold')

                        adjust_ax_limits(ax)
                        add_north_arrow(ax)
                        pixel_size = st.session_state.get('pixel_size', 1.0)
                        scale_params = calculate_accurate_scalebar_params(pixel_size, grid.shape, 25, fig, ax)
                        draw_accurate_scalebar(fig, ax, pixel_size, scale_params, grid.shape)
                        ax.axis('off')

                        # PNG 다운로드를 위한 이미지 버퍼 생성
                        img_buffer = io.BytesIO()
                        fig.savefig(img_buffer, format='png',
                                    bbox_inches='tight', dpi=150)
                        img_buffer.seek(0)

                        st.pyplot(fig)
                        plt.close(fig)

                        # 다운로드 버튼 추가
                        file_name_suffix = "_hillshade" if use_hillshade else ""
                        st.download_button(
                            label=f"PNG 이미지로 다운로드{' (음영기복도 포함)' if use_hillshade else ''}",
                            data=img_buffer,
                            file_name=f"elevation_analysis{file_name_suffix}_{datetime.datetime.now().strftime('%Y%m%d')}.png",
                            mime="image/png"
                        )

                    # 경사, 경사향 분석: 기존 탭 스타일 유지
                    else:
                        cmap_options = {
                            'slope': ['viridis', 'inferno', 'magma', 'RdYlGn_r'],
                            'aspect': ['hsv', 'twilight_shifted', 'twilight']
                        }
                        cmaps_to_show = cmap_options.get(
                            analysis_type, ['viridis'])
                        cmap_tabs = st.tabs([f"{cmap}" for cmap in cmaps_to_show])

                        for j, cmap in enumerate(cmaps_to_show):
                            with cmap_tabs[j]:
                                fig, ax = create_padded_fig_ax(figsize=(10, 8))
                                im = ax.imshow(
                                    np.ma.masked_invalid(grid), cmap=cmap)
                                cbar = fig.colorbar(
                                    im, ax=ax, label=analysis_map[analysis_type].get('unit', ''), shrink=0.8)
                                ax.set_title(
                                    f"{analysis_map[analysis_type].get('title', '')} - '{cmap}'")

                                adjust_ax_limits(ax)
                                add_north_arrow(ax)
                                pixel_size = st.session_state.get('pixel_size', 1.0)
                                scale_params = calculate_accurate_scalebar_params(pixel_size, grid.shape, 25, fig, ax)
                                draw_accurate_scalebar(fig, ax, pixel_size, scale_params, grid.shape)
                                ax.axis('off')
                                st.pyplot(fig)
                                plt.close(fig)

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
                                ax.legend(handles=patches, title=type_info.get('legend_title', '분류'),
                                          bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                        else:
                            # Fallback for landcover if custom colors fail
                            gdf.plot(column=class_col, ax=ax, legend=True, categorical=True,
                                     legend_kwds={'title': type_info.get('legend_title', '분류'), 'bbox_to_anchor': (1.05, 1), 'loc': 'upper left'})

                    # Logic for other vector types (soil, hsg)
                    else:
                        if class_col and class_col in gdf.columns:
                            gdf.plot(column=class_col, ax=ax, legend=True, categorical=True,
                                     legend_kwds={'title': type_info.get('legend_title', '분류'), 'bbox_to_anchor': (1.05, 1), 'loc': 'upper left'})
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
    if stats:
        unit = title_info.get('unit', '')
        total_area_m2 = stats.get('area', 0) * area_per_pixel
        summary_lines.append(f"- 최소값: {stats.get('min', 0):.2f} {unit}")
        summary_lines.append(f"- 최대값: {stats.get('max', 0):.2f} {unit}")
        summary_lines.append(f"- 평균값: {stats.get('mean', 0):.2f} {unit}")
        summary_lines.append(f"- 분석 면적: {int(total_area_m2):,} m²")

    if binned_stats:
        summary_lines.append(f"\n[{title_info.get('binned_label', '구간별 통계')}]")
        for row in binned_stats:
            binned_area_m2 = row['area'] * area_per_pixel
            summary_lines.append(
                f"- {row['bin_range']} {title_info.get('unit', '')}: {int(binned_area_m2):,} m²")

    if gdf is not None and not gdf.empty:
        gdf['area'] = gdf.geometry.area
        class_col = title_info.get('class_col')
        if class_col and class_col in gdf.columns:
            summary = gdf.groupby(class_col)[
                'area'].sum().sort_values(ascending=False)
            summary_lines.append(
                f"\n[{title_info.get('binned_label', '종류별 통계')}]")
            for item, area in summary.items():
                summary_lines.append(f"- {item}: {int(area):,} m²")
        else:
            summary_lines.append(f"- 총 분석 면적: {int(gdf.area.sum()):,} m²")
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

