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
import shapefile  # 새로 설치된 pyshp 라이브러리 사용
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Polygon, Rectangle
from scipy.ndimage import zoom
from shapely import wkb
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from utils.color_palettes import get_landcover_colormap, get_palette
from utils.plot_helpers import (
    add_north_arrow,
    add_scalebar_vector,
    adjust_ax_limits,
    calculate_accurate_scalebar_params,
    create_hillshade,
    create_padded_fig_ax,
    draw_accurate_scalebar,
    generate_aspect_bins,
    generate_custom_intervals,
    generate_slope_intervals,
)
from utils.theme_util import apply_styles


# --- pyshp를 사용한 SHP 내보내기 도우미 함수 ---
def create_shapefile_zip(gdf: gpd.GeoDataFrame, base_filename: str) -> io.BytesIO | None:
    """GeoDataFrame을 메모리에서 pyshp 라이브러리를 사용하여 압축된 Shapefile로 변환합니다."""
    if gdf.empty:
        return None

    gdf = gdf.copy()

    # --- [최종 클립] 출력이 원본 사용자 경계와 일치하는지 확인 ---
    if 'gdf' in st.session_state:
        original_gdf = st.session_state.gdf
        if not original_gdf.empty:
            # 클리핑 전 CRS 일치 확인
            if gdf.crs != original_gdf.crs:
                gdf = gdf.to_crs(original_gdf.crs)
            
            # 클립 수행
            gdf = gpd.clip(gdf, original_gdf, keep_geom_type=True)
            if gdf.empty:
                st.warning("최종 클리핑 후 SHP 파일로 변환할 유효한 데이터가 없습니다.")
                return None
    # --- 최종 클립 종료 ---

    # 단계 1: 모든 요소를 지오메트리 객체로 강제 변환하고 WKB를 올바르게 처리합니다.
    def force_to_geometry(geom):
        if isinstance(geom, str):
            try:
                return wkb.loads(geom, hex=True)
            except Exception:
                return None
        return geom if isinstance(geom, BaseGeometry) else None
    gdf['geometry'] = gdf['geometry'].apply(force_to_geometry)

    # 단계 2: 모든 지오메트리를 2D로 강제 변환합니다.
    def drop_z(geom):
        if geom is None or not geom.has_z:
            return geom
        return transform(lambda x, y, z=None: (x, y), geom)
    gdf['geometry'] = gdf['geometry'].apply(drop_z)

    # 단계 3: null이거나 유효하지 않은 지오메트리를 필터링합니다.
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty &
              gdf.geometry.is_valid]
    if gdf.empty:
        st.warning("SHP 파일로 변환할 유효한 데이터가 없습니다. (데이터 정제 후)")
        return None

    # 단계 4: Shapefile 호환성을 위해 컬럼 이름을 자릅니다.
    gdf.columns = [str(col) for col in gdf.columns]
    rename_dict = {}
    for col in gdf.columns:
        if len(col.encode('utf-8')) > 10:
            new_col = col.encode('utf-8')[:10].decode('utf-8', 'ignore')
            i = 1
            final_col = new_col
            while final_col in gdf.columns or final_col in rename_dict.values():
                suffix = f"_{i}"
                final_col = f"{col.encode('utf-8')[:10-len(suffix.encode('utf-8'))].decode('utf-8', 'ignore')}{suffix}"
                i += 1
            rename_dict[col] = final_col
    if rename_dict:
        gdf = gdf.rename(columns=rename_dict)

    # 단계 5: pyshp를 사용하여 shapefile에 씁니다.
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = str(Path(tmpdir) / f"{base_filename}.shp")
        try:
            with shapefile.Writer(shp_path) as w:
                w.autoBalance = 1  # 일관성 보장

                # GeoDataFrame 컬럼에서 필드 정의
                for col_name, dtype in gdf.dtypes.items():
                    if col_name.lower() == 'geometry':
                        continue
                    if pd.api.types.is_integer_dtype(dtype):
                        w.field(col_name, 'N')
                    elif pd.api.types.is_float_dtype(dtype):
                        w.field(col_name, 'F')
                    elif pd.api.types.is_datetime64_any_dtype(dtype):
                        w.field(col_name, 'D')
                    else:
                        w.field(col_name, 'C', size=254)

                # 지오메트리 및 레코드 작성
                for index, row in gdf.iterrows():
                    w.shape(row.geometry)

                    record_values = []
                    for col_name in gdf.columns:
                        if col_name.lower() == 'geometry':
                            continue

                        value = row[col_name]

                        # 레코드를 작성하기 전에 잠재적인 NaN 값 처리
                        if pd.isna(value):
                            dtype = gdf[col_name].dtype
                            if pd.api.types.is_integer_dtype(dtype):
                                value = 0
                            elif pd.api.types.is_float_dtype(dtype):
                                value = 0.0
                            else:
                                value = ''  # 문자열/기타 유형의 기본값

                        record_values.append(value)

                    w.record(*record_values)

            # 생성된 파일 압축
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in Path(tmpdir).glob(f'{base_filename}.*'):
                    zip_file.write(file_path, arcname=file_path.name)
            zip_buffer.seek(0)
            return zip_buffer

        except Exception as e:
            st.error(f"pyshp 라이브러리로 SHP 파일 생성 중 오류가 발생했습니다: {e}")
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
    # Linux의 경우 'NanumGothic'이 설치되어 있는지 확인합니다.
    # sudo apt-get install -y fonts-nanum*
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
        # 세션 상태를 지우기 전에 임시 TIF 파일 정리
        if 'dem_results' in st.session_state:
            for analysis_type in st.session_state.dem_results:
                results = st.session_state.dem_results.get(analysis_type, {})
                tif_path = results.get('tif_path')
                if tif_path and Path(tif_path).exists():
                    try:
                        Path(tif_path).unlink()
                    except OSError as e:
                        st.warning(f".tif 파일을 삭제하는 데 실패했습니다: {e}")

        for key in list(st.session_state.keys()):
            if key != 'upload_counter':
                del st.session_state[key]
        st.switch_page("app.py")

# --- 6. 2D 분석 결과 (탭) ---
st.markdown("### 📈 2D 상세 분석 결과")
analysis_map = {
    'elevation': {'title': "표고 분석", 'unit': "m", 'binned_label': "표고 구간별 면적"},
    'slope': {'title': "경사 분석", 'unit': "°", 'binned_label': "경사 구간별 면적"},
    'aspect': {'title': "경사향 분석", 'unit': "°", 'binned_label': "경사향 구간별 면적"},
    'soil': {'title': "토양도 분석", 'unit': "m²", 'binned_label': "토양 종류별 면적", 'class_col': 'soilsy', 'legend_title': '토양도(Soilsy)'},
    'hsg': {'title': "수문학적 토양군", 'unit': "m²", 'binned_label': "HSG 등급별 면적", 'class_col': 'hg', 'legend_title': '토양군(HSG)'},
    'landcover': {'title': "토지피복도", 'unit': "m²", 'binned_label': "토지피복별 면적", 'class_col': 'l2_name', 'legend_title': '토지피복도(Landcover)'},
}
valid_selected_types = [t for t in selected_types if t in dem_results]

if not valid_selected_types:
    st.info("표시할 2D 분석 결과가 없습니다.")
else:
    # 각 분석 유형을 반복하고 결과를 순차적으로 표시합니다.
    for analysis_type in valid_selected_types:
        with st.container(border=True):
            st.markdown(
                f"### 📈 {analysis_map.get(analysis_type, {}).get('title', analysis_type)}")

            results = dem_results[analysis_type]
            stats = results.get('stats')
            grid = results.get('grid')
            gdf = results.get('gdf')

            # [최적화 1] 데이터 유형을 변경하여 그리드의 메모리 사용량을 절반으로 줄입니다.
            if grid is not None:
                grid = grid.astype(np.float32)

                # [최적화 2 & 축척 막대 수정] 동적 다운샘플링 및 픽셀 크기 조정
                effective_pixel_size = st.session_state.get('pixel_size', 1.0)
                PIXEL_THRESHOLD = 5_000_000  # 약 2500x2000 이미지
                if grid.size > PIXEL_THRESHOLD:
                    downsample_factor = (PIXEL_THRESHOLD / grid.size) ** 0.5
                    st.info(
                        f"ℹ️ 분석 영역이 매우 커서 시각화 해상도를 원본의 {downsample_factor:.1%}로 자동 조정합니다.")
                    # order=1은 이중 선형 보간을 의미합니다.
                    grid = zoom(grid, downsample_factor, order=1)
                    # 정확한 축척 막대를 위해 새 해상도와 일치하도록 픽셀 크기 조정
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

                title = analysis_map.get(analysis_type, {}).get(
                    'title', analysis_type)
                with st.spinner(f"'{title}' 분석도를 생성하는 중입니다..."):
                    # --- 통합 DEM 분석 플로팅 ---

                    # 결과 사전에서 모든 시각화 정보 검색
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

                    # 경사향의 경우, 데이터 처리와 일치하도록 팔레트를 재정렬하여 'Flat'을 먼저 배치합니다.
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

                    # 컬러맵 및 정규화 생성
                    cmap = ListedColormap(colors)
                    norm = BoundaryNorm(bins, cmap.N)

                    # --- 플로팅 ---
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
                        # [가독성 향상] 등고선 라벨에 흰색 테두리 추가
                        plt.setp(clabels, fontweight='bold', path_effects=[
                                 path_effects.withStroke(linewidth=3, foreground='w')])

                    # --- 범례 및 지도 요소 ---
                    patches = [mpatches.Patch(color=color, label=label)
                               for color, label in zip(colors, labels)]
                    legend_titles = {
                        'elevation': '표고(Elevation)',
                        'slope': '경사(Slope)',
                        'aspect': '경사향(Aspect)'
                    }
                    legend_title = legend_titles.get(analysis_type, '범  례')
                    legend = ax.legend(handles=patches, title=legend_title,
                                       bbox_to_anchor=(0.1, 0.1), loc='lower left',
                                       bbox_transform=fig.transFigure,
                                       fontsize='small', frameon=True, framealpha=1,
                                       edgecolor='black')
                    legend.get_title().set_fontweight('bold')

                    # 범례 패치에 테두리 색상 및 너비 강제 적용
                    for legend_patch in legend.get_patches():
                        legend_patch.set_edgecolor('black')
                        legend_patch.set_linewidth(0.7)

                    adjust_ax_limits(ax)
                    add_north_arrow(ax)
                    scale_params = calculate_accurate_scalebar_params(
                        effective_pixel_size, grid.shape, 50, fig, ax)
                    draw_accurate_scalebar(
                        fig, ax, effective_pixel_size, scale_params, grid.shape)
                    ax.axis('off')

                    # --- 표시 및 다운로드 ---
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png',
                                bbox_inches='tight', dpi=150)
                    img_buffer.seek(0)
                    st.pyplot(fig)
                    plot_figures[analysis_type] = fig

            elif gdf is not None and not gdf.empty:
                title = analysis_map.get(
                    analysis_type, {}).get('title', analysis_type)
                with st.spinner(f"'{title}' 분석도를 생성하는 중입니다..."):
                    fig, ax = create_padded_fig_ax(figsize=(10, 10))
                    type_info = analysis_map.get(analysis_type, {})
                    class_col = type_info.get('class_col')

                    # 토지피복도 사용자 정의 색상 로직
                    if analysis_type == 'landcover':
                        # 컬러맵이 로드되었고 필요한 컬럼이 존재하는지 확인
                        color_map = get_landcover_colormap()
                        if color_map and 'l2_code' in gdf.columns and 'l2_name' in gdf.columns:
                            gdf['plot_color'] = gdf['l2_code'].map(color_map)
                            # 지정된 색상으로 플롯, 잠재적인 누락 색상 처리
                            gdf.plot(ax=ax, color=gdf['plot_color'].fillna(
                                '#FFFFFF'), linewidth=0.5, edgecolor='k')

                            # 사용자 정의 범례 생성
                            unique_cats = gdf[['l2_code', 'l2_name']].drop_duplicates(
                            ).sort_values(by='l2_code')
                            patches = []
                            for _, row in unique_cats.iterrows():
                                code = row['l2_code']
                                name = row['l2_name']
                                # 맵에 없으면 흰색으로 기본값 설정
                                color = color_map.get(code, '#FFFFFF')
                                patch = mpatches.Patch(
                                    color=color, label=f'{name}')
                                patches.append(patch)

                            if patches:
                                n_items = len(patches)
                                n_cols = (n_items + 19) // 20
                                legend = ax.legend(handles=patches, title=type_info.get('legend_title', '분류'),
                                                   bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small',
                                                   ncol=n_cols)
                                # 범례 패치에 테두리 색상 및 너비 강제 적용
                                for legend_patch in legend.get_patches():
                                    legend_patch.set_edgecolor('black')
                                    legend_patch.set_linewidth(0.7)
                        else:
                            # 사용자 정의 색상 실패 시 토지피복도 대체 로직
                            gdf.plot(column=class_col, ax=ax, legend=True, categorical=True,
                                     legend_kwds={'title': type_info.get('legend_title', '분류'), 'bbox_to_anchor': (1.05, 1), 'loc': 'upper left'})

                    # 다른 벡터 유형(토양, hsg)에 대한 로직
                    else:
                        if class_col and class_col in gdf.columns:
                            gdf_plot = gdf[gdf[class_col].notna()].copy()
                            unique_cats = sorted(gdf_plot[class_col].unique())

                            num_cats = len(unique_cats)
                            # 최대 20개 범주에는 'tab20'을 사용하고, 그 이상은 샘플링된 컬러맵을 사용합니다.
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
                                legend = ax.legend(handles=patches, title=type_info.get('legend_title', '분류'),
                                                   bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small',
                                                   ncol=n_cols)
                                # 범례 패치에 테두리 색상 및 너비 강제 적용
                                for legend_patch in legend.get_patches():
                                    legend_patch.set_edgecolor('black')
                                    legend_patch.set_linewidth(0.7)
                        else:
                            gdf.plot(ax=ax)
                            if class_col:
                                st.warning(
                                    f"주의: 분석에 필요한 분류 컬럼 '{class_col}'을(를) 찾을 수 없습니다. DB 테이블 스키마를 확인해주세요. 사용 가능한 컬럼: {list(gdf.columns)}")
                            else:
                                st.warning("분석 유형에 대한 분류 컬럼이 지정되지 않았습니다.")

                    adjust_ax_limits(ax)
                    add_north_arrow(ax)

                    # --- DEM 스타일과 일치하도록 축척 막대 통합 ---
                    # 1. 플롯 축에서 미터 단위의 지도 치수 가져오기
                    x_min, x_max = ax.get_xlim()
                    map_width_m = x_max - x_min

                    # 2. 이미지 모양 및 픽셀 크기에 대한 프록시 생성
                    #    (savefig dpi 및 figsize 기반)
                    dpi = 150
                    # create_padded_fig_ax에서 사용된 figsize
                    figsize_w, figsize_h = (10, 10)
                    proxy_img_width_px = int(figsize_w * dpi)
                    proxy_img_height_px = int(figsize_h * dpi)
                    proxy_img_shape = (proxy_img_height_px, proxy_img_width_px)

                    # effective_pixel_size는 미터/픽셀입니다.
                    if proxy_img_width_px > 0:
                        effective_pixel_size = map_width_m / proxy_img_width_px
                    else:
                        effective_pixel_size = 1.0  # 대체

                    # 3. 정확한 축척 막대 계산 및 그리기
                    scale_params = calculate_accurate_scalebar_params(
                        effective_pixel_size, proxy_img_shape, 50, fig, ax)
                    draw_accurate_scalebar(
                        fig, ax, effective_pixel_size, scale_params, proxy_img_shape)
                    ax.axis('off')

                    # --- 표시 및 버퍼 저장 ---
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png',
                                bbox_inches='tight', dpi=150)
                    img_buffer.seek(0)

                    st.pyplot(fig)
                    plot_figures[analysis_type] = fig

                    # --- SHP 파일 버퍼 생성 및 저장 ---
                    shp_zip_buffer = create_shapefile_zip(
                        gdf, f"{analysis_type}_{base_filename}"
                    )
                    if shp_zip_buffer:
                        shp_buffers[analysis_type] = {
                            "buffer": shp_zip_buffer,
                            "title": type_info.get('title', analysis_type)
                        }
                    # --- 로직 완료 ---

            else:
                st.info("시각화할 2D 데이터가 없습니다.")
            st.markdown("---")  # 분석 사이에 구분선 추가
# --- 7. 요약 및 최종 다운로드 ---
st.markdown("### 📋 상세 분석 보고서")

# --- TXT 요약(표시용) 및 CSV(다운로드용) 데이터 생성 ---
summary_lines = []
csv_data = []

pixel_size = st.session_state.get('pixel_size', 1.0)
area_per_pixel = pixel_size * pixel_size

# --- 보고서 헤더의 총 면적 계산 ---
# [중요 수정] 원본 사용자 제공 지오메트리의 면적을 유일한 기준으로 사용합니다.
# 이렇게 하면 보고서의 모든 부분에서 총 면적이 일관되게 유지됩니다.
report_total_area_m2 = 0
if 'gdf' in st.session_state and not st.session_state.gdf.empty:
    # 합산하기 전에 'area' 컬럼이 있는지 확인
    if 'area' not in st.session_state.gdf.columns:
        st.session_state.gdf['area'] = st.session_state.gdf.geometry.area
    report_total_area_m2 = st.session_state.gdf.area.sum()
else:
    # 주 GDF가 없는 드문 경우에 대한 대체 로직
    area_source_type = None
    if 'elevation' in valid_selected_types:
        area_source_type = 'elevation'
    elif valid_selected_types:
        area_source_type = valid_selected_types[0]

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
summary_lines.append(
    f"분석 대상: {st.session_state.get('uploaded_file_name', 'N/A')}")
csv_data.append({'분석 구분': '기본 정보', '항목': '분석 일시',
                '값': analysis_date, '단위': '', '면적(m²)': '', '비율(%)': ''})
csv_data.append({'분석 구분': '기본 정보', '항목': '분석 대상', '값': st.session_state.get(
    'uploaded_file_name', 'N/A'), '단위': '', '면적(m²)': '', '비율(%)': ''})

if len(matched_sheets) > 20:
    summary_lines.append(f"사용된 도엽: {len(matched_sheets)}개")
    csv_data.append({'분석 구분': '기본 정보', '항목': '사용된 도엽 개수', '값': len(
        matched_sheets), '단위': '개', '면적(m²)': '', '비율(%)': ''})
else:
    summary_lines.append(
        f"사용된 도엽: {len(matched_sheets)}개 ({', '.join(matched_sheets)})")
    csv_data.append({'분석 구분': '기본 정보', '항목': '사용된 도엽',
                    '값': f"{len(matched_sheets)}개 ({', '.join(matched_sheets)})", '단위': '', '면적(m²)': '', '비율(%)': ''})

summary_lines.append(f"총 분석 면적: {int(report_total_area_m2):,} m²")
csv_data.append({'분석 구분': '기본 정보', '항목': '총 분석 면적', '값': '', '단위': 'm²',
                '면적(m²)': f"{int(report_total_area_m2):,}", '비율(%)': ''})
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
        summary_lines.append(f"- 최소값: {stats.get('min', 0):.2f} {unit}")
        summary_lines.append(f"- 최대값: {stats.get('max', 0):.2f} {unit}")
        summary_lines.append(f"- 평균값: {stats.get('mean', 0):.2f} {unit}")

        csv_data.append({'분석 구분': title, '항목': '최소값',
                        '값': f"{stats.get('min', 0):.2f}", '단위': unit, '면적(m²)': '', '비율(%)': ''})
        csv_data.append({'분석 구분': title, '항목': '최대값',
                        '값': f"{stats.get('max', 0):.2f}", '단위': unit, '면적(m²)': '', '비율(%)': ''})
        csv_data.append({'분석 구분': title, '항목': '평균값',
                        '값': f"{stats.get('mean', 0):.2f}", '단위': unit, '면적(m²)': '', '비율(%)': ''})

    elif gdf is not None and not gdf.empty:
        if 'area' not in gdf.columns:
            gdf['area'] = gdf.geometry.area
        total_area_m2 = gdf.area.sum()

    if binned_stats:
        summary_lines.append(f"\n[{title_info.get('binned_label', '구간별 통계')}]")
        for row in binned_stats:
            binned_area_m2 = row['area'] * area_per_pixel
            percentage = (binned_area_m2 / total_area_m2 *
                          100) if total_area_m2 > 0 else 0
            summary_lines.append(
                f"- {row['bin_range']}: {int(binned_area_m2):,} m² ({percentage:.1f} %)")
            csv_data.append({'분석 구분': title, '항목': row['bin_range'], '값': '', '단위': '',
                            '면적(m²)': f"{int(binned_area_m2):,}", '비율(%)': f"{percentage:.1f}"})

    if gdf is not None and not gdf.empty:
        class_col = title_info.get('class_col')
        if class_col and class_col in gdf.columns:
            # [중요 수정 2] 먼저 dissolve를 사용하여 지오메트리를 병합한 다음 면적을 계산합니다.
            # 이렇게 하면 소스 데이터의 중첩된 폴리곤을 올바르게 처리합니다.
            dissolved_gdf = gdf.dissolve(by=class_col)
            dissolved_gdf['area'] = dissolved_gdf.geometry.area
            dissolved_gdf = dissolved_gdf.sort_values(by='area', ascending=False)

            summary_lines.append(
                f"\n[{title_info.get('binned_label', '종류별 통계')}]")
            for item, row in dissolved_gdf.iterrows():
                area = row['area']
                percentage = (area / total_area_m2 *
                              100) if total_area_m2 > 0 else 0
                summary_lines.append(
                    f"- {item}: {int(area):,} m² ({percentage:.1f} %)")
                csv_data.append({'분석 구분': title, '항목': item, '값': '', '단위': '',
                                '면적(m²)': f"{int(area):,}", '비율(%)': f"{percentage:.1f}"})
        else:
            # 이 부분은 GDF의 총 면적이 이미 위에서 계산되었기 때문에 까다롭습니다.
            # 주 보고서 헤더와의 혼동을 피하기 위해 다른 '총 면적' 줄을 추가하지 않습니다.
            if class_col:
                summary_lines.append(
                    f"- (상세 면적 통계를 계산하려면 '{class_col}' 컬럼이 필요합니다.)")

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
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
    # zip에 요약 CSV 추가
    zip_file.writestr(
        f"analysis_summary_{timestamp}.csv", csv_string)

    # zip에 플롯 이미지 추가
    for analysis_type, fig in plot_figures.items():
        img_buffer = io.BytesIO()

        # 올바른 파일 이름을 위해 토글 상태 다시 확인
        use_hillshade = st.session_state.get(
            f"hillshade_{analysis_type}", False)
        file_name_suffix = "_hillshade" if analysis_type == 'elevation' and use_hillshade else ""

        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)

        zip_file.writestr(
            f"{analysis_type}_analysis{file_name_suffix}.png", img_buffer.getvalue())
        plt.close(fig)  # 저장 후 그림 닫기

    # zip에 분석 TIF 파일 추가
    for analysis_type in valid_selected_types:
        results = dem_results.get(analysis_type, {})
        tif_path = results.get('tif_path')
        if tif_path and Path(tif_path).exists():
            zip_file.write(tif_path, arcname=f"{analysis_type}.tif")

    # zip의 압축을 풀고 내용을 다시 추가하여 zip에 SHP 파일 추가
    for analysis_type, data in shp_buffers.items():
        inner_zip_buffer = data["buffer"]
        with zipfile.ZipFile(inner_zip_buffer, 'r') as inner_zip:
            for file_info in inner_zip.infolist():
                # 파일을 하위 디렉토리에 넣지 않으려면 직접 작성합니다.
                zip_file.writestr(file_info.filename, inner_zip.read(file_info.filename))

zip_buffer.seek(0)

# --- 8. 최종 다운로드 섹션 ---
st.markdown("### 📥 다운로드")

# --- 모든 다운로드 항목 목록 생성 ---
download_items = []

# 주 zip 버튼에 대한 동적 라벨 및 도움말 텍스트 정의
main_zip_label = "📥 시각화자료+분석보고서 (ZIP)"
main_zip_help = "분석 리포트, 모든 분석도(PNG), 모든 원본 분석 파일(TIF)을 한번에 다운로드합니다."

if shp_buffers:  # SHP 파일이 생성되어 포함된 경우
    main_zip_label = "📥 시각화자료+분석보고서+SHP (ZIP)"
    main_zip_help = "분석 리포트, 모든 분석도(PNG), 모든 원본 분석 파일(TIF), 벡터 데이터(SHP)를 한번에 다운로드합니다."

# 주 ZIP 다운로드를 먼저 추가
download_items.append({
    "label": main_zip_label,
    "data": zip_buffer,
    "file_name": f"analysis_results_{base_filename}_{timestamp}.zip",
    "mime": "application/zip",
    "key": "main_zip_download",
    "help": main_zip_help
})

# 개별 SHP 다운로드는 이제 주 zip에 포함되므로 비활성화됩니다.
# for analysis_type, data in shp_buffers.items():
#     download_items.append({
#         "label": f"📥 {data['title']} (SHP)",
#         "data": data['buffer'],
#         "file_name": f"{analysis_type}_{base_filename}_{timestamp}.zip",
#         "mime": "application/zip",
#         "key": f"shp_download_bottom_{analysis_type}",
#         "help": f"{data['title']} 분석 결과를 SHP 파일로 다운로드합니다."
#     })

# --- 열 생성 및 버튼 표시 ---
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
                help=item["help"]
            )

st.markdown("")  # 스페이서

# --- 최종 버튼 ---
if st.button("새로운 분석 시작하기", use_container_width=True):
    # 세션 상태를 지우기 전에 임시 TIF 파일 정리
    if 'dem_results' in st.session_state:
        for analysis_type in st.session_state.dem_results:
            results = st.session_state.dem_results.get(analysis_type, {})
            tif_path = results.get('tif_path')
            if tif_path and Path(tif_path).exists():
                try:
                    Path(tif_path).unlink()
                except OSError as e:
                    st.warning(f".tif 파일을 삭제하는 데 실패했습니다: {e}")

    for key in list(st.session_state.keys()):
        if key not in ['upload_counter']:
            del st.session_state[key]
    st.switch_page("app.py")