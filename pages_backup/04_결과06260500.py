import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import glob
import pandas as pd
import re
from shapely.geometry import Point
from scipy.interpolate import griddata
import platform
import rasterio
from rasterio.features import rasterize
import io
from utils.dem_processor import clip_tif_by_polygon

from utils.visualization import create_elevation_heatmap
from utils.color_palettes import ALL_PALETTES, get_palette_preview_html
from utils.theme_util import apply_theme_toggle
from utils.dem_processor import (
    interpolate_dem_from_points,
    clip_dem_by_polygon,
    calc_elevation_stats,
)

pixel_size = 1
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
else:
    plt.rc("font", family="AppleGothic")  # macOS
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 페이지 설정/테마 ----------------
st.set_page_config(
    page_title="또초자료 운사원 - 결과",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
main_col = apply_theme_toggle()
st.markdown(
    """
<style>
.big-text { font-size: 24px !important; font-weight: bold; }
.result-text { font-size: 18px !important; font-weight: bold; }
.stButton>button {
    width: 100%;
    border-radius: 20px !important;
    font-size: 18px !important;
    padding: 10px 24px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------- 파일 체크/상태 체크 -------------
if "uploaded_file" not in st.session_state or st.session_state.uploaded_file is None:
    st.error("업로드된 파일이 없습니다. 메인 페이지로 돌아가세요.")
    if st.button("메인 페이지로 돌아가기"):
        st.switch_page("app.py")
    st.stop()
if "processing_done" not in st.session_state:
    st.error("처리가 완료되지 않았습니다. 처리 중 페이지로 돌아가세요.")
    if st.button("이전 페이지로 돌아가기"):
        st.switch_page("pages/03_처리중.py")
    st.stop()
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {
        "done": True,
        "message": "분석이 완료되었습니다!",
        "palettes": {"elevation": "spectral", "slope": "terrain"},
    }


def parse_coordinate_string(coord_str):
    # 위도, 경도 분리
    match = re.match(r"([\d\-. ]+)[, ]+([\d\-. ]+)", coord_str)
    if not match:
        return None
    lat_str, lon_str = match.group(1), match.group(2)

    def dms_to_deg(s):
        s = (
            s.replace(",", " ")
            .replace("-", " ")
            .replace("°", " ")
            .replace("'", " ")
            .replace('"', " ")
        )
        parts = [float(x) for x in re.split(r"\s+", s.strip()) if x]
        if len(parts) == 3:
            deg, minute, sec = parts
            return deg + (minute / 60) + (sec / 3600)
        elif len(parts) == 2:
            deg, minute = parts
            return deg + (minute / 60)
        else:
            return float(parts[0])

    lat = dms_to_deg(lat_str)
    lon = dms_to_deg(lon_str)
    return Point(lon, lat)


# ---------------------------------------------------------------------
# ✅ DEM/폴리곤 도엽 병합 + 표고 컬럼 통일 + DEM 보간/클리핑 (주요 변경부분)
# ---------------------------------------------------------------------
with st.expander("🔎 업로드/샘플 도엽 병합 + DEM 보간 및 클리핑 결과", expanded=True):

    # 1) 업로드 파일 경로(폴리곤 shp)
    # 아래 변수명은 실제 사용 환경에 맞게!
    user_shp_path = st.session_state.get(
        "temp_file_path", None
    ) or st.session_state.get("uploaded_file_path", None)
    if not user_shp_path or not os.path.exists(user_shp_path):
        st.error(
            "업로드 폴리곤 shp 파일 경로가 없습니다. (session_state['temp_file_path'])"
        )
        st.stop()

    # 2) C:\dev\sample2 폴더의 도엽 SHP들 전부
    sample_shp_files = glob.glob(r"C:\dev\sample2\*.shp")
    if not sample_shp_files:
        st.error("C:\\dev\\sample2 폴더에 도엽 샘플 shp 파일이 없습니다.")
        st.stop()

    # 3) 도엽 + 업로드 파일 병합 및 표고 컬럼 통일
    altitude_cols = [
        "표고",
        "등고수치",
        "수치",
        "고도",
        "elev",
        "elevation",
        "height",
        "Z",
    ]
    geometry_cols = ["geometry", "Geometry", "GEOMETRY", "geom", "Geom", "좌표"]
    elevation_field = "표고"
    target_crs = "EPSG:5186"
    dfs = []
    for shp_path in [user_shp_path] + sample_shp_files:
        try:
            gdf = gpd.read_file(shp_path)
            geom_col = next((col for col in geometry_cols if col in gdf.columns), None)
            if geom_col and geom_col != "geometry":
                gdf = gdf.rename(columns={geom_col: "geometry"})
            if "geometry" not in gdf.columns and "좌표" in gdf.columns:
                gdf["geometry"] = gdf["좌표"].apply(parse_coordinate_string)
            elif gdf["geometry"].dtype == "object" and all(
                isinstance(val, str) for val in gdf["geometry"]
            ):
                # geometry가 문자열이면 Point로 변환
                gdf["geometry"] = gdf["geometry"].apply(parse_coordinate_string)

            # 표고 컬럼 표준화
            found_col = next((col for col in altitude_cols if col in gdf.columns), None)
            if found_col and found_col != elevation_field:
                gdf = gdf.rename(columns={found_col: elevation_field})
            if elevation_field not in gdf.columns:
                gdf[elevation_field] = np.nan

            # geometry 타입 지정
            gdf = gdf.set_geometry("geometry")

            # 좌표계
            if gdf.crs is None:
                # 경도/위도일 가능성 → WGS84 임의 지정, 이후 target_crs로 변환
                gdf.set_crs("EPSG:4326", inplace=True)
            gdf = gdf.to_crs(target_crs)

            # geometry 유효성 및 NaN/Empty 제거
            gdf = gdf[gdf["geometry"].notna() & (~gdf["geometry"].is_empty)]

            # 표고 NaN은 제외(DEM에서는 NaN이 있으면 보간이 안됨)
            gdf = gdf[~gdf[elevation_field].isna()]

            dfs.append(gdf[["geometry", elevation_field]])
        except Exception as e:
            st.warning(f"{shp_path} 읽기 실패: {e}")
    if dfs:
        merged_gdf = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=dfs[0].crs)
    else:
        st.error("도엽 병합 실패")
        st.stop()

    st.success(f"업로드+도엽 병합 완료! 총 {len(merged_gdf)}개 객체")

    elevation_field = "표고"

    def extract_points(gdf, elevation_field):
        xs, ys, zs = [], [], []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            z = row[elevation_field]
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "Point":
                xs.append(geom.x)
                ys.append(geom.y)
                zs.append(z)
            elif geom.geom_type == "MultiPoint":
                for pt in geom.geoms:
                    xs.append(pt.x)
                    ys.append(pt.y)
                    zs.append(z)
            elif geom.geom_type in ["LineString"]:
                for pt in geom.coords:
                    xs.append(pt[0])
                    ys.append(pt[1])
                    zs.append(z)
            elif geom.geom_type in ["MultiLineString"]:
                for line in geom.geoms:
                    for pt in line.coords:
                        xs.append(pt[0])
                        ys.append(pt[1])
                        zs.append(z)
            # Polygon에서 centroid, boundary 등 원하는 처리 방식에 따라 추가
            elif geom.geom_type == "Polygon":
                for pt in geom.exterior.coords:
                    xs.append(pt[0])
                    ys.append(pt[1])
                    zs.append(z)
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    for pt in poly.exterior.coords:
                        xs.append(pt[0])
                        ys.append(pt[1])
                        zs.append(z)
            # 필요한 다른 geometry 타입이 있다면 여기에 추가
        return np.array(xs), np.array(ys), np.array(zs)

    # [4] DEM 보간 (griddata)
    from rasterio.transform import from_origin
    import tempfile
    from rasterio.mask import mask

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
        tif_path = tmpfile.name

    # DEM 보간 (결과: dem_grid, transform, crs)
    with st.spinner("DEM 보간 중..."):
        try:
            xs, ys, zs = extract_points(merged_gdf, elevation_field)

            pixel_size = 1
            minx, miny, maxx, maxy = merged_gdf.total_bounds
            grid_x, grid_y = np.mgrid[
                minx:maxx:pixel_size, miny:maxy:pixel_size
            ]  # (Y가 위로)

            dem_grid = griddata(
                (xs, ys), zs, (grid_x, grid_y), method="linear", fill_value=np.nan
            )

            transform = from_origin(minx, maxy, pixel_size, pixel_size)
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=dem_grid.shape[1],
                width=dem_grid.shape[0],
                count=1,
                dtype="float32",
                crs=merged_gdf.crs,
                transform=transform,
                nodata=np.nan,
            ) as dst:
                dst.write(np.flipud(dem_grid.T), 1)  # rasterio는 행-열 순서 반대

            # dem_grid, transform, crs = interpolate_dem_from_points(merged_gdf, elev_col, pixel_size)
            st.success("DEM 보간 완료!")
        except Exception as e:
            st.error(f"DEM 보간 오류: {e}")
            st.stop()

    # 사용자가 올린 shp로 클리핑
    with st.spinner("사용자 폴리곤으로 DEM 클리핑 중..."):
        try:
            user_gdf = gpd.read_file(user_shp_path)
            user_shp = gpd.read_file(user_shp_path).to_crs(merged_gdf.crs)
            clip_geoms = [g for g in user_shp.geometry if g.is_valid and not g.is_empty]

            with rasterio.open(tif_path) as src:
                clipped, clipped_transform = mask(
                    src, clip_geoms, crop=True, nodata=np.nan
                )
                clipped_dem = clipped[0]
            st.success("클리핑 완료!")
        except Exception as e:
            st.error(f"DEM 클리핑 오류: {e}")
            st.stop()

    # 결과 stats 저장 및 session_state
    stats = calc_elevation_stats(clipped_dem)
    st.session_state["dem_results"] = {
        "elevation": {"stats": stats, "grid": clipped_dem}
    }

st.markdown("## 작업이 완료되었어요")
selected_types = st.session_state.get("selected_analysis_types", [])

elevation_palette = st.session_state.analysis_results["palettes"]["elevation"]
slope_palette = st.session_state.analysis_results["palettes"]["slope"]

# 결과 요약 (실제 분석 결과 표시)
if "dem_results" in st.session_state and st.session_state.dem_results:
    dem_results = st.session_state.dem_results

    # 표고 통계 표시
    if "elevation" in selected_types and dem_results["elevation"]["stats"]:
        elev_stats = dem_results["elevation"]["stats"]
        st.markdown(
            f"## 표고는 {elev_stats['min']:.1f}~{elev_stats['max']:.1f}m, 평균표고는 {elev_stats['mean']:.1f}m로 분석되었어요."
        )

        # # 클리핑 영역 시각화 (crop 후 흰 공간 최소화)
        # mask = ~np.isnan(clipped_dem)
        # if not np.any(mask):
        #     st.warning("클리핑 결과가 없습니다.")
        # else:
        #     fig, ax = plt.subplots(figsize=(8, 6))  # 크기 더 줄이고 싶으면 조정
        #     im = ax.imshow(np.ma.masked_invalid(clipped_dem), cmap='terrain')
        #     plt.colorbar(im, ax=ax, label="Elevation(m)")
        #     ax.set_title("클리핑된 DEM")
        #     ax.axis('off')
        #     buf = io.BytesIO()
        #     plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
        #     buf.seek(0)
        #     plt.close(fig)
        #     st.image(buf, width=600)  # 원하는 px로 조절

    # 경사 통계 표시
    if "slope" in selected_types and dem_results["slope"]["stats"]:
        slope_stats = dem_results["slope"]["stats"]
        st.markdown(
            f"## 경사는 {slope_stats['min']:.1f}~{slope_stats['max']:.1f}도, 평균경사 {slope_stats['mean']:.1f}도로 분석되었어요."
        )

        # 경사 등급별 면적 정보가 있으면 표시
        if "area_by_class" in slope_stats:
            # 가장 높은 비율을 가진 등급 찾기
            max_area_class = max(
                slope_stats["area_by_class"].items(), key=lambda x: x[1]
            )
            max_area_pct = max_area_class[1] * 100
            st.markdown(
                f"## 경사는 {max_area_class[0]} 지역이 {max_area_pct:.1f}%로 대부분을 차지하네요."
            )

    # 지역 정보와 같은 추가 결과가 있는 경우 표시할 수 있음
    if "map_index_results" in st.session_state and st.session_state.map_index_results:
        matched_sheets = st.session_state.map_index_results.get("matched_sheets", [])
        if matched_sheets:
            st.markdown(f"## 총 {len(matched_sheets)}개 도엽을 기준으로 분석되었어요.")
else:
    # 분석 결과가 없는 경우
    st.markdown("## 표고는 00~00m, 평균표고는 00m로 분석되었어요.")
    st.markdown("## 경사는 0~0도, 평균경사 0도로 분석되었어요.")

# 선택하지 않은 분석 유형에 대한 메시지
other_types = [t for t in ["landcover", "soil", "hsg"] if t in selected_types]
if other_types:
    st.markdown("## 선택하신 다른 분석 결과도 아래에서 확인하실 수 있어요.")

# 결과 시각화 영역
st.markdown("### 결과사진 대표")

# 선택한 색상 팔레트 표시
# st.markdown("#### 선택하신 색상 팔레트")

# 색상 팔레트 미리보기 표시
# col1, col2 = st.columns(2)
# with col1:
#     if 'elevation' in selected_types:
#         st.markdown("##### 표고 분석 색상")
#         st.markdown(get_palette_preview_html(elevation_palette), unsafe_allow_html=True)
#         st.markdown(f"<div class='palette-label'>{ALL_PALETTES[elevation_palette]['name']}</div>", unsafe_allow_html=True)

# with col2:
#     if 'slope' in selected_types:
#         st.markdown("##### 경사 분석 색상")
#         st.markdown(get_palette_preview_html(slope_palette), unsafe_allow_html=True)
#         st.markdown(f"<div class='palette-label'>{ALL_PALETTES[slope_palette]['name']}</div>", unsafe_allow_html=True)

# # 탭으로 각 분석 결과 보여주기

elev_cmaps = ["terrain", "gist_earth", "viridis", "Spectral"]
slope_cmaps = ["RdYlBu", "coolwarm", "Spectral"]

if selected_types:
    for type_key in selected_types:
        if type_key == "elevation":
            st.markdown("### 표고 분석 컬러맵 미리보기")
            tabs = st.tabs(elev_cmaps)
            # 크롭 및 NaN 마스킹
            mask = ~np.isnan(clipped_dem)
            if not np.any(mask):
                st.warning("클리핑 결과가 없습니다.")
                continue
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            min_row, max_row = np.where(rows)[0][[0, -1]]
            min_col, max_col = np.where(cols)[0][[0, -1]]
            clipped_cropped = clipped_dem[min_row : max_row + 1, min_col : max_col + 1]
            for i, cmap in enumerate(elev_cmaps):
                with tabs[i]:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(np.ma.masked_invalid(clipped_cropped), cmap=cmap)
                    plt.colorbar(im, ax=ax, label="Elevation(m)")
                    ax.set_title(f"클리핑된 DEM - {cmap}")
                    ax.axis("off")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                    buf.seek(0)
                    plt.close(fig)
                    image_bytes = buf.getvalue()
                    st.image(image_bytes, caption=f"{cmap} 시각화", width=600)

                    # 다운로드 버튼
                    if st.button("다운로드 및 이동"):
                        # 다운로드 트리거 (실제 다운로드버튼은 별개로 써야 함)
                        st.download_button(
                            label=f"{cmap} 시각화 PNG 다운로드",
                            data=image_bytes,
                            file_name=f"dem_{cmap}.png",
                            mime="image/png",
                            key=f"download_{cmap}",
                        )
                        # 페이지 이동
                        st.switch_page("app.py")

        elif type_key == "slope":
            st.markdown("### 경사 분석 컬러맵 미리보기")
            tabs = st.tabs(slope_cmaps)
            # 임의의 data_array가 있다고 가정
            for i, cmap in enumerate(slope_cmaps):
                with tabs[i]:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(data_array, cmap=cmap, origin="lower")
                    plt.colorbar(im, ax=ax, label="Slope (deg)")
                    ax.set_title(f"경사 분석 결과 - {cmap}")
                    ax.axis("off")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                    buf.seek(0)
                    plt.close(fig)
                    st.image(buf, width=600)
        else:
            st.markdown(f"### {type_key} 분석 결과 (미리보기)")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            im = ax.imshow(data_array, cmap="viridis", origin="lower")
            plt.colorbar(im, ax=ax, label=f"{type_key}")
            ax.set_title(f"{type_key} 분석 결과")
            ax.axis("off")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            plt.close(fig)
            st.image(buf, width=400)
else:
    st.warning("선택된 분석 유형이 없습니다.")

# elev_cmaps = ['terrain', 'gist_earth', 'viridis']
# slope_cmaps = ['RdYlBu', 'coolwarm', 'Spectral']


# if selected_types:
#     tabs = st.tabs([ALL_PALETTES[elevation_palette]["name"] if type_key == "elevation" else
#                      ALL_PALETTES[slope_palette]["name"] if type_key == "slope" else
#                      type_key for type_key in selected_types])
#     tabs = st.tabs(elev_cmaps if type_key == 'elevation' else slope_cmaps)
#     for i, cmap in enumerate(elev_cmaps if type_key == 'elevation' else slope_cmaps):
#         with tabs[i]:
#             fig, ax = plt.subplots(figsize=(5, 2.5))
#             im = ax.imshow(dem_data, cmap=cmap)
#             plt.colorbar(im, ax=ax)
#             st.pyplot(fig)

#     for i, type_key in enumerate(selected_types):
#         with tabs[i]:
#             # 분석 유형에 따라 다른 시각화
#             if type_key == 'elevation':
#                 # 클리핑 영역 시각화 (crop 후 흰 공간 최소화)
#                 fig, ax = plt.subplots(figsize=(8, 6))  # 크기 더 줄이고 싶으면 조정
#                 im = ax.imshow(np.ma.masked_invalid(clipped_dem), cmap='terrain')
#                 plt.colorbar(im, ax=ax, label="Elevation(m)")
#                 ax.set_title("클리핑된 DEM")
#                 ax.axis('off')
#                 buf = io.BytesIO()
#                 plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
#                 buf.seek(0)
#                 plt.close(fig)
#                 st.image(buf, width=600)  # 원하는 px로 조절

#             elif type_key == 'slope':
#                 # 경사 시각화 (경사 팔레트 사용)
#                 fig, ax = plt.subplots(figsize=(10, 8))
#                 # 색상 팔레트 가져오기
#                 colors = ALL_PALETTES[slope_palette]['colors']
#                 # 경사 시각화
#                 im = ax.imshow(data_array, cmap=plt.cm.colors.ListedColormap(colors), origin='lower')
#                 plt.colorbar(im, ax=ax, label='경사 (도)')
#                 ax.set_title('경사 분석 결과', fontweight='bold', fontsize=14)
#                 ax.set_xlabel('X 좌표')
#                 ax.set_ylabel('Y 좌표')
#                 st.pyplot(fig)

#             else:
#                 # 기타 분석 유형 (기본 시각화)
#                 fig, ax = plt.subplots(figsize=(10, 8))
#                 im = ax.imshow(data_array, cmap='viridis', origin='lower')
#                 plt.colorbar(im, ax=ax, label=f'{type_key} 데이터')
#                 ax.set_title(f'{type_key} 분석 결과', fontweight='bold', fontsize=14)
#                 ax.set_xlabel('X 좌표')
#                 ax.set_ylabel('Y 좌표')
#                 st.pyplot(fig)
# else:
#     # 선택된 분석 유형이 없는 경우
#     st.warning("선택된 분석 유형이 없습니다.")

# 다운로드 버튼
col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="결과 다운로드",
        data="샘플 결과 데이터",
        file_name="result.txt",
        mime="text/plain",
        key="download_result",
    )

# 마무리 텍스트 및 버튼
st.markdown("---")
st.markdown("# 다운로드 완료!")
st.markdown("## 또초자료 운사원을 찾아주셔서 감사해요!")
st.markdown("## 다음에 또 기초자료 조사가 필요하시면 언제든지 방문해주세요!")

# 자료 출처 정보
st.markdown("---")
st.markdown(
    """
### 자료 출처:
- DEM - 국토지리정보원
- 토지피복도 - 환경부
- 정밀토양도 - 농촌진흥청
"""
)

# 문의 정보
st.markdown(
    """
또초자료를 이용하시면서 불편한 사항이 발생하거나
개선 또는 다시 해주었으면 하는 자료가 있으면 (메일주소) 로 문의주세요!
제가 운영진님께 잘 전달해드릴게요
"""
)

# 푸터
st.markdown("---")
st.markdown("Published by Edward Yoon", unsafe_allow_html=True)
