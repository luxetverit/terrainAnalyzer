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
from rasterio.mask import mask
from rasterio.transform import from_origin
import io
import tempfile
from sqlalchemy import create_engine

from utils.color_palettes import ALL_PALETTES, get_palette_preview_html
from utils.theme_util import apply_theme_toggle

engine = create_engine("postgresql://postgres:asdfasdf12@localhost:5432/gisdb")

# ----- Streamlit 페이지/폰트 세팅 -----
pixel_size = 1
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
else:
    plt.rc("font", family="AppleGothic")
plt.rcParams["axes.unicode_minus"] = False

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

# ----- 세션 상태 체크 -----
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
        "palettes": {"elevation": "terrain", "slope": "RdYlBu"},
    }

elevation_palette = st.session_state.analysis_results["palettes"]["elevation"]
slope_palette = st.session_state.analysis_results["palettes"]["slope"]


# ----- 좌표 문자열 파싱 함수 -----
def parse_coordinate_string(coord_str):
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


# ----- 모든 geometry에서 point 추출 -----
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
        elif geom.geom_type == "LineString":
            for pt in geom.coords:
                xs.append(pt[0])
                ys.append(pt[1])
                zs.append(z)
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                for pt in line.coords:
                    xs.append(pt[0])
                    ys.append(pt[1])
                    zs.append(z)
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
    return np.array(xs), np.array(ys), np.array(zs)


# ----- 표고 통계 -----
def calc_elevation_stats(array):
    arr = array[np.isfinite(array)]
    return {
        "min": float(np.nanmin(arr)) if arr.size > 0 else np.nan,
        "max": float(np.nanmax(arr)) if arr.size > 0 else np.nan,
        "mean": float(np.nanmean(arr)) if arr.size > 0 else np.nan,
        "area": arr.size,
    }


# ----- 캐시/임시파일 적극 활용 -----
@st.cache_data(show_spinner=False)
def merge_and_standardize(
    user_shp_path,
    sample_shp_files,
    altitude_cols,
    geometry_cols,
    elevation_field,
    target_crs,
):
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
                gdf["geometry"] = gdf["geometry"].apply(parse_coordinate_string)
            found_col = next((col for col in altitude_cols if col in gdf.columns), None)
            if found_col and found_col != elevation_field:
                gdf = gdf.rename(columns={found_col: elevation_field})
            if elevation_field not in gdf.columns:
                gdf[elevation_field] = np.nan
            gdf = gdf.set_geometry("geometry")
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            gdf = gdf.to_crs(target_crs)
            gdf = gdf[gdf["geometry"].notna() & (~gdf["geometry"].is_empty)]
            gdf = gdf[~gdf[elevation_field].isna()]
            dfs.append(gdf[["geometry", elevation_field]])
        except Exception as e:
            st.warning(f"{shp_path} 읽기 실패: {e}")
    if dfs:
        merged_gdf = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=dfs[0].crs)
        return merged_gdf
    else:
        return None


def merge_and_standardize_gdf_list(
    gdf_list, altitude_cols, geometry_cols, elevation_field, target_crs
):
    dfs = []
    for gdf in gdf_list:
        try:
            # geometry 컬럼명 표준화
            geom_col = next((col for col in geometry_cols if col in gdf.columns), None)
            if geom_col and geom_col != "geometry":
                gdf = gdf.rename(columns={geom_col: "geometry"})
            found_col = next((col for col in altitude_cols if col in gdf.columns), None)
            if found_col and found_col != elevation_field:
                gdf = gdf.rename(columns={found_col: elevation_field})
            if elevation_field not in gdf.columns:
                gdf[elevation_field] = np.nan
            gdf = gdf.set_geometry("geometry")
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            gdf = gdf.to_crs(target_crs)
            gdf = gdf[gdf["geometry"].notna() & (~gdf["geometry"].is_empty)]
            gdf = gdf[~gdf[elevation_field].isna()]
            dfs.append(gdf[["geometry", elevation_field]])
        except Exception as e:
            st.warning(f"gdf 병합 실패: {e}")
    if dfs:
        merged_gdf = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=dfs[0].crs)
        return merged_gdf
    else:
        return None


# ---- 데이터 준비 (임시파일/캐시 사용) ----
with st.expander("🔎 업로드/샘플 도엽 병합 + DEM 보간 및 클리핑 결과", expanded=True):

    user_shp_path = st.session_state.get(
        "temp_file_path", None
    ) or st.session_state.get("uploaded_file_path", None)
    if not user_shp_path or not os.path.exists(user_shp_path):
        st.error(
            "업로드 폴리곤 shp 파일 경로가 없습니다. (session_state['temp_file_path'])"
        )
        st.stop()

    sample_shp_files = glob.glob(r"C:\dev\sample2\*.shp")
    if not sample_shp_files:
        st.error("C:\\dev\\sample2 폴더에 도엽 샘플 shp 파일이 없습니다.")
        st.stop()

    user_gdf = gpd.read_file(user_shp_path)

    sheet_str = ",".join([f"'{x}'" for x in sample_sheet_codes])
    sql = f"""
    SELECT geometry, elevation
    FROM nationwide_shp
    WHERE sheet_code IN ({sheet_str})
    """

    # GeoDataFrame으로 읽기
    sample_gdf = gpd.read_postgis(sql, engine, geom_col="geometry")

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
    elevation_field = "elevation"
    target_crs = "EPSG:5186"

    merged_gdf = merge_and_standardize(
        user_shp_path,
        sample_shp_files,
        altitude_cols,
        geometry_cols,
        elevation_field,
        target_crs,
    )
    if merged_gdf is None or len(merged_gdf) == 0:
        st.error("도엽 병합 실패 또는 데이터 없음.")
        st.stop()
    st.success(f"업로드+도엽 병합 완료! 총 {len(merged_gdf)}개 객체")

    # ---- DEM 보간 및 tif 임시저장 ----
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
        tif_path = tmpfile.name

    with st.spinner("DEM 보간 중..."):
        xs, ys, zs = extract_points(merged_gdf, elevation_field)
        minx, miny, maxx, maxy = merged_gdf.total_bounds
        grid_x, grid_y = np.mgrid[minx:maxx:pixel_size, miny:maxy:pixel_size]
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
            dst.write(np.flipud(dem_grid.T), 1)
        st.success("DEM 보간 완료!")

    # ---- 클리핑 ----
    with st.spinner("사용자 폴리곤으로 DEM 클리핑 중..."):
        user_gdf = gpd.read_file(user_shp_path).to_crs(merged_gdf.crs)
        clip_geoms = [g for g in user_gdf.geometry if g.is_valid and not g.is_empty]
        with rasterio.open(tif_path) as src:
            clipped, clipped_transform = mask(src, clip_geoms, crop=True, nodata=np.nan)
            clipped_dem = clipped[0]
        st.success("클리핑 완료!")

    # ---- 통계 session_state 저장 ----
    stats = calc_elevation_stats(clipped_dem)
    st.session_state["dem_results"] = {
        "elevation": {"stats": stats, "grid": clipped_dem}
    }

# ----- 시각화/컬러맵/다운로드 -----
st.markdown("## 작업이 완료되었어요")
selected_types = st.session_state.get("selected_analysis_types", [])

if "dem_results" in st.session_state and st.session_state.dem_results:
    dem_results = st.session_state.dem_results
    if "elevation" in selected_types and dem_results["elevation"]["stats"]:
        elev_stats = dem_results["elevation"]["stats"]
        st.markdown(
            f"## 표고는 {elev_stats['min']:.1f}~{elev_stats['max']:.1f}m, 평균표고는 {elev_stats['mean']:.1f}m로 분석되었어요."
        )

# ---- 컬러맵별 탭 시각화 및 다운로드 ----
elev_cmaps = ["terrain", "gist_earth", "viridis", "Spectral"]
if "elevation" in selected_types:
    st.markdown("### 표고 분석 컬러맵 미리보기")
    tabs = st.tabs(elev_cmaps)
    mask = ~np.isnan(clipped_dem)
    if not np.any(mask):
        st.warning("클리핑 결과가 없습니다.")
    else:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]
        clipped_cropped = clipped_dem[min_row : max_row + 1, min_col : max_col + 1]
        for i, cmap in enumerate(elev_cmaps):
            with tabs[i]:
                fig, ax = plt.subplots(figsize=(7, 4))
                im = ax.imshow(np.ma.masked_invalid(clipped_cropped), cmap=cmap)
                plt.colorbar(im, ax=ax, label="Elevation(m)")
                ax.set_title(f"클리핑된 DEM - {cmap}")
                ax.axis("off")
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                buf.seek(0)
                plt.close(fig)
                image_bytes = buf.getvalue()
                st.image(image_bytes, caption=f"{cmap} 시각화", width=480)
                st.download_button(
                    label=f"{cmap} PNG 다운로드",
                    data=image_bytes,
                    file_name=f"dem_{cmap}.png",
                    mime="image/png",
                    key=f"download_{cmap}",
                )

# ---- 예시: 추가 분석 유형 등... ----
# slope 등 다른 유형도 위 방식대로 쉽게 확장 가능

# ---- 다운로드/마무리 ----
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="결과(샘플) 다운로드",
        data="샘플 결과 데이터",
        file_name="result.txt",
        mime="text/plain",
        key="download_result_txt",
    )

st.markdown("---")
st.markdown("# 다운로드 완료!")
st.markdown("## 또초자료 운사원을 찾아주셔서 감사해요!")
st.markdown("## 다음에 또 기초자료 조사가 필요하시면 언제든지 방문해주세요!")
st.markdown("---")
st.markdown(
    """
### 자료 출처:
- DEM - 국토지리정보원
- 토지피복도 - 환경부
- 정밀토양도 - 농촌진흥청
"""
)
st.markdown(
    """
또초자료를 이용하시면서 불편한 사항이 발생하거나
개선 또는 다시 해주었으면 하는 자료가 있으면 (메일주소) 로 문의주세요!
제가 운영진님께 잘 전달해드릴게요
"""
)
st.markdown("---")
st.markdown("Published by Edward Yoon", unsafe_allow_html=True)
