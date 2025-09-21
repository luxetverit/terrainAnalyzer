import os
import sys
from pathlib import Path
import platform
import pyproj
import streamlit as st

# --- PROJ Data Directory Configuration (Cross-Platform Final Version) ---
try:
    conda_prefix = Path(sys.prefix)
    if platform.system() == "Windows":
        proj_data_dir = conda_prefix / "Library" / "share" / "proj"
    else:
        proj_data_dir = conda_prefix / "share" / "proj"

    if proj_data_dir.exists():
        pyproj.datadir.set_data_dir(str(proj_data_dir))
    # No else needed, pyproj will try its default finding mechanism.
except Exception:
    # Fails silently if something goes wrong, as it might not be a critical error.
    pass
# --- End of Configuration ---

from utils.file_processor import validate_file
from utils.theme_util import apply_styles


# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# --- 1. Page Configuration and Styling ---
st.set_page_config(
    page_title="지형 분석 서비스",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply global styles from main.css
apply_styles()

# --- 2. Session State Initialization ---
if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0

# --- 3. Page Header ---
st.markdown(
    """
<div class="page-header">
    <h1>지형 분석 서비스</h1>
    <p>반복되는 기초자료조사, 이제는 자동화하세요.<br>SHP 또는 DXF 파일을 업로드하여 간편하게 지형 분석을 시작할 수 있습니다.</p>
</div>
""",
    unsafe_allow_html=True,
)

# --- 4. Main Content ---

st.subheader("1. 분석할 파일 업로드")

# Use a unique key for the file uploader to allow re-uploads
upload_key = f"file_uploader_{st.session_state.upload_counter}"
uploaded_file = st.file_uploader(
    "조사하고 싶은 공간의 SHP 파일(ZIP으로 압축)이나, DXF 파일을 업로드해주세요.",
    type=["dxf", "zip"],
    key=upload_key,
    label_visibility="visible",  # Make label visible for clarity
)

st.subheader("2. 원본 좌표계 선택")
st.markdown("업로드한 자료의 좌표계를 선택해주세요. (기본값: EPSG:5186)")

epsg_options = {
    "EPSG:5186 / Korea 2000 / Central Belt 2010": 5186,
    "EPSG:5183 / Korea 2000 / East Belt": 5183,
    "EPSG:5185 / Korea 2000 / West Belt 2010": 5185,
    "EPSG:5187 / Korea 2000 / East Belt 2010": 5187,
    "EPSG:5179 / Korea 2000 / Unified CS": 5179,
    "EPSG:5174 / Korea 1985 / Central Belt": 5174,
    "EPSG:4326 / WGS84": 4326,
}
selected_epsg_name = st.selectbox(
    "EPSG 선택", options=list(epsg_options.keys()), index=0
)
epsg_code = epsg_options[selected_epsg_name]

# --- 5. File Processing and Navigation ---
temp_file_path_for_next = None
if uploaded_file:
    logging.info(f"--- 파일 업로드 감지: {uploaded_file.name} ---")
    with st.spinner("파일 유효성 검사 중..."):
        logging.info("validate_file 함수 호출 시작")
        is_valid, message, temp_file_path = validate_file(uploaded_file)
        logging.info(f"validate_file 함수 반환: is_valid={is_valid}, message={message}")

    if is_valid:
        temp_file_path_for_next = temp_file_path
        st.success(f"**{uploaded_file.name}** 파일이 성공적으로 업로드되었습니다.")
        st.info("좌표계를 확인하고 '다음 단계' 버튼을 클릭하여 분석을 계속하세요.")
    else:
        st.error(f"파일 오류: {message}")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("다음 단계로 이동", use_container_width=True, type="primary"):
    if uploaded_file and temp_file_path_for_next:
        # Store necessary info in session state for the next page
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.temp_file_path = temp_file_path_for_next
        st.session_state.epsg_code = epsg_code
        st.session_state.selected_epsg_name = selected_epsg_name

        # Increment counter for next upload
        st.session_state.upload_counter += 1

        st.switch_page("pages/01_기초분석.py")
    else:
        st.warning("분석을 진행하려면 먼저 유효한 파일을 업로드해야 합니다.")
