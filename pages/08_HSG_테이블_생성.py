
import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from utils.config import get_db_engine
from utils.theme_util import apply_styles

# --- 페이지 설정 ---
st.set_page_config(page_title="HSG 조회 테이블 생성", page_icon="🛠️", layout="wide")
apply_styles()

# --- 1. 설정 ---

# hsg_lookup 데이터가 포함된 엑셀 파일 경로
EXCEL_FILE_PATH = r'C:\dev\terrainAnalyzer_st\utils\HG.xlsx'
# 생성할 테이블 이름
TARGET_TABLE_NAME = 'hsg_lookup'


# --- 2. 핵심 기능 함수 ---

@st.cache_data
def load_and_preview_data(file_path):
    """엑셀 파일을 읽고 필요한 데이터를 추출하여 미리보기를 생성합니다."""
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        # 필요한 컬럼만 선택하고 이름 변경 (엑셀 파일의 실제 컬럼명에 따라 조정될 수 있음)
        # 'SOILSY' -> 'soilsy', 'HG' -> 'hg'
        if 'SOILSY' in df.columns and 'HG' in df.columns:
            lookup_df = df[['SOILSY', 'HG']].copy()
            lookup_df.rename(
                columns={'SOILSY': 'soilsy', 'HG': 'hg'}, inplace=True)
            # 중복 및 누락 데이터 제거
            lookup_df.dropna(inplace=True)
            lookup_df.drop_duplicates(subset=['soilsy'], inplace=True)
            return lookup_df
        else:
            st.error("엑셀 파일에서 '토양통' 또는 'HG' 컬럼을 찾을 수 없습니다.")
            return None
    except Exception as e:
        st.error(f"엑셀 파일을 읽는 중 오류 발생: {e}")
        return None

# --- 3. Streamlit UI 구성 ---


st.markdown("### 🛠️ HSG 조회 테이블 생성")
st.write(f"""
이 페이지는 로컬의 엑셀 파일(`{os.path.basename(EXCEL_FILE_PATH)}`)을 읽어, 
데이터베이스에 **`{TARGET_TABLE_NAME}`** 테이블을 생성합니다.
""")
st.warning(f"""
**주의:** 만약 데이터베이스에 `{TARGET_TABLE_NAME}` 테이블이 이미 존재한다면, 
기존 테이블은 삭제되고 **새로운 데이터로 완전히 대체됩니다.**
""", icon="⚠️")

st.markdown("---")

# 데이터 미리보기
st.subheader("📄 미리보기: 엑셀 파일 데이터")
preview_df = load_and_preview_data(EXCEL_FILE_PATH)

if preview_df is not None:
    st.write(f"총 {len(preview_df)}개의 고유한 토양통-HG 매핑 정보를 찾았습니다.")
    st.dataframe(preview_df)

    st.markdown("---")

    if st.button(f"**'{TARGET_TABLE_NAME}' 테이블 생성 실행**", type="primary"):
        try:
            with st.spinner("데이터베이스에 연결하고 테이블을 생성하는 중..."):
                engine = get_db_engine()

                # pandas DataFrame을 SQL 테이블로 저장
                preview_df.to_sql(
                    name=TARGET_TABLE_NAME,
                    con=engine,
                    if_exists='replace',  # 테이블이 있으면 대체
                    index=False
                )

            st.success(
                f"🎉 성공! 데이터베이스에 `{TARGET_TABLE_NAME}` 테이블을 성공적으로 생성했습니다.")
            st.info("이제 다른 분석 페이지를 정상적으로 이용할 수 있습니다.")

        except Exception as e:
            st.error(f"테이블 생성 중 심각한 오류 발생: {e}")
else:
    st.error("데이터 미리보기에 실패하여 테이블을 생성할 수 없습니다. 엑셀 파일의 경로와 내용을 확인해주세요.")
