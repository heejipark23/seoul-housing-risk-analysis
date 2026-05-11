# %% [markdown]
# # 법정동별 기초 주거취약성 데이터셋 전처리
#
# - 목적: 행정동 단위 인구, 가구, 기초생활수급 가구 데이터를 법정동 기준으로 변환한다.
# - 기준: 법정동별 총인구를 가중치로 행정동 값을 인구비례 분할하며, 소수점 이하는 제외한다.
# - 최종 산출물: `data/processed/legal_dong_risk_base_2024.csv`

# %%
from pathlib import Path
import re

import numpy as np
import pandas as pd


# %% [markdown]
# ## 1. 기본 경로와 상수 정의

# %%
def find_project_root():
    """스크립트/노트북 실행 위치가 달라도 프로젝트 루트를 찾는다."""
    starts = []
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    starts.append(Path.cwd().resolve())

    seen = set()
    for start in starts:
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "data").exists() and (candidate / "notebooks").exists():
                return candidate

    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

POPULATION_PATH = RAW_DIR / "population_by_age.csv"
HOUSEHOLDS_PATH = RAW_DIR / "sgis_households_2024_adm_dong.csv"
BASIC_LIVELIHOOD_PATH = RAW_DIR / "number_of_basic_livelihood_recipient.csv"
MAPPING_PATH = RAW_DIR / "seoul_legal_dong_information_202506.csv"
LEGAL_DONG_POPULATION_PATH = RAW_DIR / "legal_dong_population.csv"

FINAL_OUTPUT_PATH = PROCESSED_DIR / "legal_dong_risk_base_2024.csv"
ALLOCATION_CHECK_PATH = OUTPUT_DIR / "legal_dong_allocation_check.csv"
SUMMARY_PATH = REPORTS_DIR / "preprocessing_summary.md"

BASE_YEAR = 2024
BASE_YEARMONTH = str(BASE_YEAR)
LEGAL_DONG_POPULATION_BASE_DATE = f"{BASE_YEAR}-12-31"
MISSING_LEGAL_DONG_WEIGHT = 1.0
POPULATION_BASE_MONTH_COL = "2024.12 월"
YOUTH_AGE_STANDARD = "20-39세"
YOUTH_AGE_LABELS = {"20-24세", "25-29세", "30-34세", "35-39세"}

SEOUL_GU_LIST = [
    "종로구",
    "중구",
    "용산구",
    "성동구",
    "광진구",
    "동대문구",
    "중랑구",
    "성북구",
    "강북구",
    "도봉구",
    "노원구",
    "은평구",
    "서대문구",
    "마포구",
    "양천구",
    "강서구",
    "구로구",
    "금천구",
    "영등포구",
    "동작구",
    "관악구",
    "서초구",
    "강남구",
    "송파구",
    "강동구",
]


# %% [markdown]
# ## 2. 공통 유틸리티 함수

# %%
def ensure_directories():
    """전처리 산출물 저장에 필요한 폴더를 생성한다."""
    for directory in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, REPORTS_DIR, NOTEBOOKS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[폴더 확인] {directory}")


def safe_read_csv(path, dtype=str):
    """여러 인코딩 후보를 순서대로 시도해 CSV를 안전하게 읽는다."""
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding, dtype=dtype, keep_default_na=False, low_memory=False)
            print(f"[파일 읽기] {path.name}: encoding={encoding}, shape={df.shape}")
            return df, encoding
        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"{path} 파일을 지원 인코딩으로 읽지 못했습니다. 마지막 오류: {last_error}",
    )


def normalize_dong_name(value):
    """행정동명과 법정동명의 표기 차이를 줄이기 위한 정규화 함수."""
    if pd.isna(value):
        return ""

    name = str(value).strip()
    name = re.sub(r"\([^)]*\)|\[[^\]]*\]|（[^）]*）", "", name)
    name = re.sub(r"[ㆍ.・･∙‧]", "·", name)
    name = re.sub(r"\s+", "", name)
    name = re.sub(r"홍제제(?=\d)", "홍제", name)
    name = re.sub(r"(?<!홍)제(?=\d)", "", name)
    return name.strip()


def clean_numeric(series):
    """쉼표, 공백, 결측 표기를 제거하고 숫자형으로 변환한다."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"": "0", "-": "0", "nan": "0", "NaN": "0"})
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0)


def normalize_code(series):
    """행정동코드, 법정동코드를 문자열 코드로 유지한다."""
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def safe_ratio(numerator, denominator):
    """분모가 0이면 0, 아니면 numerator / denominator를 반환한다."""
    numerator = pd.to_numeric(numerator, errors="coerce").fillna(0)
    denominator = pd.to_numeric(denominator, errors="coerce").fillna(0)
    return np.where(denominator > 0, numerator / denominator, 0)


def save_csv(df, path):
    """모든 CSV 산출물을 utf-8-sig로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[저장] {path} shape={df.shape}")


def dataframe_to_markdown(df):
    """추가 의존성 없이 DataFrame을 Markdown 표 문자열로 변환한다."""
    if df.empty:
        return "데이터 없음"

    table = df.copy()
    table = table.astype(str).replace({"\n": " "}, regex=True)
    columns = list(table.columns)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in table.iterrows():
        values = [str(row[col]).replace("|", "/") for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


# %% [markdown]
# ## 3. 행정동-법정동 연결 정보 전처리

# %%
def preprocess_mapping(path=MAPPING_PATH):
    required_cols = [
        "시도명",
        "시군구명",
        "행정동명",
        "법정동명",
        "행정동코드",
        "법정동코드",
        "연결번호",
    ]

    raw, _ = safe_read_csv(path)
    raw.columns = raw.columns.str.strip()

    missing_cols = [col for col in required_cols if col not in raw.columns]
    if missing_cols:
        raise ValueError(f"행정동-법정동 연결표 필수 컬럼 누락: {missing_cols}")

    mapping = raw.loc[raw["시도명"].str.strip() == "서울특별시", required_cols].copy()
    mapping = mapping.drop_duplicates()
    mapping = mapping.drop_duplicates(
        subset=["시군구명", "행정동명", "법정동명", "행정동코드", "법정동코드"]
    )

    for col in ["시도명", "시군구명", "행정동명", "법정동명"]:
        mapping[col] = mapping[col].astype(str).str.strip()

    for col in ["행정동코드", "법정동코드", "연결번호"]:
        mapping[col] = normalize_code(mapping[col])

    mapping = mapping.loc[~mapping["법정동코드"].str.endswith("00000")].copy()

    mapping["행정동명_정규화"] = mapping["행정동명"].apply(normalize_dong_name)
    mapping["법정동명_정규화"] = mapping["법정동명"].apply(normalize_dong_name)

    admin_legal_counts = (
        mapping.groupby(["시군구명", "행정동명_정규화"])["법정동코드"]
        .nunique()
        .reset_index(name="법정동_연결수")
    )
    legal_admin_counts = (
        mapping.groupby(["시군구명", "법정동코드"])["행정동명_정규화"]
        .nunique()
        .reset_index(name="행정동_연결수")
    )

    mapping = mapping.merge(admin_legal_counts, on=["시군구명", "행정동명_정규화"], how="left")
    mapping = mapping.merge(legal_admin_counts, on=["시군구명", "법정동코드"], how="left")

    print("[매핑 전처리 완료]")
    print(f"- 연결표 행 수: {len(mapping):,}")
    print(f"- 행정동 수: {mapping[['시군구명', '행정동명_정규화']].drop_duplicates().shape[0]:,}")
    print(f"- 법정동 수: {mapping[['시군구명', '법정동코드']].drop_duplicates().shape[0]:,}")

    return mapping


# %% [markdown]
# ## 4. 법정동 인구 가중치 전처리

# %%
def preprocess_legal_dong_population(path=LEGAL_DONG_POPULATION_PATH):
    """법정동별 총인구를 행정동-법정동 인구비례 분할 가중치로 정리한다."""
    required_cols = ["법정동코드", "기준연월", "시도명", "시군구명", "읍면동명", "계"]

    raw, _ = safe_read_csv(path)
    raw.columns = raw.columns.str.strip()

    missing_cols = [col for col in required_cols if col not in raw.columns]
    if missing_cols:
        raise ValueError(f"법정동 인구 가중치 필수 컬럼 누락: {missing_cols}")

    population = raw.loc[
        (raw["시도명"].astype(str).str.strip() == "서울특별시")
        & (raw["기준연월"].astype(str).str.strip() == LEGAL_DONG_POPULATION_BASE_DATE),
        required_cols,
    ].copy()

    if population.empty:
        raise ValueError(
            f"법정동 인구 가중치에 서울특별시 {LEGAL_DONG_POPULATION_BASE_DATE} 자료가 없습니다."
        )

    for col in ["시도명", "시군구명", "읍면동명"]:
        population[col] = population[col].astype(str).str.strip()

    population["법정동코드"] = normalize_code(population["법정동코드"])
    population["법정동코드8"] = population["법정동코드"].str[:8]
    population["법정동_인구가중치"] = clean_numeric(population["계"])

    weights = (
        population.groupby(["법정동코드8"], as_index=False)["법정동_인구가중치"]
        .sum()
        .sort_values("법정동코드8")
    )

    print("[법정동 인구 가중치 전처리 완료]")
    print(f"- 기준일: {LEGAL_DONG_POPULATION_BASE_DATE}")
    print(f"- 법정동 수: {len(weights):,}")
    print(f"- 법정동 인구 합계: {weights['법정동_인구가중치'].sum():,.0f}")

    return weights


# %% [markdown]
# ## 5. 연령별 인구 데이터 전처리

# %%
def preprocess_population(path=POPULATION_PATH):
    raw, _ = safe_read_csv(path)
    raw.columns = raw.columns.str.strip()
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")].copy()

    required_cols = ["행정구역(동읍면)별", "5세별", "항목", POPULATION_BASE_MONTH_COL]
    missing_cols = [col for col in required_cols if col not in raw.columns]
    if missing_cols:
        raise ValueError(f"인구 데이터 필수 컬럼 누락: {missing_cols}")

    df = raw.loc[raw["항목"].astype(str).str.strip() == "총인구수[명]"].copy()
    df["지역명"] = df["행정구역(동읍면)별"].astype(str).str.strip()

    current_gu = None
    gu_values = []
    is_admin_dong = []

    for region_name in df["지역명"]:
        if region_name == "서울특별시":
            current_gu = None
            gu_values.append("")
            is_admin_dong.append(False)
        elif region_name in SEOUL_GU_LIST:
            current_gu = region_name
            gu_values.append(region_name)
            is_admin_dong.append(False)
        else:
            gu_values.append(current_gu or "")
            is_admin_dong.append(bool(current_gu))

    df["시군구명"] = gu_values
    df = df.loc[is_admin_dong].copy()
    df["행정동명"] = df["지역명"]
    df["행정동명_정규화"] = df["행정동명"].apply(normalize_dong_name)
    df["연령구간_정규화"] = df["5세별"].astype(str).str.replace(r"\s+", "", regex=True)
    df["인구수"] = clean_numeric(df[POPULATION_BASE_MONTH_COL])

    key_cols = ["시군구명", "행정동명", "행정동명_정규화"]
    total_population = (
        df.loc[df["연령구간_정규화"] == "계"]
        .groupby(key_cols, as_index=False)["인구수"]
        .sum()
        .rename(columns={"인구수": "총인구수"})
    )
    youth_population = (
        df.loc[df["연령구간_정규화"].isin(YOUTH_AGE_LABELS)]
        .groupby(key_cols, as_index=False)["인구수"]
        .sum()
        .rename(columns={"인구수": "청년인구수"})
    )

    population = total_population.merge(youth_population, on=key_cols, how="outer").fillna(0)
    population["총인구수"] = np.floor(population["총인구수"]).astype("int64")
    population["청년인구수"] = np.floor(population["청년인구수"]).astype("int64")
    population = population.loc[
        ~((population["총인구수"] == 0) & (population["청년인구수"] == 0))
    ].copy()
    population["청년연령기준"] = YOUTH_AGE_STANDARD
    population["기준연월"] = BASE_YEARMONTH

    population = population[
        [
            "시군구명",
            "행정동명",
            "행정동명_정규화",
            "총인구수",
            "청년인구수",
            "청년연령기준",
            "기준연월",
        ]
    ].sort_values(["시군구명", "행정동명"])

    print("[인구 데이터 전처리 완료]")
    print(population.head())
    print(f"- 행정동 수: {len(population):,}")
    print(f"- 총인구수 합계: {population['총인구수'].sum():,}")
    print(f"- 청년인구수 합계: {population['청년인구수'].sum():,}")

    return population


# %% [markdown]
# ## 5. 가구 데이터 전처리

# %%
def preprocess_households(path=HOUSEHOLDS_PATH):
    raw, _ = safe_read_csv(path)
    raw.columns = raw.columns.str.strip()

    required_cols = ["year", "sgg_nm", "adm_nm", "total_households", "one_person_households"]
    missing_cols = [col for col in required_cols if col not in raw.columns]
    if missing_cols:
        raise ValueError(f"가구 데이터 필수 컬럼 누락: {missing_cols}")

    df = raw.loc[clean_numeric(raw["year"]) == BASE_YEAR].copy()
    df = df.rename(
        columns={
            "sgg_nm": "시군구명",
            "adm_nm": "행정동명",
            "total_households": "총가구수",
            "one_person_households": "1인가구수",
        }
    )

    df["시군구명"] = df["시군구명"].astype(str).str.strip()
    df["행정동명"] = df["행정동명"].astype(str).str.strip()
    df = df.loc[~df["행정동명"].isin(["합계", "소계", "기타"])].copy()
    df["행정동명_정규화"] = df["행정동명"].apply(normalize_dong_name)
    df["총가구수"] = np.floor(clean_numeric(df["총가구수"])).astype("int64")
    df["1인가구수"] = np.floor(clean_numeric(df["1인가구수"])).astype("int64")
    df["기준연도"] = BASE_YEAR

    households = (
        df.groupby(["시군구명", "행정동명", "행정동명_정규화", "기준연도"], as_index=False)[
            ["총가구수", "1인가구수"]
        ]
        .sum()
        .sort_values(["시군구명", "행정동명"])
    )

    print("[가구 데이터 전처리 완료]")
    print(households.head())
    print(f"- 행정동 수: {len(households):,}")
    print(f"- 총가구수 합계: {households['총가구수'].sum():,}")
    print(f"- 1인가구수 합계: {households['1인가구수'].sum():,}")

    return households


# %% [markdown]
# ## 6. 기초생활수급 가구 데이터 전처리

# %%
def preprocess_basic_livelihood(path=BASIC_LIVELIHOOD_PATH):
    raw, _ = safe_read_csv(path)
    raw.columns = raw.columns.str.strip()

    required_cols = ["자치구", "행정동", "총 수급가구_2024"]
    missing_cols = [col for col in required_cols if col not in raw.columns]
    if missing_cols:
        raise ValueError(f"기초생활수급 데이터 필수 컬럼 누락: {missing_cols}")

    df = raw.loc[raw["자치구"].astype(str).str.strip() != "합계"].copy()
    df = df.loc[~df["행정동"].astype(str).str.strip().isin(["소계", "기타"])].copy()
    df = df.rename(
        columns={
            "자치구": "시군구명",
            "행정동": "행정동명",
            "총 수급가구_2024": "기초수급가구수",
        }
    )

    df["시군구명"] = df["시군구명"].astype(str).str.strip()
    df["행정동명"] = df["행정동명"].astype(str).str.strip()
    df["행정동명_정규화"] = df["행정동명"].apply(normalize_dong_name)
    df["기초수급가구수"] = np.floor(clean_numeric(df["기초수급가구수"])).astype("int64")
    df["기준연도"] = BASE_YEAR

    basic_livelihood = (
        df.groupby(["시군구명", "행정동명", "행정동명_정규화", "기준연도"], as_index=False)[
            ["기초수급가구수"]
        ]
        .sum()
        .sort_values(["시군구명", "행정동명"])
    )

    print("[기초생활수급 데이터 전처리 완료]")
    print(basic_livelihood.head())
    print(f"- 행정동 수: {len(basic_livelihood):,}")
    print(f"- 기초수급가구수 합계: {basic_livelihood['기초수급가구수'].sum():,}")

    return basic_livelihood


# %% [markdown]
# ## 8. 행정동 데이터를 법정동 인구비례로 분할

# %%
def allocate_admin_values_to_legal_dong(
    adm_df,
    mapping_df,
    legal_population_weights_df,
    value_cols,
    dataset_name,
):
    """행정동 단위 count 변수를 법정동 인구 가중치 비율로 나누고 법정동 기준으로 집계한다."""
    merge_keys = ["시군구명", "행정동명_정규화"]
    mapping_cols = [
        "시도명",
        "시군구명",
        "행정동명_정규화",
        "법정동명",
        "법정동코드",
    ]

    adm = adm_df.copy()
    for col in value_cols:
        adm[col] = clean_numeric(adm[col])

    weights = legal_population_weights_df[["법정동코드8", "법정동_인구가중치"]].copy()
    weights["법정동코드8"] = normalize_code(weights["법정동코드8"])
    weights["법정동_인구가중치"] = clean_numeric(weights["법정동_인구가중치"])

    allocation_mapping = mapping_df[mapping_cols].drop_duplicates().copy()
    allocation_mapping["법정동코드"] = normalize_code(allocation_mapping["법정동코드"])
    allocation_mapping["법정동코드8"] = allocation_mapping["법정동코드"].str[:8]
    allocation_mapping = allocation_mapping.merge(weights, on="법정동코드8", how="left")
    allocation_mapping["인구가중치_누락"] = allocation_mapping["법정동_인구가중치"].isna()
    allocation_mapping["법정동_분할가중치"] = allocation_mapping["법정동_인구가중치"].fillna(
        MISSING_LEGAL_DONG_WEIGHT
    )

    if (allocation_mapping["법정동_분할가중치"] < 0).any():
        raise ValueError("법정동 분할 가중치에 음수 값이 있습니다.")

    merged = adm.merge(
        allocation_mapping,
        on=merge_keys,
        how="left",
        suffixes=("_원본", "_매핑"),
        indicator=True,
    )

    unmatched_mask = merged["_merge"] == "left_only"
    unmatched_cols = [col for col in adm_df.columns if col in merged.columns]
    unmatched_df = (
        merged.loc[unmatched_mask, unmatched_cols]
        .drop_duplicates()
        .assign(데이터셋=dataset_name)
        .sort_values(["시군구명", "행정동명"])
    )

    matched = merged.loc[~unmatched_mask].copy()
    matched["행정동_가중치합계"] = matched.groupby(merge_keys)["법정동_분할가중치"].transform("sum")
    zero_weight_admin_count = matched.loc[
        matched["행정동_가중치합계"] <= 0, merge_keys
    ].drop_duplicates().shape[0]
    if zero_weight_admin_count:
        raise ValueError(f"{dataset_name}: 분할 가중치 합계가 0인 행정동이 있습니다.")

    matched = matched.loc[matched["행정동_가중치합계"] > 0].copy()
    matched["분할비율"] = matched["법정동_분할가중치"] / matched["행정동_가중치합계"]

    for col in value_cols:
        matched[f"{col}_분할"] = np.floor(matched[col] * matched["분할비율"]).astype("int64")

    group_cols = ["시도명", "시군구명", "법정동명", "법정동코드"]
    allocated_cols = [f"{col}_분할" for col in value_cols]

    if matched.empty:
        legal_df = pd.DataFrame(columns=group_cols + value_cols)
    else:
        legal_df = matched.groupby(group_cols, as_index=False)[allocated_cols].sum()
        legal_df = legal_df.rename(columns={f"{col}_분할": col for col in value_cols})
        legal_df["법정동코드"] = normalize_code(legal_df["법정동코드"])

    check_records = []
    for col in value_cols:
        original_sum = int(np.floor(adm[col]).sum())
        allocated_sum = int(legal_df[col].sum()) if col in legal_df.columns else 0
        difference = original_sum - allocated_sum
        difference_rate = difference / original_sum if original_sum else 0

        if matched.empty:
            discarded_by_floor = 0
        else:
            matched_admin = matched[merge_keys + value_cols].drop_duplicates(subset=merge_keys)
            matched_value_sum = int(np.floor(matched_admin[col]).sum())
            discarded_by_floor = int(
                matched_value_sum - matched[f"{col}_분할"].sum()
            )

        unmatched_value_sum = int(np.floor(unmatched_df[col]).sum()) if col in unmatched_df.columns else 0
        missing_weight_legal_count = (
            matched.loc[matched["인구가중치_누락"], "법정동코드8"].drop_duplicates().shape[0]
            if not matched.empty
            else 0
        )

        check_records.append(
            {
                "데이터셋": dataset_name,
                "변수": col,
                "분할방식": "법정동인구비례",
                "원본_행정동_합계": original_sum,
                "법정동_분할후_합계": allocated_sum,
                "차이": difference,
                "차이율": round(difference_rate, 6),
                "분할과정_버려진값": discarded_by_floor,
                "누락가중치_법정동수": missing_weight_legal_count,
                "가중치합계0_행정동수": zero_weight_admin_count,
                "병합실패_행정동수": unmatched_df[merge_keys].drop_duplicates().shape[0],
                "병합실패_값합계": unmatched_value_sum,
            }
        )

        if difference_rate >= 0.01:
            print(
                f"[경고] {dataset_name} - {col}: 원본 대비 차이율이 {difference_rate:.2%}입니다."
            )

    if not unmatched_df.empty:
        print(f"[경고] {dataset_name}: 병합 실패 행정동 {len(unmatched_df):,}건")

    allocation_check_df = pd.DataFrame(check_records)

    print(f"[법정동 분할 완료] {dataset_name}")
    print(f"- 법정동 행 수: {len(legal_df):,}")
    print(
        f"- 기본 가중치 적용 법정동 수: "
        f"{matched.loc[matched['인구가중치_누락'], '법정동코드8'].drop_duplicates().shape[0] if not matched.empty else 0:,}"
    )
    print(f"- 병합 실패 행정동 수: {unmatched_df[merge_keys].drop_duplicates().shape[0]:,}")

    return legal_df, unmatched_df, allocation_check_df


# %% [markdown]
# ## 9. 최종 법정동 기준 데이터 병합

# %%
def make_legal_dong_base(mapping_df):
    base = (
        mapping_df[
            [
                "시도명",
                "시군구명",
                "법정동명",
                "법정동코드",
                "행정동_연결수",
            ]
        ]
        .drop_duplicates(subset=["시군구명", "법정동코드"])
        .rename(columns={"행정동_연결수": "연결된_행정동수"})
        .copy()
    )
    base["법정동코드"] = normalize_code(base["법정동코드"])
    base["연결된_행정동수"] = np.floor(clean_numeric(base["연결된_행정동수"])).astype("int64")
    return base


def merge_allocated_dataset(base_df, allocated_df, value_cols):
    merge_keys = ["시군구명", "법정동명", "법정동코드"]

    if allocated_df.empty:
        allocated = pd.DataFrame(columns=merge_keys + value_cols)
    else:
        allocated = allocated_df[merge_keys + value_cols].copy()
        allocated["법정동코드"] = normalize_code(allocated["법정동코드"])

    merged = base_df.merge(allocated, on=merge_keys, how="left")
    for col in value_cols:
        merged[col] = np.floor(clean_numeric(merged[col])).astype("int64")

    return merged


def build_final_dataset(mapping_df, population_legal, households_legal, basic_legal):
    final = make_legal_dong_base(mapping_df)
    final = merge_allocated_dataset(final, population_legal, ["총인구수", "청년인구수"])
    final = merge_allocated_dataset(final, households_legal, ["총가구수", "1인가구수"])
    final = merge_allocated_dataset(final, basic_legal, ["기초수급가구수"])

    count_cols = ["총인구수", "청년인구수", "총가구수", "1인가구수", "기초수급가구수"]
    for col in count_cols:
        final[col] = np.floor(clean_numeric(final[col])).astype("int64")

    final["기준연월"] = BASE_YEARMONTH
    final["청년연령기준"] = YOUTH_AGE_STANDARD
    final["청년비율"] = safe_ratio(final["청년인구수"], final["총인구수"])
    final["1인가구비율"] = safe_ratio(final["1인가구수"], final["총가구수"])
    final["기초수급가구비율"] = safe_ratio(final["기초수급가구수"], final["총가구수"])

    final = final[
        [
            "시도명",
            "시군구명",
            "법정동명",
            "법정동코드",
            "기준연월",
            "청년연령기준",
            "총인구수",
            "청년인구수",
            "청년비율",
            "총가구수",
            "1인가구수",
            "1인가구비율",
            "기초수급가구수",
            "기초수급가구비율",
            "연결된_행정동수",
        ]
    ].sort_values(["시군구명", "법정동코드", "법정동명"])

    final["법정동코드"] = normalize_code(final["법정동코드"])

    return final


# %% [markdown]
# ## 9. 품질 점검

# %%
def run_quality_checks(final_df, allocation_check_df, unmatched_dfs):
    count_cols = ["총인구수", "총가구수", "기초수급가구수"]
    all_count_cols = ["총인구수", "청년인구수", "총가구수", "1인가구수", "기초수급가구수"]
    ratio_cols = ["청년비율", "1인가구비율", "기초수급가구비율"]

    duplicate_legal_code_count = int(final_df["법정동코드"].duplicated().sum())
    missing_counts = final_df.isna().sum()
    negative_counts = {
        col: int((pd.to_numeric(final_df[col], errors="coerce").fillna(0) < 0).sum())
        for col in count_cols
    }
    ratio_out_of_range_counts = {
        col: int(((final_df[col] < 0) | (final_df[col] > 1)).sum()) for col in ratio_cols
    }
    all_zero_rows = int((final_df[all_count_cols].sum(axis=1) == 0).sum())
    unmatched_counts = {
        name: df[["시군구명", "행정동명_정규화"]].drop_duplicates().shape[0] if not df.empty else 0
        for name, df in unmatched_dfs.items()
    }

    print("\n[품질 점검 결과]")
    print(f"1. 법정동코드 중복 수: {duplicate_legal_code_count:,}")
    print("2. 결측치 개수:")
    print(missing_counts[missing_counts > 0] if (missing_counts > 0).any() else "   결측치 없음")
    print(f"3. 음수 값 개수: {negative_counts}")
    print(f"4. 비율 범위 이탈 개수: {ratio_out_of_range_counts}")
    print(f"5. 모든 count 값이 0인 법정동 수: {all_zero_rows:,}")
    print(f"6. 병합 실패 행정동 수: {unmatched_counts}")
    print("7. 원본 행정동 합계와 법정동 분할 후 합계 차이:")
    print(allocation_check_df)

    return {
        "duplicate_legal_code_count": duplicate_legal_code_count,
        "missing_counts": missing_counts,
        "negative_counts": negative_counts,
        "ratio_out_of_range_counts": ratio_out_of_range_counts,
        "all_zero_rows": all_zero_rows,
        "unmatched_counts": unmatched_counts,
    }


# %% [markdown]
# ## 10. 전처리 요약 문서 작성

# %%
def write_preprocessing_summary(final_df, allocation_check_df, quality_result):
    """전처리 결과와 품질 점검 결과를 Markdown으로 저장한다."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    missing_counts = quality_result["missing_counts"]
    missing_df = (
        missing_counts[missing_counts > 0]
        .rename_axis("컬럼")
        .reset_index(name="결측치수")
    )
    if missing_df.empty:
        missing_text = "결측치 없음"
    else:
        missing_text = dataframe_to_markdown(missing_df)

    ratio_summary = final_df[
        ["청년비율", "1인가구비율", "기초수급가구비율"]
    ].agg(["min", "mean", "max"]).round(6).reset_index().rename(columns={"index": "통계"})

    lines = [
        "# 법정동별 기초 주거취약성 데이터셋 전처리 요약",
        "",
        "## 산출물",
        f"- 최종 CSV: `{FINAL_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- 행/열: {final_df.shape[0]:,}행 x {final_df.shape[1]:,}열",
        f"- 기준연월: {BASE_YEARMONTH}",
        f"- 청년 연령 기준: {YOUTH_AGE_STANDARD}",
        "",
        "## 품질 점검",
        f"- 법정동코드 중복 수: {quality_result['duplicate_legal_code_count']:,}",
        f"- 모든 count 값이 0인 법정동 수: {quality_result['all_zero_rows']:,}",
        f"- 음수 값 개수: {quality_result['negative_counts']}",
        f"- 비율 범위 이탈 개수: {quality_result['ratio_out_of_range_counts']}",
        f"- 병합 실패 행정동 수: {quality_result['unmatched_counts']}",
        "",
        "## 결측치",
        missing_text,
        "",
        "## 비율 변수 요약",
        dataframe_to_markdown(ratio_summary),
        "",
        "## 행정동-법정동 분할 합계 점검",
        dataframe_to_markdown(allocation_check_df),
        "",
    ]

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[저장] {SUMMARY_PATH}")

# %% [markdown]
# ## 11. 전체 전처리 실행 함수

# %%
def main():
    ensure_directories()

    mapping = preprocess_mapping()
    legal_population_weights = preprocess_legal_dong_population()
    population = preprocess_population()
    households = preprocess_households()
    basic_livelihood = preprocess_basic_livelihood()

    population_legal, unmatched_population, population_check = allocate_admin_values_to_legal_dong(
        adm_df=population,
        mapping_df=mapping,
        legal_population_weights_df=legal_population_weights,
        value_cols=["총인구수", "청년인구수"],
        dataset_name="population",
    )
    households_legal, unmatched_households, households_check = allocate_admin_values_to_legal_dong(
        adm_df=households,
        mapping_df=mapping,
        legal_population_weights_df=legal_population_weights,
        value_cols=["총가구수", "1인가구수"],
        dataset_name="households",
    )
    basic_legal, unmatched_basic, basic_check = allocate_admin_values_to_legal_dong(
        adm_df=basic_livelihood,
        mapping_df=mapping,
        legal_population_weights_df=legal_population_weights,
        value_cols=["기초수급가구수"],
        dataset_name="basic_livelihood",
    )

    allocation_check = pd.concat(
        [population_check, households_check, basic_check],
        ignore_index=True,
    )
    save_csv(allocation_check, ALLOCATION_CHECK_PATH)

    final = build_final_dataset(
        mapping_df=mapping,
        population_legal=population_legal,
        households_legal=households_legal,
        basic_legal=basic_legal,
    )
    save_csv(final, FINAL_OUTPUT_PATH)

    quality_result = run_quality_checks(
        final_df=final,
        allocation_check_df=allocation_check,
        unmatched_dfs={
            "population": unmatched_population,
            "households": unmatched_households,
            "basic_livelihood": unmatched_basic,
        },
    )
    write_preprocessing_summary(final, allocation_check, quality_result)

    print("\n[전처리 완료]")
    print(f"- 최종 파일: {FINAL_OUTPUT_PATH}")
    print(f"- 최종 데이터 크기: {final.shape[0]:,}행 x {final.shape[1]:,}열")
    print(f"- 주요 컬럼: {list(final.columns)}")

    return final, allocation_check, quality_result


# %% [markdown]
# ## 12. 스크립트 실행
#
# 아래 블록은 Python 파일로 실행할 때 전체 전처리를 수행한다.
# VS Code Jupyter 셀에서 단계별 실행할 경우 `main()`을 직접 호출하면 된다.

# %%
if __name__ == "__main__":
    main()
