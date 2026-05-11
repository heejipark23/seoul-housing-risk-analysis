from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

CROSSWALK_PATH = DATA_DIR / "raw" / "seoul_legal_dong_information_202506.csv"
LEGAL_POPULATION_PATH = DATA_DIR / "raw" / "legal_dong_population.csv"
GEOJSON_PATH = DATA_DIR / "raw" / "seoul_neighborhoods_geo_simple_2015.geojson"

CONCENTRATION_PATH = OUTPUT_DIR / "seoul_legal_dong_concentration.csv"
RENT_PATH = OUTPUT_DIR / "전월세_법정동별_구간별_상승률_거래건수_통합_법정동코드수정_전체포함.csv"
PRIORITY_SCORE_PATH = OUTPUT_DIR / "ldri" / "rist_index_score.csv"
LDRI_MASTER_PATH = OUTPUT_DIR / "ldri" / "01_master_dataset.csv"
HOTSPOT_PATH = OUTPUT_DIR / "ldri" / "05_hotspot_result.csv"

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr")


def read_csv_safely(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"CSV encoding could not be resolved: {path}",
    ) from last_error


def normalize_code(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_code_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def unique_list(values: pd.Series) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def load_legal_admin_crosswalk() -> pd.DataFrame:
    required = ["시도명", "시군구명", "행정동명", "법정동명", "행정동코드", "법정동코드", "연결번호"]
    crosswalk = read_csv_safely(CROSSWALK_PATH)
    missing = [column for column in required if column not in crosswalk.columns]
    if missing:
        raise ValueError(f"행정동-법정동 연결표 필수 컬럼 누락: {missing}")

    crosswalk = crosswalk.loc[crosswalk["시도명"].astype(str).str.strip() == "서울특별시", required].copy()
    crosswalk = crosswalk.drop_duplicates()
    crosswalk = crosswalk.drop_duplicates(
        subset=["시군구명", "행정동명", "법정동명", "행정동코드", "법정동코드"]
    )

    for column in ["시도명", "시군구명", "행정동명", "법정동명"]:
        crosswalk[column] = crosswalk[column].astype(str).str.strip()
    for column in ["행정동코드", "법정동코드", "연결번호"]:
        crosswalk[column] = normalize_code_series(crosswalk[column])

    crosswalk = crosswalk.loc[~crosswalk["법정동코드"].str.endswith("00000")].copy()
    crosswalk["법정동코드8"] = crosswalk["법정동코드"].str[:8]

    legal_admin_counts = (
        crosswalk.groupby(["시군구명", "법정동코드"], as_index=False)["행정동코드"]
        .nunique()
        .rename(columns={"행정동코드": "행정동_연결수"})
    )
    admin_legal_counts = (
        crosswalk.groupby(["시군구명", "행정동코드"], as_index=False)["법정동코드"]
        .nunique()
        .rename(columns={"법정동코드": "법정동_연결수"})
    )
    crosswalk = crosswalk.merge(legal_admin_counts, on=["시군구명", "법정동코드"], how="left")
    crosswalk = crosswalk.merge(admin_legal_counts, on=["시군구명", "행정동코드"], how="left")

    return crosswalk.sort_values(["시군구명", "행정동명", "법정동명", "법정동코드"]).reset_index(drop=True)


def load_legal_population_weights() -> pd.DataFrame:
    weights = read_csv_safely(LEGAL_POPULATION_PATH)
    required = ["법정동코드", "기준연월", "시도명", "시군구명", "읍면동명", "계"]
    missing = [column for column in required if column not in weights.columns]
    if missing:
        raise ValueError(f"법정동 인구 가중치 필수 컬럼 누락: {missing}")

    weights = weights.loc[weights["시도명"].astype(str).str.strip() == "서울특별시", required].copy()
    weights["법정동코드"] = normalize_code_series(weights["법정동코드"])
    weights["법정동코드8"] = weights["법정동코드"].str[:8]
    weights["법정동_인구가중치"] = numeric_series(weights["계"])
    result = (
        weights.groupby("법정동코드8", as_index=False)["법정동_인구가중치"]
        .sum()
        .sort_values("법정동코드8")
    )
    return result


def build_legal_admin_display(crosswalk: pd.DataFrame) -> pd.DataFrame:
    display = (
        crosswalk.groupby(["시군구명", "법정동코드", "법정동코드8", "법정동명"], as_index=False)
        .agg(
            행정동목록=("행정동명", unique_list),
            행정동코드목록=("행정동코드", unique_list),
            행정동수=("행정동코드", "nunique"),
        )
        .sort_values(["시군구명", "법정동코드"])
    )

    def make_label(row: pd.Series) -> str:
        admins = row["행정동목록"]
        if not admins:
            return str(row["법정동명"])
        if len(admins) <= 3:
            return f"{row['법정동명']} · {', '.join(admins)}"
        return f"{row['법정동명']} · 행정동 {len(admins)}개"

    display["표시명"] = display.apply(make_label, axis=1)
    display["행정동_요약"] = display["행정동목록"].apply(lambda values: ", ".join(values))
    return display


def load_rent_data() -> pd.DataFrame:
    rent = read_csv_safely(RENT_PATH)
    rent["법정동코드"] = normalize_code_series(rent["법정동코드"])
    rent["법정동코드8"] = rent["법정동코드"].str[:8]
    for column in rent.columns:
        if any(token in column for token in ["임대료", "거래건수", "상승률"]):
            rent[column] = numeric_series(rent[column])
    return rent


def build_legal_dataset(crosswalk: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    concentration = read_csv_safely(CONCENTRATION_PATH)
    priority = read_csv_safely(PRIORITY_SCORE_PATH)
    master = read_csv_safely(LDRI_MASTER_PATH)
    hotspot = read_csv_safely(HOTSPOT_PATH)
    display = build_legal_admin_display(crosswalk)

    for frame in [concentration, priority, master, hotspot, display]:
        if "법정동코드" in frame.columns:
            frame["법정동코드"] = normalize_code_series(frame["법정동코드"])
            frame["법정동코드8"] = frame["법정동코드"].str[:8]

    priority = priority.rename(
        columns={
            "순위": "우선지원순위",
            "LDRI_raw": "우선지원_raw",
            "점수": "보조점수",
            "총점_raw": "우선지원총점_raw",
            "총점": "우선지원점수",
        }
    )
    master = master.rename(
        columns={
            "CI": "LDRI_CI",
            "LDRI_raw": "LDRI_raw_CI_x_주거비",
            "LDRI": "LDRI_표시점수",
        }
    )
    hotspot = hotspot.rename(columns={"LDRI": "핫스팟_LDRI"})

    keep_priority = [
        "법정동코드",
        "우선지원순위",
        "우선지원_raw",
        "보조점수",
        "우선지원총점_raw",
        "우선지원점수",
    ]
    keep_master = [
        "법정동코드",
        "관측품질",
        "주거비_압박지수",
        "D1_score",
        "LDRI_raw_CI_x_주거비",
        "LDRI_표시점수",
        "LDRI_rank",
        "LDRI_등급",
        "위험유형",
        "차원비교유형",
    ]
    keep_hotspot = ["법정동코드", "neighbor_count", "gi_star_z", "gi_star_p", "hotspot_class"]

    legal = concentration.merge(priority[keep_priority], on="법정동코드", how="left")
    legal = legal.merge(master[keep_master], on="법정동코드", how="left")
    legal = legal.merge(hotspot[keep_hotspot], on="법정동코드", how="left", suffixes=("", "_hotspot"))
    legal = legal.merge(display, on=["시군구명", "법정동코드", "법정동코드8", "법정동명"], how="left")
    legal = legal.merge(weights, on="법정동코드8", how="left")

    numeric_columns = [
        "총인구수",
        "청년인구수",
        "청년비율",
        "총가구수",
        "1인가구수",
        "1인가구비율",
        "기초수급가구수",
        "기초수급가구비율",
        "concentration_index",
        "concentration_rank",
        "우선지원순위",
        "우선지원점수",
        "우선지원_raw",
        "주거비_압박지수",
        "LDRI_표시점수",
        "LDRI_rank",
        "법정동_인구가중치",
        "행정동수",
        "gi_star_z",
        "gi_star_p",
    ]
    for column in numeric_columns:
        if column in legal.columns:
            legal[column] = numeric_series(legal[column])

    legal["법정동_인구가중치"] = legal["법정동_인구가중치"].fillna(1)
    legal["관측품질"] = legal["관측품질"].fillna("주거비자료없음")
    legal["위험유형"] = legal["위험유형"].fillna("자료없음").replace("", "자료없음")
    legal["hotspot_class"] = legal["hotspot_class"].fillna("자료없음")
    legal["행정동목록"] = legal["행정동목록"].apply(lambda value: value if isinstance(value, list) else [])
    legal["행정동_요약"] = legal["행정동_요약"].fillna("")
    legal["표시명"] = legal["표시명"].fillna(legal["법정동명"])
    legal["is_top30"] = legal["우선지원순위"].le(30)

    return legal.sort_values(["우선지원순위", "시군구명", "법정동명"], na_position="last").reset_index(drop=True)


def build_admin_policy_score(crosswalk: pd.DataFrame, legal: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "법정동코드",
        "법정동코드8",
        "법정동명",
        "우선지원점수",
        "우선지원순위",
        "우선지원_raw",
        "법정동_인구가중치",
        "관측품질",
        "위험유형",
        "주거비_압박지수",
        "concentration_index",
        "is_top30",
    ]
    joined = crosswalk.merge(legal[fields], on=["법정동코드", "법정동코드8", "법정동명"], how="left")
    joined["우선지원점수"] = numeric_series(joined["우선지원점수"])
    joined["법정동_인구가중치"] = numeric_series(joined["법정동_인구가중치"]).fillna(1)
    joined["score_weight"] = joined["우선지원점수"] * joined["법정동_인구가중치"]

    rows: list[dict[str, Any]] = []
    for (gu, admin_code, admin_name), group in joined.groupby(["시군구명", "행정동코드", "행정동명"]):
        scored = group.loc[group["우선지원점수"].notna()].copy()
        observed = group.loc[group["관측품질"].eq("주거비관측")]
        if scored.empty:
            policy_score = float("nan")
            max_score = float("nan")
            representative = ""
        else:
            denominator = scored["법정동_인구가중치"].sum()
            policy_score = scored["score_weight"].sum() / denominator if denominator else scored["우선지원점수"].mean()
            max_row = scored.sort_values(["우선지원점수", "우선지원순위"], ascending=[False, True]).iloc[0]
            max_score = max_row["우선지원점수"]
            representative = str(max_row["법정동명"])

        rows.append(
            {
                "시군구명": gu,
                "행정동코드": admin_code,
                "행정동명": admin_name,
                "행정동_정책점수": policy_score,
                "행정동_최고위험점수": max_score,
                "대표_고위험법정동": representative,
                "연결_법정동수": int(group["법정동코드"].nunique()),
                "Top30_법정동수": int(group.loc[group["is_top30"].fillna(False), "법정동코드"].nunique()),
                "주거비자료_커버리지": len(observed["법정동코드"].drop_duplicates()) / max(group["법정동코드"].nunique(), 1),
                "연결_법정동목록": ", ".join(unique_list(group["법정동명"])),
                "주요_위험유형": group["위험유형"].dropna().mode().iloc[0] if not group["위험유형"].dropna().empty else "자료없음",
            }
        )

    admin = pd.DataFrame(rows)
    admin = admin.sort_values(["행정동_정책점수", "행정동_최고위험점수"], ascending=[False, False], na_position="last")
    admin["행정동_정책순위"] = range(1, len(admin) + 1)
    return admin.reset_index(drop=True)


def load_geojson() -> dict[str, Any]:
    with GEOJSON_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_all_data() -> dict[str, Any]:
    crosswalk = load_legal_admin_crosswalk()
    weights = load_legal_population_weights()
    legal = build_legal_dataset(crosswalk, weights)
    admin_policy = build_admin_policy_score(crosswalk, legal)
    rent = load_rent_data()
    geojson = load_geojson()

    quality = {
        "법정동수": int(crosswalk[["시군구명", "법정동코드"]].drop_duplicates().shape[0]),
        "행정동수": int(crosswalk[["시군구명", "행정동코드"]].drop_duplicates().shape[0]),
        "연결쌍수": int(crosswalk[["시군구명", "행정동코드", "법정동코드"]].drop_duplicates().shape[0]),
        "다중행정동_법정동수": int(
            crosswalk.groupby(["시군구명", "법정동코드"])["행정동코드"].nunique().gt(1).sum()
        ),
        "Top30_매핑누락수": int(
            legal.loc[legal["우선지원순위"].le(30), "행정동목록"].apply(lambda values: len(values) == 0).sum()
        ),
    }

    return {
        "crosswalk": crosswalk,
        "legal": legal,
        "admin_policy": admin_policy,
        "rent": rent,
        "geojson": geojson,
        "quality": quality,
    }
