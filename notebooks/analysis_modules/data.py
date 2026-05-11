from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from .config import GEOJSON_PATH, GEOJSON_URL, INPUT_PATH, PROJECT_ROOT


def load_base_dataset(path: Path = INPUT_PATH) -> pd.DataFrame:
    required_cols = [
        "시도명",
        "시군구명",
        "법정동명",
        "법정동코드",
        "총인구수",
        "청년인구수",
        "청년비율",
        "총가구수",
        "1인가구수",
        "1인가구비율",
        "기초수급가구수",
        "기초수급가구비율",
    ]

    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"법정동코드": str})
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"입력 데이터 필수 컬럼 누락: {missing}")

    numeric_cols = [
        "총인구수",
        "청년인구수",
        "청년비율",
        "총가구수",
        "1인가구수",
        "1인가구비율",
        "기초수급가구수",
        "기초수급가구비율",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    valid = df.copy()
    valid["법정동코드"] = valid["법정동코드"].astype(str).str.replace(r"\.0$", "", regex=True)
    valid["법정동코드8"] = valid["법정동코드"].str[:8]

    if len(valid) != 467:
        raise ValueError(f"분석 대상 법정동 수가 467개가 아닙니다: {len(valid):,}개")
    if valid["법정동코드8"].duplicated().any():
        duplicated = valid.loc[valid["법정동코드8"].duplicated(), "법정동코드8"].tolist()
        raise ValueError(f"법정동코드 앞 8자리 중복 발생: {duplicated[:10]}")

    zero_population_count = int((valid["총인구수"] == 0).sum())
    print(
        f"[입력] {path.relative_to(PROJECT_ROOT)}: 원본 {len(df):,}행, "
        f"분석 대상 {len(valid):,}행, 총인구수 0 법정동 {zero_population_count:,}행"
    )
    return valid


def load_or_download_geojson(path: Path = GEOJSON_PATH) -> dict:
    if not path.exists():
        print(f"[다운로드] 서울 법정동 경계 GeoJSON: {GEOJSON_URL}")
        response = requests.get(GEOJSON_URL, timeout=30)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")

    with path.open(encoding="utf-8") as file:
        geojson = json.load(file)

    feature_count = len(geojson.get("features", []))
    print(f"[경계] {path.relative_to(PROJECT_ROOT)}: {feature_count:,}개 feature")
    return geojson


def validate_geojson_match(df: pd.DataFrame, geojson: dict) -> None:
    geo_codes = {str(feature["properties"]["EMD_CD"]) for feature in geojson["features"]}
    data_codes = set(df["법정동코드8"])
    missing_in_geo = sorted(data_codes - geo_codes)
    extra_in_geo = sorted(geo_codes - data_codes)

    if missing_in_geo or extra_in_geo:
        raise ValueError(
            "GeoJSON과 분석 데이터 매칭 실패: "
            f"missing_in_geo={missing_in_geo[:10]}, extra_in_geo={extra_in_geo[:10]}"
        )

    print("[경계 매칭] EMD_CD와 법정동코드 앞 8자리 467개 모두 매칭")
