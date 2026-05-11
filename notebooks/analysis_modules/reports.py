from __future__ import annotations

import pandas as pd

from .common import dataframe_to_markdown
from .config import (
    CI_RANK_OUTPUT_PATH,
    CLUSTER_OUTPUT_PATH,
    CONCENTRATION_OUTPUT_PATH,
    FIGURE_DIR,
    GEOJSON_PATH,
    INPUT_PATH,
    NORM_COLS,
    PROJECT_ROOT,
    RATIO_COLS,
    SUMMARY_PATH,
    TOP30_OUTPUT_PATH,
)


def write_analysis_summary(
    df: pd.DataFrame,
    top30: pd.DataFrame,
    cluster_centers: pd.DataFrame,
    moran_summary: dict[str, float],
) -> None:
    top10 = top30.head(10)[
        [
            "priority_rank",
            "시군구명",
            "법정동명",
            "concentration_index",
            "cluster_label",
            "lisa_cluster",
        ]
    ].copy()
    top10["concentration_index"] = top10["concentration_index"].round(4)

    cluster_table = cluster_centers.copy()
    for col in RATIO_COLS + ["concentration_index"]:
        cluster_table[col] = cluster_table[col].round(4)

    lisa_counts = (
        df["lisa_cluster"].value_counts().rename_axis("lisa_cluster").reset_index(name="법정동수")
    )

    lines = [
        "# 서울시 청년·저소득·임차가구 밀집 분석 요약",
        "",
        "## 분석 설정",
        f"- 입력 파일: `{INPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- 분석 대상: 총인구수 0 포함 법정동 {len(df):,}개",
        "- CI 기본 가중치: 청년 0.35, 1인가구 0.30, 기초수급 0.35",
        f"- 경계 데이터: `southkorea/seoul-maps` JUSO 2015 GeoJSON (`{GEOJSON_PATH.relative_to(PROJECT_ROOT)}`)",
        "",
        "## 주요 결과 TOP 10",
        dataframe_to_markdown(top10),
        "",
        "## K-means 군집 요약",
        dataframe_to_markdown(cluster_table),
        "",
        "## 공간 자기상관 요약",
        f"- Moran's I: {moran_summary['global_moran_i']:.4f}",
        f"- Queen 이웃 링크 수: {moran_summary['neighbor_links']:,}",
        f"- 이웃이 없는 법정동 수: {moran_summary['isolated_count']:,}",
        f"- LISA permutation 수: {moran_summary['permutations']:,}",
        "",
        dataframe_to_markdown(lisa_counts),
        "",
        "## 산출물",
        f"- `{CONCENTRATION_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{CI_RANK_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{CLUSTER_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{TOP30_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{FIGURE_DIR.relative_to(PROJECT_ROOT)}`",
        "",
    ]

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[저장] {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


def validate_outputs(df: pd.DataFrame, top30: pd.DataFrame, geojson: dict) -> None:
    if len(df) != 467:
        raise AssertionError(f"분석 대상 행 수 오류: {len(df):,}")
    if len(geojson["features"]) != 467:
        raise AssertionError(f"GeoJSON feature 수 오류: {len(geojson['features']):,}")
    for col in NORM_COLS + ["concentration_index"]:
        values = df[col].astype(float)
        if ((values < -1e-12) | (values > 1 + 1e-12)).any():
            raise AssertionError(f"{col} 값이 [0, 1] 범위를 벗어났습니다.")
    if len(top30) != 30:
        raise AssertionError(f"TOP30 행 수 오류: {len(top30):,}")
    if not top30["concentration_index"].is_monotonic_decreasing:
        raise AssertionError("TOP30이 CI 내림차순으로 정렬되어 있지 않습니다.")
    if df["cluster_label"].nunique() != 4:
        raise AssertionError(f"K-means 라벨 수 오류: {df['cluster_label'].nunique():,}")

    expected_figures = [
        "choropleth_youth_ratio.png",
        "choropleth_single_household_ratio.png",
        "choropleth_basic_livelihood_ratio.png",
        "choropleth_concentration_index.png",
        "lisa_hotspots.png",
        "kmeans_clusters.png",
        "top20_youth_ratio_bar.png",
        "top20_single_household_ratio_bar.png",
        "top20_basic_livelihood_ratio_bar.png",
    ]
    missing_figures = [name for name in expected_figures if not (FIGURE_DIR / name).exists()]
    if missing_figures:
        raise AssertionError(f"생성되지 않은 그림 파일: {missing_figures}")

    print("[검증] 행 수, 매칭, 정규화 범위, TOP30, 군집, 그림 산출물 확인 완료")
