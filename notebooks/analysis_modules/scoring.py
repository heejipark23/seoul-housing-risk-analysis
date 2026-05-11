from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_WEIGHTS, NORM_COLS, RATIO_COLS


def minmax_normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or pd.isna(max_value):
        raise ValueError(f"{series.name} 정규화 실패: 숫자형 값이 없습니다.")
    if np.isclose(max_value, min_value):
        return pd.Series(0.0, index=series.index)
    return (values - min_value) / (max_value - min_value)


def weighted_sum(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"가중치 합이 1이 아닙니다: {total_weight}")
    score = pd.Series(0.0, index=df.index)
    for col, weight in weights.items():
        score = score + df[col].astype(float) * weight
    return score


def calculate_concentration_index(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for ratio_col, norm_col in zip(RATIO_COLS, NORM_COLS):
        result[norm_col] = minmax_normalize(result[ratio_col])

    result["concentration_index"] = weighted_sum(result, DEFAULT_WEIGHTS)
    result["concentration_rank"] = (
        result["concentration_index"].rank(ascending=False, method="min").astype("int64")
    )
    return result.sort_values(["concentration_rank", "시군구명", "법정동명"]).reset_index(drop=True)


def build_ci_rank_output(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["시군구명", "법정동명", "법정동코드", "concentration_index"]]
        .rename(columns={"concentration_index": "CI"})
        .sort_values("CI", ascending=False)
        .reset_index(drop=True)
    )


def assign_cluster_labels(clustered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    centers = (
        clustered.groupby("cluster_id", as_index=False)[RATIO_COLS + NORM_COLS + ["concentration_index"]]
        .mean()
        .sort_values("cluster_id")
    )

    labels: dict[int, str] = {}
    remaining = set(centers["cluster_id"])

    youth_cluster = int(centers.loc[centers["청년비율"].idxmax(), "cluster_id"])
    labels[youth_cluster] = "청년 밀집형"
    remaining.discard(youth_cluster)

    low_income_candidates = centers.loc[centers["cluster_id"].isin(remaining)]
    low_income_cluster = int(low_income_candidates.loc[low_income_candidates["기초수급가구비율"].idxmax(), "cluster_id"])
    labels[low_income_cluster] = "저소득 밀집형"
    remaining.discard(low_income_cluster)

    general_candidates = centers.loc[centers["cluster_id"].isin(remaining)]
    general_cluster = int(general_candidates.loc[general_candidates["concentration_index"].idxmin(), "cluster_id"])
    labels[general_cluster] = "일반형"
    remaining.discard(general_cluster)

    for cluster_id in remaining:
        labels[int(cluster_id)] = "복합 밀집형"

    clustered = clustered.copy()
    clustered["cluster_label"] = clustered["cluster_id"].map(labels)

    counts = clustered["cluster_id"].value_counts().rename_axis("cluster_id").reset_index(name="법정동수")
    centers = centers.merge(counts, on="cluster_id", how="left")
    centers["cluster_label"] = centers["cluster_id"].map(labels)
    centers = centers[
        ["cluster_id", "cluster_label", "법정동수", *RATIO_COLS, "concentration_index"]
    ].sort_values("cluster_label")

    return clustered, centers


def run_kmeans_clustering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()
    features = result[RATIO_COLS].astype(float)
    scaled_features = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=4, random_state=42, n_init=20)
    result["cluster_id"] = model.fit_predict(scaled_features).astype("int64")

    return assign_cluster_labels(result)


def build_top30(df: pd.DataFrame) -> pd.DataFrame:
    top30 = df.sort_values("concentration_rank").head(30).copy()
    top30["is_lisa_hotspot_hh"] = top30["lisa_cluster"] == "HH"
    top30["priority_rank"] = np.arange(1, len(top30) + 1)
    return top30[
        [
            "priority_rank",
            "시군구명",
            "법정동명",
            "법정동코드",
            "concentration_index",
            "concentration_rank",
            "청년비율",
            "1인가구비율",
            "기초수급가구비율",
            "cluster_label",
            "lisa_cluster",
            "lisa_p_value",
            "is_lisa_hotspot_hh",
        ]
    ]
