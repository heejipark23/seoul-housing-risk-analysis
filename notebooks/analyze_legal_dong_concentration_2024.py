# %% [markdown]
# # 서울시 청년·저소득·임차가구 밀집 법정동 분석
#
# 이 파일은 전체 분석 실행 순서만 담는다. 세부 로직은 `analysis_modules/`에
# 기능별로 분리해 두었으므로, 나중에 `.ipynb`로 옮길 때도 셀 단위로 가져오기 쉽다.

# %%
from __future__ import annotations

import pandas as pd

from analysis_modules.common import configure_matplotlib, save_csv
from analysis_modules.config import (
    CI_RANK_OUTPUT_PATH,
    CLUSTER_OUTPUT_PATH,
    CONCENTRATION_OUTPUT_PATH,
    OUTPUT_DIR,
    RATIO_COLS,
    TOP30_OUTPUT_PATH,
    ensure_directories,
)
from analysis_modules.data import load_base_dataset, load_or_download_geojson, validate_geojson_match
from analysis_modules.plots import create_all_maps, create_top20_ratio_bar_charts
from analysis_modules.reports import validate_outputs, write_analysis_summary
from analysis_modules.scoring import (
    build_ci_rank_output,
    build_top30,
    calculate_concentration_index,
    run_kmeans_clustering,
)
from analysis_modules.spatial import build_queen_neighbors, calculate_moran_lisa


# %% [markdown]
# ## 1. 분석 실행 함수

# %%
def main() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    ensure_directories()
    configure_matplotlib()

    base = load_base_dataset()
    geojson = load_or_download_geojson()
    validate_geojson_match(base, geojson)

    concentration = calculate_concentration_index(base)
    clustered, cluster_centers = run_kmeans_clustering(concentration)

    neighbors = build_queen_neighbors(geojson)
    analyzed, moran_summary = calculate_moran_lisa(clustered, neighbors)
    analyzed = analyzed.sort_values("concentration_rank").reset_index(drop=True)

    top30 = build_top30(analyzed)
    ci_rank = build_ci_rank_output(analyzed)

    concentration_cols = [
        "시도명",
        "시군구명",
        "법정동명",
        "법정동코드",
        "법정동코드8",
        "기준연월",
        "청년연령기준",
        "총인구수",
        "청년인구수",
        "청년비율",
        "청년비율_norm",
        "총가구수",
        "1인가구수",
        "1인가구비율",
        "1인가구비율_norm",
        "기초수급가구수",
        "기초수급가구비율",
        "기초수급가구비율_norm",
        "concentration_index",
        "concentration_rank",
        "cluster_id",
        "cluster_label",
        "neighbor_count",
        "spatial_lag_ci_centered",
        "local_moran_i",
        "lisa_p_value",
        "lisa_cluster",
        "연결된_행정동수",
    ]
    concentration_cols = [col for col in concentration_cols if col in analyzed.columns]

    save_csv(analyzed[concentration_cols], CONCENTRATION_OUTPUT_PATH)
    save_csv(ci_rank, CI_RANK_OUTPUT_PATH)
    save_csv(
        analyzed[
            [
                "시군구명",
                "법정동명",
                "법정동코드",
                "cluster_id",
                "cluster_label",
                *RATIO_COLS,
                "concentration_index",
                "concentration_rank",
            ]
        ].sort_values(["cluster_label", "concentration_rank"]),
        CLUSTER_OUTPUT_PATH,
    )
    save_csv(top30, TOP30_OUTPUT_PATH)

    create_all_maps(analyzed, geojson)
    create_top20_ratio_bar_charts(analyzed)
    write_analysis_summary(analyzed, top30, cluster_centers, moran_summary)
    validate_outputs(analyzed, top30, geojson)

    print("\n[분석 완료]")
    print(f"- CSV 산출 폴더: {OUTPUT_DIR}")
    print(f"- CI 순위 CSV: {CI_RANK_OUTPUT_PATH}")
    print(f"- TOP30: {TOP30_OUTPUT_PATH}")
    print(f"- Moran's I: {moran_summary['global_moran_i']:.4f}")
    print(f"- LISA HH hot spot 수: {(analyzed['lisa_cluster'] == 'HH').sum():,}")

    return analyzed, top30, moran_summary


# %%
if __name__ == "__main__":
    main()
