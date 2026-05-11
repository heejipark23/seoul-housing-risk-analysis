from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")


def find_project_root() -> Path:
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
FIGURE_DIR = OUTPUT_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

INPUT_PATH = PROCESSED_DIR / "legal_dong_risk_base_2024.csv"
GEOJSON_PATH = RAW_DIR / "seoul_neighborhoods_geo_simple_2015.geojson"
GEOJSON_URL = (
    "https://raw.githubusercontent.com/southkorea/seoul-maps/refs/heads/master/"
    "juso/2015/json/seoul_neighborhoods_geo_simple.json"
)

CONCENTRATION_OUTPUT_PATH = OUTPUT_DIR / "seoul_legal_dong_concentration.csv"
CI_RANK_OUTPUT_PATH = OUTPUT_DIR / "seoul_legal_dong_ci_rank.csv"
CLUSTER_OUTPUT_PATH = OUTPUT_DIR / "seoul_legal_dong_clusters.csv"
TOP30_OUTPUT_PATH = OUTPUT_DIR / "seoul_priority_top30.csv"
SUMMARY_PATH = DOCS_DIR / "analysis_summary.md"

RATIO_COLS = ["청년비율", "1인가구비율", "기초수급가구비율"]
NORM_COLS = ["청년비율_norm", "1인가구비율_norm", "기초수급가구비율_norm"]
DEFAULT_WEIGHTS = {
    "청년비율_norm": 0.35,
    "1인가구비율_norm": 0.30,
    "기초수급가구비율_norm": 0.35,
}


def ensure_directories() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURE_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
