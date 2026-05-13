from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch, Polygon as MplPolygon

from .config import FIGURE_DIR, PROJECT_ROOT
from .spatial import geometry_rings


BLUE_5_CMAP = ListedColormap(
    ["#E8EFFB", "#B4CEF6", "#4B84EC", "#0D5ACB", "#03142E"],
    name="presentation_blue_5",
)


def make_polygon_patches(feature: dict) -> list[MplPolygon]:
    patches = []
    for ring in geometry_rings(feature["geometry"]):
        if len(ring) >= 3:
            patches.append(MplPolygon(np.asarray(ring, dtype=float), closed=True))
    return patches


def set_map_extent(ax, geojson: dict) -> None:
    xs = []
    ys = []
    for feature in geojson["features"]:
        for ring in geometry_rings(feature["geometry"]):
            arr = np.asarray(ring, dtype=float)
            xs.extend(arr[:, 0].tolist())
            ys.extend(arr[:, 1].tolist())
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.set_aspect("equal")
    ax.axis("off")


def plot_choropleth(
    df: pd.DataFrame,
    geojson: dict,
    value_col: str,
    title: str,
    output_path: Path,
    cmap: str | ListedColormap = "viridis",
) -> None:
    value_by_code = df.set_index("법정동코드8")[value_col].astype(float).to_dict()
    patches = []
    values = []

    for feature in geojson["features"]:
        code = str(feature["properties"]["EMD_CD"])
        for patch in make_polygon_patches(feature):
            patches.append(patch)
            values.append(value_by_code[code])

    fig, ax = plt.subplots(figsize=(9, 9))
    collection = PatchCollection(
        patches,
        cmap=cmap,
        norm=Normalize(vmin=min(values), vmax=max(values)),
        linewidth=0.25,
        edgecolor="#ffffff",
    )
    collection.set_array(np.asarray(values))
    ax.add_collection(collection)
    set_map_extent(ax, geojson)
    ax.set_title(title, fontsize=16, pad=14)
    cbar = fig.colorbar(collection, ax=ax, fraction=0.035, pad=0.01)
    cbar.ax.tick_params(labelsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {output_path.relative_to(PROJECT_ROOT)}")


def plot_categorical_map(
    df: pd.DataFrame,
    geojson: dict,
    category_col: str,
    title: str,
    output_path: Path,
    color_map: dict[str, str],
) -> None:
    category_by_code = df.set_index("법정동코드8")[category_col].astype(str).to_dict()

    fig, ax = plt.subplots(figsize=(9, 9))
    for category, color in color_map.items():
        patches = []
        for feature in geojson["features"]:
            code = str(feature["properties"]["EMD_CD"])
            if category_by_code.get(code) != category:
                continue
            patches.extend(make_polygon_patches(feature))
        if patches:
            collection = PatchCollection(
                patches,
                facecolor=color,
                linewidth=0.25,
                edgecolor="#ffffff",
            )
            ax.add_collection(collection)

    set_map_extent(ax, geojson)
    ax.set_title(title, fontsize=16, pad=14)
    handles = [
        Patch(facecolor=color, edgecolor="#555555", label=category)
        for category, color in color_map.items()
        if category in set(category_by_code.values())
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {output_path.relative_to(PROJECT_ROOT)}")


def create_all_maps(df: pd.DataFrame, geojson: dict) -> None:
    plot_choropleth(
        df,
        geojson,
        "청년비율",
        "서울 법정동별 청년인구 비율",
        FIGURE_DIR / "choropleth_youth_ratio.png",
        cmap="YlGnBu",
    )
    plot_choropleth(
        df,
        geojson,
        "1인가구비율",
        "서울 법정동별 1인가구 비율",
        FIGURE_DIR / "choropleth_single_household_ratio.png",
        cmap="PuBu",
    )
    plot_choropleth(
        df,
        geojson,
        "기초수급가구비율",
        "서울 법정동별 기초수급가구 비율",
        FIGURE_DIR / "choropleth_basic_livelihood_ratio.png",
        cmap="OrRd",
    )
    plot_choropleth(
        df,
        geojson,
        "concentration_index",
        "서울 청년·저소득·임차가구 밀집지수(CI)",
        FIGURE_DIR / "choropleth_concentration_index.png",
        cmap=BLUE_5_CMAP,
    )

    lisa_colors = {
        "HH": "#d73027",
        "HL": "#fc8d59",
        "LH": "#91bfdb",
        "LL": "#4575b4",
        "Not significant": "#d9d9d9",
    }
    plot_categorical_map(
        df,
        geojson,
        "lisa_cluster",
        "LISA Hot Spot 분석",
        FIGURE_DIR / "lisa_hotspots.png",
        lisa_colors,
    )

    cluster_colors = {
        "복합 밀집형": "#d73027",
        "청년 밀집형": "#1a9850",
        "저소득 밀집형": "#984ea3",
        "일반형": "#bdbdbd",
    }
    plot_categorical_map(
        df,
        geojson,
        "cluster_label",
        "K-means 법정동 유형 분류",
        FIGURE_DIR / "kmeans_clusters.png",
        cluster_colors,
    )


def plot_top20_ratio_bar(
    df: pd.DataFrame,
    ratio_col: str,
    title: str,
    output_path: Path,
    color: str,
) -> None:
    top20 = (
        df.sort_values([ratio_col, "시군구명", "법정동명"], ascending=[False, True, True])
        .head(20)
        .copy()
    )
    top20["label"] = top20["시군구명"] + " " + top20["법정동명"]
    top20 = top20.sort_values(ratio_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    values = top20[ratio_col].astype(float) * 100
    ax.barh(top20["label"], values, color=color)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("비율(%)")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_value = float(values.max()) if len(values) else 0
    for idx, value in enumerate(values):
        ax.text(value + max_value * 0.01, idx, f"{value:.1f}%", va="center", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {output_path.relative_to(PROJECT_ROOT)}")


def create_top20_ratio_bar_charts(df: pd.DataFrame) -> None:
    plot_top20_ratio_bar(
        df,
        "청년비율",
        "청년인구 비율 상위 20개 법정동",
        FIGURE_DIR / "top20_youth_ratio_bar.png",
        "#2c7fb8",
    )
    plot_top20_ratio_bar(
        df,
        "1인가구비율",
        "1인가구 비율 상위 20개 법정동",
        FIGURE_DIR / "top20_single_household_ratio_bar.png",
        "#41ab5d",
    )
    plot_top20_ratio_bar(
        df,
        "기초수급가구비율",
        "기초수급가구 비율 상위 20개 법정동",
        FIGURE_DIR / "top20_basic_livelihood_ratio_bar.png",
        "#d95f0e",
    )
