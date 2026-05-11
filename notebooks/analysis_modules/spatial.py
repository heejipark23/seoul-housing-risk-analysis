from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def geometry_rings(geometry: dict) -> list[list[list[float]]]:
    geom_type = geometry["type"]
    coordinates = geometry["coordinates"]
    if geom_type == "Polygon":
        return coordinates
    if geom_type == "MultiPolygon":
        rings = []
        for polygon in coordinates:
            rings.extend(polygon)
        return rings
    raise ValueError(f"지원하지 않는 geometry type: {geom_type}")


def build_queen_neighbors(geojson: dict, precision: int = 8) -> dict[str, set[str]]:
    neighbors = {
        str(feature["properties"]["EMD_CD"]): set()
        for feature in geojson["features"]
    }
    vertex_to_codes: dict[tuple[float, float], set[str]] = defaultdict(set)

    for feature in geojson["features"]:
        code = str(feature["properties"]["EMD_CD"])
        for ring in geometry_rings(feature["geometry"]):
            for lon, lat in ring:
                vertex_to_codes[(round(float(lon), precision), round(float(lat), precision))].add(code)

    for codes in vertex_to_codes.values():
        if len(codes) < 2:
            continue
        for code in codes:
            neighbors[code].update(codes - {code})

    return neighbors


def calculate_moran_lisa(
    df: pd.DataFrame,
    neighbors: dict[str, set[str]],
    value_col: str = "concentration_index",
    permutations: int = 999,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = df.copy().sort_values("법정동코드8").reset_index(drop=True)
    codes = result["법정동코드8"].tolist()
    code_to_idx = {code: idx for idx, code in enumerate(codes)}
    neighbor_indices = [
        [code_to_idx[neighbor] for neighbor in sorted(neighbors.get(code, set())) if neighbor in code_to_idx]
        for code in codes
    ]

    values = result[value_col].astype(float).to_numpy()
    z = values - values.mean()
    denominator = float(np.sum(z**2))
    s0 = float(sum(len(items) for items in neighbor_indices))

    if s0 == 0 or np.isclose(denominator, 0):
        raise ValueError("Moran's I 계산 실패: 공간 이웃 또는 분산이 없습니다.")

    numerator = 0.0
    spatial_lag = np.zeros(len(values), dtype=float)
    for i, items in enumerate(neighbor_indices):
        if items:
            neighbor_z = z[items]
            numerator += float(np.sum(z[i] * neighbor_z))
            spatial_lag[i] = float(np.mean(neighbor_z))

    global_moran_i = (len(values) / s0) * (numerator / denominator)
    local_i = z * spatial_lag

    rng = np.random.default_rng(seed)
    permuted_local_i = np.zeros((permutations, len(values)), dtype=float)
    for permutation_idx in range(permutations):
        permuted_z = rng.permutation(z)
        for i, items in enumerate(neighbor_indices):
            if items:
                permuted_local_i[permutation_idx, i] = z[i] * float(np.mean(permuted_z[items]))

    p_values = (
        (np.abs(permuted_local_i) >= np.abs(local_i)).sum(axis=0) + 1
    ) / (permutations + 1)

    lisa_labels = []
    for zi, lag_i, p_value, items in zip(z, spatial_lag, p_values, neighbor_indices):
        if not items or p_value > 0.05:
            lisa_labels.append("Not significant")
        elif zi > 0 and lag_i > 0:
            lisa_labels.append("HH")
        elif zi < 0 and lag_i < 0:
            lisa_labels.append("LL")
        elif zi > 0 and lag_i < 0:
            lisa_labels.append("HL")
        elif zi < 0 and lag_i > 0:
            lisa_labels.append("LH")
        else:
            lisa_labels.append("Not significant")

    result["neighbor_count"] = [len(items) for items in neighbor_indices]
    result["spatial_lag_ci_centered"] = spatial_lag
    result["local_moran_i"] = local_i
    result["lisa_p_value"] = p_values
    result["lisa_cluster"] = lisa_labels

    moran_summary = {
        "global_moran_i": float(global_moran_i),
        "neighbor_links": int(s0),
        "isolated_count": int(sum(1 for items in neighbor_indices if not items)),
        "permutations": permutations,
    }

    return result, moran_summary
