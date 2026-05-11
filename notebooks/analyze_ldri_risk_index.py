from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


MAP_WIDTH = 900
MAP_PADDING = 24
GRADE_LABELS = ["매우낮음", "낮음", "보통", "높음", "매우높음"]


def find_project_root() -> Path:
    starts = []
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    starts.append(Path.cwd().resolve())

    seen: set[Path] = set()
    for start in starts:
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "data").exists() and (candidate / "output").exists():
                return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()
OUTPUT_DIR = PROJECT_ROOT / "output"
LDRI_DIR = OUTPUT_DIR / "ldri"
MAP_DIR = LDRI_DIR / "maps"

CI_PATH = OUTPUT_DIR / "seoul_legal_dong_ci_rank.csv"
HOUSING_COST_PATH = OUTPUT_DIR / "housing_Cost.csv"
GEOJSON_PATH = PROJECT_ROOT / "data" / "raw" / "seoul_neighborhoods_geo_simple_2015.geojson"

CI_REQUIRED_COLS = ["시군구명", "법정동명", "법정동코드", "CI"]
HOUSING_REQUIRED_COLS = ["자치구명", "법정동코드", "법정동명", "주거비_압박지수"]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    for encoding in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                return rows, list(reader.fieldnames or [])
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"CSV 인코딩을 확인할 수 없습니다: {path}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[저장] {path.relative_to(PROJECT_ROOT)} rows={len(rows):,}")


def require_columns(fieldnames: list[str], required: list[str], label: str) -> None:
    missing = [col for col in required if col not in fieldnames]
    if missing:
        raise ValueError(f"{label} 필수 컬럼 누락: {missing}")


def normalize_code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def to_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(str(value).strip().replace(",", ""))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def out_num(value: Any, digits: int = 6) -> str:
    number = to_float(value)
    if number is None:
        return ""
    if abs(number) < 10 ** (-(digits + 1)):
        number = 0.0
    text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def minmax_map(rows: list[dict[str, Any]], value_key: str, out_key: str) -> tuple[float, float]:
    values = [to_float(row.get(value_key)) for row in rows]
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        raise ValueError(f"정규화할 값이 없습니다: {value_key}")

    min_value = min(clean_values)
    max_value = max(clean_values)
    denominator = max_value - min_value
    for row in rows:
        value = to_float(row.get(value_key))
        if value is None:
            row[out_key] = None
        elif math.isclose(denominator, 0.0):
            row[out_key] = 0.0
        else:
            row[out_key] = (value - min_value) / denominator
    return min_value, max_value


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("중앙값을 계산할 값이 없습니다.")
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def check_unique_codes(rows: list[dict[str, str]], code_col: str, label: str) -> None:
    codes = [normalize_code(row.get(code_col, "")) for row in rows]
    duplicates = sorted(code for code, count in Counter(codes).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} 법정동코드 중복: {duplicates[:10]}")


def load_inputs() -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    ci_rows, ci_fields = read_csv(CI_PATH)
    housing_rows, housing_fields = read_csv(HOUSING_COST_PATH)
    require_columns(ci_fields, CI_REQUIRED_COLS, "seoul_legal_dong_ci_rank.csv")
    require_columns(housing_fields, HOUSING_REQUIRED_COLS, "housing_Cost.csv")
    check_unique_codes(ci_rows, "법정동코드", "seoul_legal_dong_ci_rank.csv")
    check_unique_codes(housing_rows, "법정동코드", "housing_Cost.csv")

    housing_by_code: dict[str, dict[str, Any]] = {}
    for row in housing_rows:
        code = normalize_code(row["법정동코드"])
        pressure = to_float(row["주거비_압박지수"])
        if pressure is None:
            raise ValueError(f"주거비_압박지수 숫자 변환 실패: {code}")
        housing_by_code[code] = {
            "자치구명": row["자치구명"].strip(),
            "법정동코드": code,
            "법정동코드8": code[:8],
            "법정동명": row["법정동명"].strip(),
            "주거비_압박지수": pressure,
        }

    return ci_rows, housing_by_code


def build_master(ci_rows: list[dict[str, str]], housing_by_code: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    master: list[dict[str, Any]] = []
    for row in ci_rows:
        code = normalize_code(row["법정동코드"])
        ci_value = to_float(row["CI"])
        if ci_value is None:
            raise ValueError(f"CI 숫자 변환 실패: {code}")

        housing = housing_by_code.get(code)
        result: dict[str, Any] = {
            "법정동코드": code,
            "법정동코드8": code[:8],
            "시군구명": row["시군구명"].strip(),
            "법정동명": row["법정동명"].strip(),
            "CI": ci_value,
        }

        if housing is None:
            result.update(
                {
                    "주거비자료_매칭여부": "미매칭",
                    "관측품질": "주거비자료없음",
                    "주거비_압박지수": None,
                    "D1_score": None,
                    "LDRI_raw": None,
                    "LDRI": None,
                    "LDRI_rank": None,
                    "LDRI_등급": "",
                    "위험유형": "",
                    "차원비교유형": "",
                }
            )
        else:
            result.update(
                {
                    "주거비자료_매칭여부": "매칭",
                    "관측품질": "주거비관측",
                    "주거비_압박지수": housing["주거비_압박지수"],
                    "D1_score": housing["주거비_압박지수"],
                    "LDRI_raw": ci_value * housing["주거비_압박지수"],
                    "LDRI": None,
                    "LDRI_rank": None,
                    "LDRI_등급": "",
                    "위험유형": "",
                    "차원비교유형": "",
                }
            )
        master.append(result)

    scored = [row for row in master if row["주거비자료_매칭여부"] == "매칭"]
    minmax_map(scored, "LDRI_raw", "LDRI")
    for row in scored:
        row["LDRI"] = row["LDRI"] * 100

    assign_ranks_and_classes(scored)
    return master


def assign_ranks_and_classes(scored_rows: list[dict[str, Any]]) -> None:
    ranked_desc = sorted(
        scored_rows,
        key=lambda row: (-row["LDRI"], row["시군구명"], row["법정동명"], row["법정동코드"]),
    )
    for idx, row in enumerate(ranked_desc, start=1):
        row["LDRI_rank"] = idx

    ranked_asc = sorted(
        scored_rows,
        key=lambda row: (row["LDRI"], row["시군구명"], row["법정동명"], row["법정동코드"]),
    )
    total = len(ranked_asc)
    for idx, row in enumerate(ranked_asc):
        grade_index = min(len(GRADE_LABELS) - 1, int(idx * len(GRADE_LABELS) / total))
        row["LDRI_등급"] = GRADE_LABELS[grade_index]

    ci_threshold = median([row["CI"] for row in scored_rows])
    d1_threshold = median([row["D1_score"] for row in scored_rows])
    for row in scored_rows:
        ci_high = row["CI"] >= ci_threshold
        d1_high = row["D1_score"] >= d1_threshold
        if ci_high and d1_high:
            row["위험유형"] = "강제잔류·압박형"
            row["차원비교유형"] = "CI 높음 / 주거비 높음"
        elif ci_high and not d1_high:
            row["위험유형"] = "잠재위험형"
            row["차원비교유형"] = "CI 높음 / 주거비 낮음"
        elif not ci_high and d1_high:
            row["위험유형"] = "이탈진행형"
            row["차원비교유형"] = "CI 낮음 / 주거비 높음"
        else:
            row["위험유형"] = "안정형"
            row["차원비교유형"] = "CI 낮음 / 주거비 낮음"


def output_row(row: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    numeric_cols = {"CI", "주거비_압박지수", "D1_score", "LDRI_raw", "LDRI", "gi_star_z", "gi_star_p"}
    integer_cols = {"LDRI_rank", "neighbor_count"}
    result: dict[str, Any] = {}
    for col in fieldnames:
        value = row.get(col, "")
        if value is None:
            result[col] = ""
        elif col in numeric_cols:
            result[col] = out_num(value, 6)
        elif col in integer_cols:
            number = to_float(value)
            result[col] = "" if number is None else str(int(round(number)))
        else:
            result[col] = value
    return result


def geometry_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    geom_type = geometry["type"]
    coordinates = geometry["coordinates"]
    if geom_type == "Polygon":
        return coordinates
    if geom_type == "MultiPolygon":
        rings: list[list[list[float]]] = []
        for polygon in coordinates:
            rings.extend(polygon)
        return rings
    raise ValueError(f"지원하지 않는 geometry type: {geom_type}")


def load_geojson() -> dict[str, Any]:
    with GEOJSON_PATH.open("r", encoding="utf-8") as file:
        geojson = json.load(file)
    if len(geojson.get("features", [])) != 467:
        raise ValueError(f"GeoJSON feature 수가 467개가 아닙니다: {len(geojson.get('features', []))}")
    return geojson


def validate_geo_match(master: list[dict[str, Any]], scored: list[dict[str, Any]], geojson: dict[str, Any]) -> None:
    geo_codes = {str(feature["properties"]["EMD_CD"]) for feature in geojson["features"]}
    master_codes = {row["법정동코드8"] for row in master}
    scored_codes = {row["법정동코드8"] for row in scored}
    missing_master = sorted(master_codes - geo_codes)
    missing_scored = sorted(scored_codes - geo_codes)
    if missing_master or missing_scored:
        raise ValueError(
            f"GeoJSON 매칭 실패: master_missing={missing_master[:10]}, scored_missing={missing_scored[:10]}"
        )


def build_queen_neighbors(geojson: dict[str, Any], precision: int = 8) -> dict[str, set[str]]:
    neighbors = {str(feature["properties"]["EMD_CD"]): set() for feature in geojson["features"]}
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


def calculate_gi_star(scored_rows: list[dict[str, Any]], neighbors: dict[str, set[str]]) -> list[dict[str, Any]]:
    by_code8 = {row["법정동코드8"]: row for row in scored_rows}
    values = [row["LDRI"] for row in scored_rows]
    n = len(values)
    mean_value = sum(values) / n
    variance = sum(value * value for value in values) / n - mean_value * mean_value
    std_value = math.sqrt(max(variance, 0.0))
    if math.isclose(std_value, 0.0):
        raise ValueError("Gi* 계산 실패: LDRI 분산이 없습니다.")

    hotspot_rows: list[dict[str, Any]] = []
    for row in sorted(scored_rows, key=lambda item: item["법정동코드8"]):
        code8 = row["법정동코드8"]
        eligible_neighbors = sorted(neighbor for neighbor in neighbors.get(code8, set()) if neighbor in by_code8)
        weight_codes = [code8, *eligible_neighbors]
        sum_w = float(len(weight_codes))
        sum_w2 = sum_w
        weighted_sum = sum(by_code8[code]["LDRI"] for code in weight_codes)
        denominator_term = (n * sum_w2 - sum_w * sum_w) / (n - 1)
        denominator = std_value * math.sqrt(max(denominator_term, 0.0))
        z_score = 0.0 if math.isclose(denominator, 0.0) else (weighted_sum - mean_value * sum_w) / denominator
        p_value = math.erfc(abs(z_score) / math.sqrt(2))

        hotspot_rows.append(
            {
                "법정동코드": row["법정동코드"],
                "법정동코드8": code8,
                "시군구명": row["시군구명"],
                "법정동명": row["법정동명"],
                "관측품질": row["관측품질"],
                "LDRI": row["LDRI"],
                "neighbor_count": len(eligible_neighbors),
                "gi_star_z": z_score,
                "gi_star_p": p_value,
                "hotspot_class": classify_hotspot(z_score),
            }
        )
    return hotspot_rows


def classify_hotspot(z_score: float) -> str:
    if z_score >= 2.576:
        return "Hot Spot 99%"
    if z_score >= 1.96:
        return "Hot Spot 95%"
    if z_score >= 1.645:
        return "Hot Spot 90%"
    if z_score <= -2.576:
        return "Cold Spot 99%"
    if z_score <= -1.96:
        return "Cold Spot 95%"
    if z_score <= -1.645:
        return "Cold Spot 90%"
    return "Not significant"


def geo_bounds(geojson: dict[str, Any]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for feature in geojson["features"]:
        for ring in geometry_rings(feature["geometry"]):
            for lon, lat in ring:
                xs.append(float(lon))
                ys.append(float(lat))
    return min(xs), min(ys), max(xs), max(ys)


def feature_to_svg_path(feature: dict[str, Any], min_x: float, max_y: float, scale: float, padding: int) -> str:
    path_parts: list[str] = []
    for ring in geometry_rings(feature["geometry"]):
        if len(ring) < 3:
            continue
        commands = []
        for idx, (lon, lat) in enumerate(ring):
            x = (float(lon) - min_x) * scale + padding
            y = (max_y - float(lat)) * scale + padding
            commands.append(f"{'M' if idx == 0 else 'L'}{x:.2f},{y:.2f}")
        commands.append("Z")
        path_parts.append(" ".join(commands))
    return " ".join(path_parts)


def make_svg_map(
    geojson: dict[str, Any],
    rows_by_code8: dict[str, dict[str, Any]],
    title: str,
    subtitle: str,
    output_path: Path,
    color_for_row: Callable[[dict[str, Any] | None], str],
    legend_items: list[tuple[str, str]],
    value_for_title: Callable[[str, dict[str, Any] | None], str],
) -> None:
    min_x, min_y, max_x, max_y = geo_bounds(geojson)
    usable_width = MAP_WIDTH - MAP_PADDING * 2
    scale = usable_width / (max_x - min_x)
    height = int((max_y - min_y) * scale + MAP_PADDING * 2)

    paths: list[str] = []
    for feature in geojson["features"]:
        code8 = str(feature["properties"]["EMD_CD"])
        row = rows_by_code8.get(code8)
        color = color_for_row(row)
        label = value_for_title(code8, row)
        path = feature_to_svg_path(feature, min_x, max_y, scale, MAP_PADDING)
        paths.append(
            f'<path d="{path}" fill="{color}" stroke="#ffffff" stroke-width="0.55" '
            f'fill-rule="evenodd"><title>{html.escape(label)}</title></path>'
        )

    legend_html = "\n".join(
        f'<span class="legend-item"><span class="swatch" style="background:{color}"></span>{html.escape(label)}</span>'
        for label, color in legend_items
    )
    svg = "\n      ".join(paths)
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif;
      color: #111827;
      background: #f8fafc;
    }}
    main {{
      max-width: {MAP_WIDTH + 80}px;
      margin: 0 auto;
      padding: 28px 24px 36px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0 0 18px;
      color: #475569;
      line-height: 1.5;
    }}
    svg {{
      width: 100%;
      height: auto;
      background: #ffffff;
      border: 1px solid #e5e7eb;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin: 16px 0 0;
      font-size: 13px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border: 1px solid #64748b;
      display: inline-block;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    <p>{html.escape(subtitle)}</p>
    <svg viewBox="0 0 {MAP_WIDTH} {height}" role="img" aria-label="{html.escape(title)}">
      {svg}
    </svg>
    <div class="legend">{legend_html}</div>
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"[저장] {output_path.relative_to(PROJECT_ROOT)}")


def create_maps(master: list[dict[str, Any]], hotspot_rows: list[dict[str, Any]], geojson: dict[str, Any]) -> None:
    rows_by_code8 = {row["법정동코드8"]: row for row in master}
    hotspot_by_code8 = {row["법정동코드8"]: row for row in hotspot_rows}
    top30_codes = {
        row["법정동코드8"]
        for row in sorted(
            [row for row in master if row.get("LDRI") is not None],
            key=lambda item: item["LDRI_rank"],
        )[:30]
    }

    grade_colors = {
        "매우낮음": "#dbeafe",
        "낮음": "#93c5fd",
        "보통": "#fef3c7",
        "높음": "#fb923c",
        "매우높음": "#dc2626",
        "주거비자료없음": "#d1d5db",
    }
    quadrant_colors = {
        "강제잔류·압박형": "#dc2626",
        "잠재위험형": "#f97316",
        "이탈진행형": "#eab308",
        "안정형": "#16a34a",
        "주거비자료없음": "#d1d5db",
    }
    hotspot_colors = {
        "Hot Spot 99%": "#7f1d1d",
        "Hot Spot 95%": "#dc2626",
        "Hot Spot 90%": "#fca5a5",
        "Not significant": "#e5e7eb",
        "Cold Spot 90%": "#bfdbfe",
        "Cold Spot 95%": "#2563eb",
        "Cold Spot 99%": "#1e3a8a",
        "주거비자료없음": "#d1d5db",
    }
    top30_colors = {"Top 30": "#dc2626", "산출가능": "#fee2e2", "주거비자료없음": "#d1d5db"}

    def standard_title(code8: str, row: dict[str, Any] | None) -> str:
        if not row:
            return f"{code8}: 데이터 없음"
        ldri = out_num(row.get("LDRI"), 2)
        ldri_text = ldri if ldri else "산출 제외"
        return (
            f"{row['시군구명']} {row['법정동명']} | LDRI {ldri_text} | "
            f"CI {out_num(row.get('CI'), 4)} | 주거비 {out_num(row.get('주거비_압박지수'), 4)}"
        )

    make_svg_map(
        geojson,
        rows_by_code8,
        "LDRI 5등급 지도",
        "LDRI = CI × 주거비_압박지수. 최종 LDRI만 0~100 표시용으로 환산했습니다.",
        MAP_DIR / "ldri_grade_map.html",
        lambda row: grade_colors.get(row.get("LDRI_등급") if row else "주거비자료없음", "#d1d5db"),
        [(label, grade_colors[label]) for label in [*GRADE_LABELS, "주거비자료없음"]],
        standard_title,
    )
    make_svg_map(
        geojson,
        rows_by_code8,
        "4사분면 위험유형 지도",
        "CI와 주거비 압박지수의 중앙값을 기준으로 위험유형을 분류했습니다.",
        MAP_DIR / "quadrant_map.html",
        lambda row: quadrant_colors.get(row.get("위험유형") if row else "주거비자료없음", "#d1d5db"),
        [(label, color) for label, color in quadrant_colors.items()],
        standard_title,
    )
    make_svg_map(
        geojson,
        hotspot_by_code8,
        "Getis-Ord Gi* 핫스팟 지도",
        "LDRI 산출 가능 331개 법정동만 대상으로 Queen 인접성과 자기 자신을 포함해 Gi* z-score를 계산했습니다.",
        MAP_DIR / "hotspot_map.html",
        lambda row: hotspot_colors.get(row.get("hotspot_class") if row else "주거비자료없음", "#d1d5db"),
        [(label, color) for label, color in hotspot_colors.items()],
        lambda code8, row: (
            f"{row['시군구명']} {row['법정동명']} | {row['hotspot_class']} | z={out_num(row['gi_star_z'], 3)}"
            if row
            else f"{code8}: 주거비자료없음"
        ),
    )
    make_svg_map(
        geojson,
        rows_by_code8,
        "CI / 주거비 차원 비교 지도",
        "두 기존 지표의 중앙값 기준 위치를 비교했습니다.",
        MAP_DIR / "dimension_compare_map.html",
        lambda row: quadrant_colors.get(row.get("위험유형") if row else "주거비자료없음", "#d1d5db"),
        [(label, color) for label, color in quadrant_colors.items()],
        standard_title,
    )
    make_svg_map(
        geojson,
        rows_by_code8,
        "LDRI Top 30 강조 지도",
        "LDRI 상위 30개 법정동을 진하게 표시했습니다.",
        MAP_DIR / "top30_map.html",
        lambda row: (
            top30_colors["Top 30"]
            if row and row["법정동코드8"] in top30_codes
            else top30_colors["산출가능"]
            if row and row.get("LDRI") is not None
            else top30_colors["주거비자료없음"]
        ),
        [(label, color) for label, color in top30_colors.items()],
        standard_title,
    )


def build_quality_report(
    ci_rows: list[dict[str, str]],
    housing_by_code: dict[str, dict[str, Any]],
    master: list[dict[str, Any]],
    hotspot_rows: list[dict[str, Any]],
    geojson: dict[str, Any],
) -> list[dict[str, Any]]:
    ci_codes = {normalize_code(row["법정동코드"]) for row in ci_rows}
    housing_codes = set(housing_by_code)
    missing_housing = [row for row in master if row["주거비자료_매칭여부"] == "미매칭"]
    matched = [row for row in master if row["주거비자료_매칭여부"] == "매칭"]
    missing_by_gu = Counter(row["시군구명"] for row in missing_housing)

    rows: list[dict[str, Any]] = []

    def add(item: str, value: Any, note: str = "") -> None:
        rows.append({"항목": item, "값": value, "비고": note})

    add("입력_CI_파일", CI_PATH.relative_to(PROJECT_ROOT))
    add("입력_주거비_파일", HOUSING_COST_PATH.relative_to(PROJECT_ROOT))
    add("LDRI_산식", "CI × 주거비_압박지수", "기존 두 지표를 재정규화하지 않음")
    add("LDRI_표시점수", "MinMax(LDRI_raw) × 100", "순위/등급 표시용")
    add("CI_행수", len(ci_rows))
    add("주거비_행수", len(housing_by_code))
    add("CI_주거비_매칭", len(ci_codes & housing_codes))
    add("CI_주거비_미매칭", len(ci_codes - housing_codes), "LDRI 산출 제외")
    add("주거비_CI_외부코드", len(housing_codes - ci_codes))
    add("LDRI_산출가능", len(matched))
    add("LDRI_산출제외", len(missing_housing))
    add("GeoJSON_feature_수", len(geojson.get("features", [])))
    add("핫스팟_산출행수", len(hotspot_rows))
    add("핫스팟_평균_인접법정동수", out_num(sum(row["neighbor_count"] for row in hotspot_rows) / len(hotspot_rows), 4))

    for gu, count in sorted(missing_by_gu.items()):
        add("주거비자료없음_자치구별", count, gu)
    return rows


def main() -> None:
    LDRI_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    ci_rows, housing_by_code = load_inputs()
    master = build_master(ci_rows, housing_by_code)
    scored = [row for row in master if row.get("LDRI") is not None]

    geojson = load_geojson()
    validate_geo_match(master, scored, geojson)
    neighbors = build_queen_neighbors(geojson)
    hotspot_rows = calculate_gi_star(scored, neighbors)

    master_fields = [
        "법정동코드",
        "법정동코드8",
        "시군구명",
        "법정동명",
        "CI",
        "주거비자료_매칭여부",
        "관측품질",
        "주거비_압박지수",
        "D1_score",
        "LDRI_raw",
        "LDRI",
        "LDRI_rank",
        "LDRI_등급",
        "위험유형",
        "차원비교유형",
    ]
    score_fields = [
        "LDRI_rank",
        "법정동코드",
        "시군구명",
        "법정동명",
        "관측품질",
        "CI",
        "주거비_압박지수",
        "D1_score",
        "LDRI_raw",
        "LDRI",
        "LDRI_등급",
        "위험유형",
    ]
    quadrant_fields = [
        "법정동코드",
        "시군구명",
        "법정동명",
        "관측품질",
        "CI",
        "주거비_압박지수",
        "LDRI",
        "위험유형",
        "차원비교유형",
    ]
    hotspot_fields = [
        "법정동코드",
        "시군구명",
        "법정동명",
        "관측품질",
        "LDRI",
        "neighbor_count",
        "gi_star_z",
        "gi_star_p",
        "hotspot_class",
    ]

    master_sorted = sorted(master, key=lambda row: (row["시군구명"], row["법정동명"], row["법정동코드"]))
    scored_sorted = sorted(master, key=lambda row: (row["LDRI_rank"] is None, row["LDRI_rank"] or 10**9))
    top30 = [row for row in scored_sorted if row.get("LDRI_rank") is not None][:30]
    quadrant_sorted = sorted(
        master,
        key=lambda row: (
            row["관측품질"] == "주거비자료없음",
            row["위험유형"],
            row["시군구명"],
            row["법정동명"],
        ),
    )

    write_csv(LDRI_DIR / "01_master_dataset.csv", [output_row(row, master_fields) for row in master_sorted], master_fields)
    write_csv(LDRI_DIR / "02_LDRI_score.csv", [output_row(row, score_fields) for row in scored_sorted], score_fields)
    write_csv(LDRI_DIR / "03_top30_priority.csv", [output_row(row, score_fields) for row in top30], score_fields)
    write_csv(
        LDRI_DIR / "04_quadrant_matrix.csv",
        [output_row(row, quadrant_fields) for row in quadrant_sorted],
        quadrant_fields,
    )
    write_csv(
        LDRI_DIR / "05_hotspot_result.csv",
        [output_row(row, hotspot_fields) for row in hotspot_rows],
        hotspot_fields,
    )

    quality_report = build_quality_report(ci_rows, housing_by_code, master, hotspot_rows, geojson)
    write_csv(LDRI_DIR / "00_quality_report.csv", quality_report, ["항목", "값", "비고"])
    create_maps(master, hotspot_rows, geojson)

    print("\n[LDRI 분석 완료]")
    print(f"- 산출 폴더: {LDRI_DIR.relative_to(PROJECT_ROOT)}")
    print("- 산식: LDRI_raw = CI × 주거비_압박지수")
    print("- 최종 LDRI: LDRI_raw를 0~100으로 환산한 표시점수")
    print(f"- LDRI 산출 가능: {len(scored):,}개")
    print(f"- 주거비자료없음: {len(master) - len(scored):,}개")


if __name__ == "__main__":
    main()
