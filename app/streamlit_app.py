from __future__ import annotations

import base64
from copy import deepcopy
from html import escape
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    from data_loader import load_all_data
except ModuleNotFoundError:
    from app.data_loader import load_all_data


# ── 색상 상수 ──────────────────────────────────────────────
ACCENT = "#C2410C"
ACCENT_RGB = (194, 65, 12)
PALE_RGB = (250, 245, 241)
GRAY_RGB = (226, 232, 240)
DARK_RGB = (30, 41, 59)
MAP_BLUE_COLORS = ["#E8EFFB", "#B4CEF6", "#4B84EC", "#0D5ACB", "#03142E"]
MAP_MISSING_COLOR = "#CBD5E1"
MAP_STROKE_COLOR = "rgba(13,90,203,0.28)"
MAP_SELECTED_LABEL = "rgba(3,20,46,0.96)"
MAP_LABEL = "rgba(30,41,59,0.92)"
BACK_ICON_PATH = Path(__file__).resolve().parent / "assets" / "back.png"

RISK_COLORS = {
    "강제잔류·압박형": "#C2410C",
    "이탈진행형": "#B45309",
    "잠재위험형": "#0F766E",
    "안정형": "#64748B",
    "자료없음": "#CBD5E1",
}

SEOUL_MAP_METRICS = {
    "우선지원 위험도": ("서울_위험도", "점", "구 내 상위 3개 법정동 우선지원 점수 평균"),
    "Top30 법정동수": ("Top30_법정동수", "개", "서울 Top30 우선지원 법정동 포함 수"),
    "주거비 관측 커버리지": ("주거비자료_커버리지", "%", "구 내 법정동 중 주거비 자료가 관측된 비율"),
}

PLOTLY_CONFIG = {"displayModeBar": False}

GU_LABEL_OFFSETS = {
    "종로구": (-0.006, 0.002),
    "중구": (0.004, -0.002),
    "용산구": (-0.002, -0.006),
    "성동구": (0.006, -0.001),
    "동대문구": (0.005, 0.004),
    "서대문구": (-0.007, 0.000),
    "마포구": (-0.010, 0.001),
    "영등포구": (-0.004, -0.004),
    "동작구": (0.001, -0.006),
    "강남구": (0.010, -0.002),
    "서초구": (0.004, -0.008),
}


# ── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(
    page_title="서울시 청년·저소득 임차가구의 생활권 이탈 위험지역 대시보드",
    layout="wide",
)


# ── 데이터 로드 ────────────────────────────────────────────
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def cached_data() -> dict[str, Any]:
    return load_all_data()


@st.cache_data(show_spinner=False)
def image_data_uri(path_text: str) -> str:
    path = Path(path_text)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ── 유틸리티 함수 ──────────────────────────────────────────
def clean_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_score(value: Any, digits: int = 1) -> str:
    number = clean_number(value)
    if number is None:
        return "자료 없음"
    return f"{number:.{digits}f}"


def format_percent(value: Any, digits: int = 1) -> str:
    number = clean_number(value)
    if number is None:
        return "자료 없음"
    return f"{number * 100:.{digits}f}%"


def display_top_dong_name(value: Any) -> str:
    name = str(value)
    if name == "양평동2가":
        return "양평동 2가"
    return name


def render_back_link(icon_uri: str) -> None:
    st.markdown(
        f"""
        <a class="back-icon-link" href="?back=1" target="_self" aria-label="서울 전체로 돌아가기">
          <img src="{icon_uri}" alt="" />
        </a>
        """,
        unsafe_allow_html=True,
    )


def iter_lonlat(geometry: dict[str, Any]) -> Iterable[tuple[float, float]]:
    if geometry["type"] == "Polygon":
        for ring in geometry["coordinates"]:
            for lon, lat in ring:
                yield float(lon), float(lat)
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            for ring in polygon:
                for lon, lat in ring:
                    yield float(lon), float(lat)


def iter_rings(geometry: dict[str, Any]) -> Iterable[list[list[float]]]:
    if geometry["type"] == "Polygon":
        yield from geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            yield from polygon


def svg_path_from_geometry(
    geometry: dict[str, Any],
    min_lon: float,
    max_lat: float,
    scale: float,
    padding: float,
) -> str:
    commands: list[str] = []
    for ring in iter_rings(geometry):
        if len(ring) < 3:
            continue
        points = [
            (padding + (float(lon) - min_lon) * scale, padding + (max_lat - float(lat)) * scale)
            for lon, lat in ring
        ]
        first_x, first_y = points[0]
        path = [f"M {first_x:.1f} {first_y:.1f}"]
        path.extend(f"L {x:.1f} {y:.1f}" for x, y in points[1:])
        path.append("Z")
        commands.append(" ".join(path))
    return " ".join(commands)


def color_from_range(value: Any, minimum: float, maximum: float) -> str:
    number = clean_number(value)
    if number is None:
        return MAP_MISSING_COLOR
    if maximum <= minimum:
        ratio = 0.6
    else:
        ratio = max(0.0, min((number - minimum) / (maximum - minimum), 1.0))
    color_idx = min(int(ratio * len(MAP_BLUE_COLORS)), len(MAP_BLUE_COLORS) - 1)
    return MAP_BLUE_COLORS[color_idx]


def interpolate_color_hex(score: Any) -> str:
    """점수를 hex 색상으로 변환 (법정동 지도용)."""
    number = clean_number(score)
    if number is None:
        return MAP_MISSING_COLOR
    ratio = max(0.0, min(number / 100.0, 1.0))
    color_idx = min(int(ratio * len(MAP_BLUE_COLORS)), len(MAP_BLUE_COLORS) - 1)
    return MAP_BLUE_COLORS[color_idx]


# ── 데이터 빌드 함수 ──────────────────────────────────────
def build_gu_summary(legal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gu, group in legal.groupby("시군구명"):
        scored = group.loc[group["우선지원점수"].notna()].sort_values("우선지원점수", ascending=False)
        top3 = scored.head(3)
        top_row = scored.iloc[0] if not scored.empty else group.iloc[0]
        rows.append(
            {
                "시군구명": gu,
                "서울_위험도": top3["우선지원점수"].mean() if not top3.empty else None,
                "최고점수": scored["우선지원점수"].max() if not scored.empty else None,
                "대표법정동": top_row["법정동명"],
                "법정동수": int(group["법정동코드"].nunique()),
                "Top30_법정동수": int(group["is_top30"].sum()),
                "주거비자료_커버리지": group["관측품질"].eq("주거비관측").mean(),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values("서울_위험도", ascending=False, na_position="last").reset_index(drop=True)


def render_summary_cards(items: list[dict[str, str]]) -> None:
    cards = "\n".join(
        (
            '<div class="summary-card">'
            f'<div class="summary-label">{escape(item["label"])}</div>'
            f'<div class="summary-value">{escape(item["value"])}</div>'
            f'<div class="summary-caption">{escape(item.get("caption", ""))}</div>'
            "</div>"
        )
        for item in items
    )
    st.markdown(f"<div class='summary-grid'>{cards}</div>", unsafe_allow_html=True)


def render_glossary_group(title: str, items: list[dict[str, str]]) -> None:
    terms = "\n".join(
        (
            '<div class="glossary-item">'
            f'<div class="glossary-term">{escape(item["term"])}</div>'
            f'<div class="glossary-desc">{escape(item["desc"])}</div>'
            "</div>"
        )
        for item in items
    )
    st.markdown(
        f"""
        <div class="glossary-group">
          <div class="glossary-title">{escape(title)}</div>
          {terms}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_type_glossary() -> None:
    risk_descriptions = {
        "강제잔류·압박형": "취약계층 밀집도와 주거비 압박이 모두 높아, 이탈도 어렵고 부담도 큰 최우선 위험 유형입니다.",
        "이탈진행형": "주거비 압박은 높지만 취약계층 밀집도는 낮아, 기존 취약계층이 이미 밀려났을 가능성이 있는 유형입니다.",
        "잠재위험형": "취약계층 밀집도는 높지만 현재 주거비 압박은 상대적으로 낮아, 선제 모니터링이 필요한 유형입니다.",
        "안정형": "두 지표가 모두 상대적으로 낮아 현재 위험 신호가 약한 유형입니다.",
        "자료없음": "주거비 등 일부 핵심 자료가 부족해 위험유형을 판단하기 어려운 지역입니다.",
    }
    rows = "\n".join(
        (
            '<div class="risk-term">'
            f'<span class="legend-dot" style="background:{color}"></span>'
            '<div>'
            f'<div class="glossary-term">{escape(risk)}</div>'
            f'<div class="glossary-desc">{escape(risk_descriptions[risk])}</div>'
            "</div>"
            "</div>"
        )
        for risk, color in RISK_COLORS.items()
    )
    st.markdown(
        f"""
        <div class="glossary-group">
          <div class="glossary-title">위험유형</div>
          {rows}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 지도 렌더링: 서울 전체 (자치구 레벨) ────────────────────
def render_seoul_gu_map(
    geojson: dict[str, Any],
    legal: pd.DataFrame,
    selected_gu: str | None,
    metric_label: str,
) -> str | None:
    """서울 전체 자치구 지도를 클릭 가능한 정적 SVG로 렌더링."""

    summary = build_gu_summary(legal)
    metric_col, unit, _description = SEOUL_MAP_METRICS[metric_label]
    value_by_gu = summary.set_index("시군구명")[metric_col].to_dict()
    summary_by_gu = summary.set_index("시군구명").to_dict("index")
    legal_by_code8 = legal.set_index("법정동코드8").to_dict("index")
    clean_values = [clean_number(v) for v in value_by_gu.values()]
    clean_values = [v for v in clean_values if v is not None]
    minimum = min(clean_values) if clean_values else 0
    maximum = max(clean_values) if clean_values else 1

    all_points = [point for feature in geojson["features"] for point in iter_lonlat(feature["geometry"])]
    if not all_points:
        st.info("지도 경계 데이터를 불러오지 못했습니다.")
        return None

    min_lon = min(point[0] for point in all_points)
    max_lon = max(point[0] for point in all_points)
    min_lat = min(point[1] for point in all_points)
    max_lat = max(point[1] for point in all_points)
    padding = 34.0
    svg_width = 920.0
    scale = (svg_width - padding * 2) / max(max_lon - min_lon, 0.001)
    svg_height = (max_lat - min_lat) * scale + padding * 2

    label_points: dict[str, list[tuple[float, float]]] = {}
    paths: list[str] = []

    for feature in geojson["features"]:
        row = legal_by_code8.get(str(feature["properties"]["EMD_CD"]))
        if row is None:
            continue
        gu = str(row["시군구명"])
        color = color_from_range(value_by_gu.get(gu), minimum, maximum)
        label_points.setdefault(gu, []).extend(iter_lonlat(feature["geometry"]))
        path_data = svg_path_from_geometry(feature["geometry"], min_lon, max_lat, scale, padding)
        if path_data:
            paths.append(
                f'<path class="gu-shape" d="{path_data}" fill="{color}" stroke="{MAP_STROKE_COLOR}" />'
            )

    labels: list[str] = []
    for gu, points in label_points.items():
        if not points:
            continue
        lon = sum(p[0] for p in points) / len(points)
        lat = sum(p[1] for p in points) / len(points)
        dx, dy = GU_LABEL_OFFSETS.get(gu, (0.0, 0.0))
        lon += dx
        lat += dy
        x = padding + (lon - min_lon) * scale
        y = padding + (max_lat - lat) * scale
        item = summary_by_gu.get(gu, {})
        value = clean_number(item.get(metric_col))
        if unit == "%":
            value_text = format_percent(value, 0)
        elif unit == "점":
            value_text = f"{value:.1f}점" if value is not None else "자료 없음"
        else:
            value_text = f"{int(value):,}{unit}" if value is not None else "자료 없음"

        is_selected = gu == selected_gu
        label_color = MAP_SELECTED_LABEL if is_selected else MAP_LABEL
        href = f"/?drill_gu={quote(gu)}"
        left = x / svg_width * 100
        top = y / svg_height * 100
        labels.append(
            f'<a class="gu-label-link" href="{href}" target="_self" aria-label="{escape(gu)} 법정동 상세 보기" '
            f'style="left:{left:.3f}%; top:{top:.3f}%; background:{label_color};">'
            f'<span class="gu-name">{escape(gu)}</span>'
            f'<span class="gu-value">{escape(value_text)}</span>'
            "</a>"
        )

    html = f"""
    <!-- dong-map-component-v2 -->
    <style>
      .seoul-map-wrap {{
        position: relative;
        width: 100%;
        background: white;
      }}
      .seoul-svg {{
        width: 100%;
        height: auto;
        display: block;
        background: white;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      .gu-shape {{
        stroke-width: 0.75;
        vector-effect: non-scaling-stroke;
        pointer-events: none;
      }}
      .gu-label-link {{
        position: absolute;
        width: 64px;
        min-height: 44px;
        transform: translate(-50%, -50%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: white !important;
        text-decoration: none !important;
        border: 1px solid rgba(15,23,42,0.62);
        box-sizing: border-box;
        cursor: pointer;
        line-height: 1.15;
        transition: filter 120ms ease, opacity 120ms ease;
      }}
      .gu-label-link:hover {{
        color: white !important;
        filter: brightness(1.08);
        opacity: 0.96;
      }}
      .gu-name {{
        font-size: 13px;
        font-weight: 800;
      }}
      .gu-value {{
        font-size: 12px;
        font-weight: 650;
      }}
    </style>
    <div class="seoul-map-wrap">
      <svg class="seoul-svg" viewBox="0 0 {svg_width:.1f} {svg_height:.1f}" role="img" aria-label="서울 전체 자치구 지도">
        <g>{''.join(paths)}</g>
      </svg>
      {''.join(labels)}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    return None


# ── 지도 렌더링: 자치구 내 법정동 (드릴다운) ────────────────
def render_gu_dong_map(
    geojson: dict[str, Any],
    legal: pd.DataFrame,
    gu: str,
    initial_zoom: float = 1.0,
    initial_scroll_x: float = 0.0,
    initial_scroll_y: float = 0.0,
) -> str | None:
    """선택한 자치구의 법정동 지도를 클릭 가능한 정적 SVG로 렌더링."""

    legal_by_code8 = legal.set_index("법정동코드8").to_dict("index")
    gu_legal = legal.loc[legal["시군구명"].eq(gu)]

    all_points = []
    for feature in geojson["features"]:
        row = legal_by_code8.get(str(feature["properties"]["EMD_CD"]))
        if row and str(row["시군구명"]) == gu:
            all_points.extend(iter_lonlat(feature["geometry"]))
    if not all_points:
        st.info("이 자치구의 지도 경계 데이터를 불러오지 못했습니다.")
        return None

    min_lon = min(point[0] for point in all_points)
    max_lon = max(point[0] for point in all_points)
    min_lat = min(point[1] for point in all_points)
    max_lat = max(point[1] for point in all_points)
    padding = 36.0
    svg_width = 920.0
    scale = min(
        (svg_width - padding * 2) / max(max_lon - min_lon, 0.001),
        560.0 / max(max_lat - min_lat, 0.001),
    )
    svg_height = (max_lat - min_lat) * scale + padding * 2

    paths: list[str] = []
    dong_centers: dict[str, dict] = {}
    selected_code = st.session_state.get("selected_dong_code")

    for feature in geojson["features"]:
        code8 = str(feature["properties"]["EMD_CD"])
        row = legal_by_code8.get(code8)
        if row is None or str(row["시군구명"]) != gu:
            continue

        score = clean_number(row.get("우선지원점수"))
        fill_color = interpolate_color_hex(score)
        legal_code = str(row["법정동코드"])
        dong_name = str(row.get("법정동명") or row.get("표시명"))

        pts = list(iter_lonlat(feature["geometry"]))
        if pts:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            if legal_code not in dong_centers:
                dong_centers[legal_code] = {
                    "lon": cx, "lat": cy, "name": dong_name,
                    "score": score, "code": legal_code,
                }

        path_data = svg_path_from_geometry(feature["geometry"], min_lon, max_lat, scale, padding)
        if path_data:
            paths.append(
                f'<path class="dong-shape" d="{path_data}" fill="{fill_color}" />'
            )

    labels: list[str] = []
    for code, info in dong_centers.items():
        x = padding + (info["lon"] - min_lon) * scale
        y = padding + (max_lat - info["lat"]) * scale
        left = x / svg_width * 100
        top = y / svg_height * 100
        label_color = MAP_SELECTED_LABEL if code == selected_code else MAP_LABEL
        labels.append(
            f'<div class="dong-label" aria-label="{escape(info["name"])}" '
            f'style="left:{left:.3f}%; top:{top:.3f}%; background:{label_color};">'
            f'{escape(info["name"])}</div>'
        )

    min_zoom = 1.0
    max_zoom = 3.0
    zoom = max(min_zoom, min(initial_zoom, max_zoom))
    scroll_height = min(640, max(420, int(svg_height + 18)))
    component_height = min(660, max(440, int(svg_height + 38)))

    html = f"""
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        background: white;
        overflow: hidden;
      }}
      .dong-map-scroll {{
        height: {scroll_height}px;
        overflow: auto;
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overscroll-behavior: contain;
      }}
      .dong-map-wrap {{
        position: relative;
        width: min(100%, {svg_width:.1f}px);
        margin: 0 auto;
        background: white;
      }}
      .dong-svg {{
        width: 100%;
        height: auto;
        display: block;
        background: white;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      .dong-shape {{
        stroke: white;
        stroke-width: 0.85;
        vector-effect: non-scaling-stroke;
        pointer-events: none;
      }}
      .dong-label {{
        position: absolute;
        min-width: 50px;
        max-width: 96px;
        transform: translate(-50%, -50%);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: white !important;
        text-decoration: none !important;
        border: 1px solid rgba(15,23,42,0.62);
        box-sizing: border-box;
        line-height: 1.12;
        padding: 3px 5px;
        font-size: 10px;
        font-weight: 800;
        text-align: center;
        word-break: keep-all;
      }}
    </style>
    <div class="dong-map-scroll" id="dong-map-scroll">
      <div class="dong-map-wrap">
        <svg class="dong-svg" viewBox="0 0 {svg_width:.1f} {svg_height:.1f}" role="img" aria-label="{escape(gu)} 법정동 지도">
          <g>{''.join(paths)}</g>
        </svg>
        {''.join(labels)}
      </div>
    </div>
    <script>
      const box = document.getElementById("dong-map-scroll");
      const map = box.querySelector(".dong-map-wrap");
      const nativeWidth = {svg_width:.1f};
      const minZoom = {min_zoom:.1f};
      const maxZoom = {max_zoom:.1f};
      let zoom = {zoom:.3f};
      let fitWidth = Math.min(box.clientWidth, nativeWidth);
      map.style.width = `${{fitWidth * zoom}}px`;
      map.style.maxWidth = "none";
      box.scrollLeft = {max(0.0, initial_scroll_x):.1f};
      box.scrollTop = {max(0.0, initial_scroll_y):.1f};
      box.addEventListener("wheel", (event) => {{
        if (!event.ctrlKey) return;
        event.preventDefault();
        const oldZoom = zoom;
        zoom = Math.min(maxZoom, Math.max(minZoom, zoom * (event.deltaY < 0 ? 1.12 : 0.89)));
        const rect = box.getBoundingClientRect();
        const pointerX = event.clientX - rect.left;
        const pointerY = event.clientY - rect.top;
        const contentX = box.scrollLeft + pointerX;
        const contentY = box.scrollTop + pointerY;
        const ratio = zoom / oldZoom;
        map.style.width = `${{fitWidth * zoom}}px`;
        box.scrollLeft = contentX * ratio - pointerX;
        box.scrollTop = contentY * ratio - pointerY;
      }}, {{ passive: false }});
      window.addEventListener("resize", () => {{
        if (zoom !== minZoom) return;
        fitWidth = Math.min(box.clientWidth, nativeWidth);
        map.style.width = `${{fitWidth}}px`;
      }});
    </script>
    """
    components.html(html, height=component_height, scrolling=False)
    return None


# ── 법정동 상세 카드 ──────────────────────────────────────
def render_legal_detail_card(row: pd.Series, rent: pd.DataFrame) -> None:
    """법정동 상세 정보를 expander 안에 깔끔하게 렌더링."""

    # 상단 핵심 지표
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("우선지원 점수", format_score(row["우선지원점수"]))
    c2.metric("서울 순위", "-" if pd.isna(row["우선지원순위"]) else f"{int(row['우선지원순위']):,}위")
    c3.metric("위험유형", str(row["위험유형"]))
    c4.metric("핫스팟", str(row["hotspot_class"]))

    # 행정동 칩
    admins = row["행정동목록"] if isinstance(row["행정동목록"], list) else []
    if admins:
        st.markdown(
            " ".join(f"<span class='chip'>{a}</span>" for a in admins),
            unsafe_allow_html=True,
        )

    # 탭으로 세부정보 분리
    tab_indicator, tab_rent = st.tabs(["세부 지표", "전월세 상승률"])

    with tab_indicator:
        detail = pd.DataFrame([
            {"지표": "청년비율", "값": format_percent(row["청년비율"])},
            {"지표": "1인가구비율", "값": format_percent(row["1인가구비율"])},
            {"지표": "기초수급가구비율", "값": format_percent(row["기초수급가구비율"])},
            {"지표": "CI (취약계층 밀집도)", "값": format_score(row["concentration_index"], 3)},
            {"지표": "주거비 압박지수", "값": format_score(row["주거비_압박지수"], 3)},
            {"지표": "관측품질", "값": str(row["관측품질"])},
        ])
        st.dataframe(detail, hide_index=True, use_container_width=True)

    with tab_rent:
        rent_rows = rent.loc[rent["법정동코드"].eq(row["법정동코드"])].copy()
        if rent_rows.empty:
            st.info("이 법정동의 전월세 상승률 자료가 없습니다.")
        else:
            chart_df = rent_rows[["구분", "2023→2025_상승률(%)", "2025_m2당_월환산임대료", "전체_거래건수", "신뢰도"]].copy()
            fig = px.bar(
                chart_df, x="구분", y="2023→2025_상승률(%)",
                text="2023→2025_상승률(%)", color_discrete_sequence=[ACCENT],
                hover_data=["2025_m2당_월환산임대료", "전체_거래건수", "신뢰도"],
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                height=280, margin=dict(l=8, r=8, t=18, b=8),
                showlegend=False, yaxis_title="상승률(%)", xaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ── 정책 브릿지 카드 ──────────────────────────────────────
def render_policy_reason(admin_row: pd.Series, linked_legal: pd.DataFrame) -> None:
    top_legal = linked_legal.sort_values("우선지원점수", ascending=False, na_position="last").head(3)
    top_names = ", ".join(top_legal["법정동명"].tolist())
    risk_counts = linked_legal["위험유형"].value_counts()
    main_risk = risk_counts.index[0] if len(risk_counts) else "자료없음"
    top30_count = int(admin_row["Top30_법정동수"])

    risk_text = {
        "강제잔류·압박형": "취약계층 밀집과 주거비 압박이 동시에 높아 직접 지원과 임대료 부담 완화가 우선입니다.",
        "이탈진행형": "주거비 압박은 높지만 취약계층 밀집은 낮아져 이미 유입·잔류 장벽이 커졌을 가능성을 봐야 합니다.",
        "잠재위험형": "취약계층 밀집은 높지만 현재 주거비 압박은 낮아 선제 모니터링과 예방적 지원이 적합합니다.",
        "안정형": "상대적으로 안정적이나 주변 고위험 법정동과 함께 관찰할 필요가 있습니다.",
    }.get(main_risk, "주거비 자료가 제한적이므로 매핑 근거와 주변 지역을 함께 확인해야 합니다.")

    st.markdown(
        f"""
        <div class="reason-box">
            <div class="reason-title">왜 {admin_row['행정동명']}인가</div>
            <p><b>{top_names}</b> 법정동 근거가 이 행정동의 정책 점수를 끌어올립니다.
            Top30 법정동은 <b>{top30_count}개</b> 연결되어 있고, 주거비 자료 커버리지는
            <b>{format_percent(admin_row['주거비자료_커버리지'])}</b>입니다.</p>
            <p>{risk_text}</p>
            <p class="note">행정동 점수는 예산 총량 배분값이 아니라 법정동 분석을 정책 집행 단위로 번역한 우선순위 신호입니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 메인 ──────────────────────────────────────────────────
def main() -> None:
    back_icon_uri = image_data_uri(str(BACK_ICON_PATH))

    # ── 커스텀 스타일 ──
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        h1 { letter-spacing: 0; font-size: 2.3rem; line-height: 1.16; }
        .title-gap { height: 1.25rem; }
        .chip {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            margin: 0.15rem 0.2rem 0.15rem 0;
            border-radius: 999px;
            background: #F1F5F9;
            color: #334155;
            font-size: 0.82rem;
        }
        .reason-box {
            border-left: 4px solid #C2410C;
            background: #FFF7ED;
            padding: 0.9rem 1rem;
            margin: 0.5rem 0 1rem 0;
            border-radius: 0 8px 8px 0;
        }
        .reason-title { font-weight: 700; color: #7C2D12; margin-bottom: 0.35rem; }
        .note { color: #64748B; font-size: 0.88rem; margin-bottom: 0; }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.75rem;
            align-items: stretch;
        }
        .summary-card {
            min-height: 118px;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-width: 0;
        }
        .summary-label {
            color: #475569;
            font-size: 0.82rem;
            font-weight: 650;
            line-height: 1.25;
        }
        .summary-value {
            color: #0F172A;
            font-size: 1.36rem;
            font-weight: 760;
            line-height: 1.15;
            overflow-wrap: anywhere;
            word-break: keep-all;
        }
        .summary-caption {
            color: #64748B;
            font-size: 0.8rem;
            line-height: 1.25;
            min-height: 1.05rem;
            overflow-wrap: anywhere;
        }
        @media (max-width: 1200px) {
            .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 720px) {
            .summary-grid { grid-template-columns: 1fr; }
        }
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            padding: 0.8rem;
            border-radius: 8px;
        }
        .back-icon-link {
            width: 42px;
            height: 42px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            text-decoration: none;
            border: none;
            background: transparent;
            margin-top: 0.2rem;
        }
        .back-icon-link:hover {
            background: #F8FAFC;
            border-radius: 8px;
        }
        .back-icon-link img {
            width: 30px;
            height: 30px;
            display: block;
            object-fit: contain;
        }
        .glossary-group {
            border-top: 1px solid #CBD5E1;
            padding-top: 1rem;
            margin-top: 1rem;
        }
        .glossary-title {
            color: #0F172A;
            font-size: 0.98rem;
            font-weight: 800;
            margin-bottom: 0.72rem;
        }
        .glossary-item {
            margin-bottom: 0.82rem;
        }
        .glossary-term {
            color: #1E293B;
            font-size: 0.88rem;
            font-weight: 750;
            line-height: 1.28;
        }
        .glossary-desc {
            color: #64748B;
            font-size: 0.79rem;
            line-height: 1.42;
            margin-top: 0.14rem;
            word-break: keep-all;
        }
        .risk-term {
            display: grid;
            grid-template-columns: 12px 1fr;
            gap: 0.55rem;
            align-items: start;
            margin-bottom: 0.86rem;
        }
        .legend-dot {
            width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0;
            margin-top: 0.18rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    data = cached_data()
    legal = data["legal"]
    quality = data["quality"]

    # ── 헤더 ──
    st.title("서울 청년·임차가구 생활권 이탈 위험 대시보드")
    st.markdown("<div class='title-gap'></div>", unsafe_allow_html=True)

    metric_label = "우선지원 위험도"

    # ── 사이드바 ──
    with st.sidebar:
        st.header("📘 용어 설명")
        st.caption("대시보드에서 자주 쓰는 지표와 유형을 간단히 정리했습니다.")

        render_glossary_group(
            "핵심 지표",
            [
                {
                    "term": "우선지원 점수",
                    "desc": "취약계층 밀집도, 주거비 압박, 공공임대 여건을 결합해 산출한 최종 우선순위 점수입니다. 높을수록 정책 검토 우선도가 높습니다.",
                },
                {
                    "term": "취약계층 밀집도(CI)",
                    "desc": "청년비율, 1인가구비율, 기초수급가구비율을 정규화해 합친 지표입니다. 청년·저소득 주거취약 계층이 상대적으로 많이 모인 정도를 뜻합니다.",
                },
                {
                    "term": "주거비 압박지수",
                    "desc": "전월세 상승률과 임대료 수준을 반영해 주거비 부담이 커지는 정도를 나타낸 지표입니다.",
                },
                {
                    "term": "공공임대 보정",
                    "desc": "공공임대 접근성이 상대적으로 좋은 지역은 최종 위험도를 일부 낮춰 반영한 조정값입니다.",
                },
            ],
        )

        render_risk_type_glossary()

        render_glossary_group(
            "공간·정책 단위",
            [
                {
                    "term": "법정동 분석",
                    "desc": "주거비와 취약계층 밀집도를 세밀하게 보기 위해 법정동 단위로 위험도를 계산한 화면입니다.",
                },
                {
                    "term": "행정동 정책 브릿지",
                    "desc": "법정동 분석 결과를 실제 정책 집행 단위인 행정동으로 연결해 보여주는 화면입니다.",
                },
                {
                    "term": "Top30",
                    "desc": "서울시 법정동 중 최종 우선지원 점수가 높은 상위 30개 후보 지역입니다.",
                },
                {
                    "term": "핫스팟",
                    "desc": "해당 지역뿐 아니라 주변 지역까지 위험도가 함께 높은 공간적 군집 지역입니다.",
                },
            ],
        )

        render_glossary_group(
            "지도 읽기",
            [
                {
                    "term": "지도 색상",
                    "desc": "색이 진할수록 우선지원 점수 또는 선택한 위험 신호가 상대적으로 높습니다.",
                },
                {
                    "term": "자료없음",
                    "desc": "해당 법정동에 주거비 자료 등 필수 데이터가 부족해 일부 지표가 계산되지 않은 경우입니다.",
                },
            ],
        )

    # ── 상단 핵심 요약 (접을 수 있는 지표) ──
    with st.expander("📋 전체 분석 요약 지표", expanded=False):
        top = legal.sort_values("우선지원순위").head(1).iloc[0]
        render_summary_cards([
            {"label": "법정동 분석 단위", "value": f"{quality['법정동수']:,}개", "caption": "위험 분석 대상"},
            {"label": "행정동 정책 단위", "value": f"{quality['행정동수']:,}개", "caption": "집행 단위"},
            {"label": "연결쌍", "value": f"{quality['연결쌍수']:,}개", "caption": "법정동-행정동 매핑"},
            {"label": "다중 행정동 법정동", "value": f"{quality['다중행정동_법정동수']:,}개", "caption": "분할 매핑 대상"},
            {
                "label": "최우선 후보",
                "value": display_top_dong_name(top["법정동명"]),
                "caption": f"{top['시군구명']} · {format_score(top['우선지원점수'])}점",
            },
        ])
        if quality["Top30_매핑누락수"]:
            st.warning(f"Top30 법정동 중 행정동 매핑 누락이 {quality['Top30_매핑누락수']}개 있습니다.")

    # ── 세션 상태 초기화 ──
    gu_list = sorted(legal["시군구명"].dropna().unique().tolist())
    if "drill_gu" not in st.session_state:
        st.session_state["drill_gu"] = None  # None = 서울 전체 뷰
    if "selected_dong_code" not in st.session_state:
        st.session_state["selected_dong_code"] = None

    if st.query_params.get("back") == "1":
        st.session_state["drill_gu"] = None
        st.session_state["selected_dong_code"] = None
        st.session_state["_last_map_query"] = None
        st.query_params.clear()

    query_drill_gu = st.query_params.get("drill_gu")
    query_dong_code = st.query_params.get("selected_dong_code")
    query_signature = (query_drill_gu, query_dong_code)
    if query_drill_gu in gu_list and query_signature != st.session_state.get("_last_map_query"):
        st.session_state["drill_gu"] = query_drill_gu
        query_gu_codes = set(
            legal.loc[legal["시군구명"].eq(query_drill_gu), "법정동코드"].astype(str)
        )
        if query_dong_code in query_gu_codes:
            st.session_state["selected_dong_code"] = query_dong_code
            st.session_state["_selected_from_query"] = True
        else:
            st.session_state["selected_dong_code"] = None
        st.session_state["_last_map_query"] = query_signature

    drill_gu = st.session_state["drill_gu"]

    # ─────────────────────────────────────────────────────
    # 뷰 분기: 서울 전체 vs 자치구 드릴다운
    # ─────────────────────────────────────────────────────
    if drill_gu is None:
        # ── 서울 전체 자치구 지도 ──
        st.subheader("서울 전체 자치구 지도")
        st.caption("자치구를 클릭하면 해당 자치구의 법정동 상세 지도로 이동합니다.")

        render_seoul_gu_map(data["geojson"], legal, None, metric_label)

        # ── 자치구 비교 테이블 ──
        with st.expander("📊 자치구별 비교 테이블", expanded=False):
            gu_summary = build_gu_summary(legal)
            display_df = gu_summary.copy()
            display_df["서울_위험도"] = display_df["서울_위험도"].apply(lambda x: format_score(x))
            display_df["최고점수"] = display_df["최고점수"].apply(lambda x: format_score(x))
            display_df["주거비자료_커버리지"] = display_df["주거비자료_커버리지"].apply(lambda x: format_percent(x))
            display_df.columns = ["자치구", "위험도(점)", "최고점수", "대표법정동", "법정동수", "Top30수", "주거비 커버리지"]
            st.dataframe(display_df, hide_index=True, use_container_width=True)

    else:
        # ── 자치구 드릴다운 뷰 ──
        gu = drill_gu
        gu_legal = legal.loc[legal["시군구명"].eq(gu)].copy()

        # 뒤로가기 버튼
        col_back, col_title = st.columns([0.32, 6])
        with col_back:
            render_back_link(back_icon_uri)
        with col_title:
            st.subheader(f"{gu} 법정동")

        # 자치구 요약 지표
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("법정동 수", f"{len(gu_legal):,}개")
        mc2.metric("최고 점수", format_score(gu_legal["우선지원점수"].max()))
        mc3.metric("Top30 포함", f"{int(gu_legal['is_top30'].sum()):,}개")
        mc4.metric("주거비 관측", f"{int(gu_legal['관측품질'].eq('주거비관측').sum()):,}개")

        # ── 법정동 지도 ──
        st.markdown('<div id="dong-map-area"></div>', unsafe_allow_html=True)
        st.caption("지도 위에서 Ctrl+스크롤로 확대/축소할 수 있습니다.")
        render_gu_dong_map(
            data["geojson"],
            legal,
            gu,
            1.0,
            0.0,
            0.0,
        )

        # ── 탭: 법정동 분석 / 행정동 브릿지 / 위험유형 분포 ──
        tab_dong, tab_admin, tab_chart = st.tabs([
            "🏘️ 법정동 분석", "🏛️ 행정동 정책 브릿지", "📊 위험유형 분포"
        ])

        # ── 탭1: 법정동 분석 ──
        with tab_dong:
            # 법정동 선택
            options = gu_legal.sort_values(
                ["우선지원순위", "법정동명"], na_position="last"
            )["법정동코드"].tolist()
            labels = dict(zip(gu_legal["법정동코드"], gu_legal["표시명"]))

            if options:
                selected_from_query = st.session_state.pop("_selected_from_query", False)
                selected_code = st.session_state.get("selected_dong_code")
                widget_code = st.session_state.get("dong-select")

                if selected_from_query and selected_code in options:
                    st.session_state["dong-select"] = selected_code
                elif widget_code in options:
                    selected_code = widget_code
                    st.session_state["selected_dong_code"] = selected_code
                else:
                    if selected_code not in options:
                        selected_code = options[0]
                    st.session_state["selected_dong_code"] = selected_code
                    st.session_state["dong-select"] = selected_code

                selected_code = st.selectbox(
                    "법정동 선택",
                    options=options,
                    index=options.index(selected_code) if selected_code in options else 0,
                    format_func=lambda c: labels.get(c, c),
                    key="dong-select",
                )
                st.session_state["selected_dong_code"] = selected_code

                row = legal.loc[legal["법정동코드"].eq(selected_code)].iloc[0]
                render_legal_detail_card(row, data["rent"])

            # 법정동 순위 테이블
            with st.expander("📋 전체 법정동 순위 테이블", expanded=False):
                table_cols = ["우선지원순위", "표시명", "우선지원점수", "위험유형", "hotspot_class"]
                st.dataframe(
                    gu_legal.sort_values("우선지원순위", na_position="last")[table_cols],
                    hide_index=True, use_container_width=True,
                )

        # ── 탭2: 행정동 정책 브릿지 ──
        with tab_admin:
            admin = data["admin_policy"]
            crosswalk = data["crosswalk"]
            gu_admin = admin.loc[admin["시군구명"].eq(gu)].copy()

            if gu_admin.empty:
                st.warning("선택한 자치구의 행정동 매핑 자료가 없습니다.")
            else:
                admin_options = gu_admin.sort_values("행정동_정책순위")["행정동코드"].tolist()
                admin_labels = {
                    r["행정동코드"]: f"{r['행정동명']} · {format_score(r['행정동_정책점수'])}점"
                    for _, r in gu_admin.iterrows()
                }
                selected_admin = st.selectbox(
                    "행정동 선택",
                    options=admin_options,
                    format_func=lambda c: admin_labels.get(c, c),
                    key="admin-select",
                )

                admin_row = admin.loc[admin["행정동코드"].eq(selected_admin)].iloc[0]
                linked_codes = set(crosswalk.loc[crosswalk["행정동코드"].eq(selected_admin), "법정동코드"])
                linked_legal = legal.loc[legal["법정동코드"].isin(linked_codes)].copy()

                # 행정동 지표
                ac1, ac2, ac3, ac4 = st.columns(4)
                ac1.metric("행정동 정책점수", format_score(admin_row["행정동_정책점수"]))
                ac2.metric("정책순위", f"{int(admin_row['행정동_정책순위']):,}위")
                ac3.metric("연결 법정동", f"{int(admin_row['연결_법정동수']):,}개")
                ac4.metric("Top30 법정동", f"{int(admin_row['Top30_법정동수']):,}개")

                render_policy_reason(admin_row, linked_legal)

                # 연결 법정동 테이블
                with st.expander("📋 연결 법정동 근거 테이블", expanded=True):
                    st.dataframe(
                        linked_legal.sort_values("우선지원점수", ascending=False, na_position="last")[
                            ["표시명", "우선지원순위", "우선지원점수", "위험유형", "관측품질", "hotspot_class"]
                        ],
                        hide_index=True, use_container_width=True,
                    )

                # 행정동 전체 순위
                with st.expander("📋 전체 행정동 순위 테이블", expanded=False):
                    st.dataframe(
                        gu_admin.sort_values("행정동_정책순위")[
                            ["행정동_정책순위", "행정동명", "행정동_정책점수", "대표_고위험법정동", "Top30_법정동수", "주거비자료_커버리지"]
                        ],
                        hide_index=True, use_container_width=True,
                    )

        # ── 탭3: 위험유형 분포 차트 ──
        with tab_chart:
            scatter = gu_legal.loc[
                gu_legal["주거비_압박지수"].notna() & gu_legal["concentration_index"].notna()
            ].copy()

            if scatter.empty:
                st.info("이 자치구에는 주거비 압박지수와 CI가 모두 관측된 법정동이 없습니다.")
            else:
                scatter["차트_점크기"] = scatter["우선지원점수"].clip(lower=0).fillna(0)
                fig = px.scatter(
                    scatter,
                    x="concentration_index", y="주거비_압박지수",
                    size="차트_점크기", color="위험유형",
                    size_max=28,
                    color_discrete_map=RISK_COLORS,
                    hover_name="표시명",
                    hover_data={
                        "우선지원점수": ":.1f",
                        "우선지원순위": True,
                        "관측품질": True,
                        "차트_점크기": False,
                    },
                )
                fig.update_layout(
                    height=420,
                    margin=dict(l=8, r=8, t=28, b=8),
                    xaxis_title="취약계층 밀집도 (CI)",
                    yaxis_title="주거비 압박지수",
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

            # 위험유형 비율 파이차트
            risk_dist = gu_legal["위험유형"].value_counts()
            if not risk_dist.empty:
                colors = [RISK_COLORS.get(r, "#CBD5E1") for r in risk_dist.index]
                fig_pie = go.Figure(data=[go.Pie(
                    labels=risk_dist.index, values=risk_dist.values,
                    marker=dict(colors=colors),
                    textinfo="label+percent", hole=0.35,
                )])
                fig_pie.update_layout(
                    height=350, margin=dict(l=8, r=8, t=28, b=8),
                    title=f"{gu} 위험유형 분포",
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)


if __name__ == "__main__":
    main()
