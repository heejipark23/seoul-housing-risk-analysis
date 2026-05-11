from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    page_title="서울 청년·임차가구 생활권 이탈 위험 대시보드",
    page_icon="🏙️",
    layout="wide",
)


# ── 데이터 로드 ────────────────────────────────────────────
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def cached_data() -> dict[str, Any]:
    return load_all_data()


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


def color_from_range(value: Any, minimum: float, maximum: float) -> str:
    number = clean_number(value)
    if number is None:
        return "rgba(226,232,240,0.9)"
    if maximum <= minimum:
        ratio = 0.6
    else:
        ratio = max(0.0, min((number - minimum) / (maximum - minimum), 1.0))
    rgb = [round(PALE_RGB[i] + (ACCENT_RGB[i] - PALE_RGB[i]) * ratio) for i in range(3)]
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.88)"


def interpolate_color_hex(score: Any) -> str:
    """점수를 hex 색상으로 변환 (법정동 지도용)."""
    number = clean_number(score)
    if number is None:
        return "#CBD5E1"
    ratio = max(0.0, min(number / 100.0, 1.0))
    rgb = [round(PALE_RGB[i] + (ACCENT_RGB[i] - PALE_RGB[i]) * ratio) for i in range(3)]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


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


# ── 지도 렌더링: 서울 전체 (자치구 레벨) ────────────────────
def render_seoul_gu_map(
    geojson: dict[str, Any],
    legal: pd.DataFrame,
    selected_gu: str | None,
    metric_label: str,
) -> str | None:
    """서울 전체 자치구 지도를 Plotly로 렌더링. 클릭된 자치구 이름 반환."""

    summary = build_gu_summary(legal)
    metric_col, unit, description = SEOUL_MAP_METRICS[metric_label]
    value_by_gu = summary.set_index("시군구명")[metric_col].to_dict()
    summary_by_gu = summary.set_index("시군구명").to_dict("index")
    legal_by_code8 = legal.set_index("법정동코드8").to_dict("index")
    clean_values = [clean_number(v) for v in value_by_gu.values()]
    clean_values = [v for v in clean_values if v is not None]
    minimum = min(clean_values) if clean_values else 0
    maximum = max(clean_values) if clean_values else 1

    fig = go.Figure()
    label_points: dict[str, list[tuple[float, float]]] = {}

    for feature in geojson["features"]:
        row = legal_by_code8.get(str(feature["properties"]["EMD_CD"]))
        if row is None:
            continue
        gu = str(row["시군구명"])
        color = color_from_range(value_by_gu.get(gu), minimum, maximum)
        label_points.setdefault(gu, []).extend(iter_lonlat(feature["geometry"]))
        for ring in iter_rings(feature["geometry"]):
            if len(ring) < 3:
                continue
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines", fill="toself",
                    fillcolor=color, line=dict(color=color, width=0.6),
                    hoverinfo="skip", showlegend=False,
                )
            )

    hover_x, hover_y, hover_gu, hover_text = [], [], [], []
    for gu, points in label_points.items():
        if not points:
            continue
        lon = sum(p[0] for p in points) / len(points)
        lat = sum(p[1] for p in points) / len(points)
        dx, dy = GU_LABEL_OFFSETS.get(gu, (0.0, 0.0))
        lon += dx
        lat += dy
        item = summary_by_gu.get(gu, {})
        value = clean_number(item.get(metric_col))
        if unit == "%":
            value_text = format_percent(value, 0)
        elif unit == "점":
            value_text = f"{value:.1f}점" if value is not None else "자료 없음"
        else:
            value_text = f"{int(value):,}{unit}" if value is not None else "자료 없음"

        is_selected = gu == selected_gu
        fig.add_annotation(
            x=lon, y=lat,
            text=f"<b>{gu}</b><br>{value_text}",
            showarrow=False,
            font=dict(color="white", size=13 if is_selected else 12),
            bgcolor="rgba(124,45,18,0.94)" if is_selected else "rgba(71,67,73,0.88)",
            bordercolor="rgba(15,23,42,0.92)" if is_selected else "rgba(71,67,73,0.88)",
            borderpad=5, opacity=0.96,
        )
        hover_x.append(lon)
        hover_y.append(lat)
        hover_gu.append(gu)
        hover_text.append(
            f"<b>{gu}</b><br>"
            f"{description}: {value_text}<br>"
            f"대표 법정동: {item.get('대표법정동', '-')}<br>"
            f"최고점수: {format_score(item.get('최고점수'))}<br>"
            f"Top30: {int(item.get('Top30_법정동수', 0))}개<br>"
            f"주거비 관측: {format_percent(item.get('주거비자료_커버리지'))}"
        )

    fig.add_trace(
        go.Scatter(
            x=hover_x, y=hover_y, mode="markers",
            marker=dict(size=66, color="rgba(71,67,73,0.01)"),
            customdata=hover_gu, hovertext=hover_text,
            hoverinfo="text", showlegend=False,
        )
    )
    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(visible=False, constrain="domain"),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )

    event = st.plotly_chart(
        fig, use_container_width=True,
        key="seoul-gu-click-map", on_select="rerun",
        selection_mode="points", config={"displayModeBar": False},
    )

    # 클릭 이벤트 처리
    selection = getattr(event, "selection", None) if event else None
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    points = (selection.get("points", []) if isinstance(selection, dict)
              else getattr(selection, "points", []))
    if points:
        pt = points[0]
        if isinstance(pt, dict):
            cd = pt.get("customdata")
            if isinstance(cd, str):
                return cd
    return None


# ── 지도 렌더링: 자치구 내 법정동 (드릴다운) ────────────────
def render_gu_dong_map(
    geojson: dict[str, Any],
    legal: pd.DataFrame,
    gu: str,
) -> str | None:
    """선택한 자치구의 법정동 지도를 Plotly로 렌더링. 클릭된 법정동코드 반환."""

    legal_by_code8 = legal.set_index("법정동코드8").to_dict("index")
    gu_legal = legal.loc[legal["시군구명"].eq(gu)]

    fig = go.Figure()
    dong_centers: dict[str, dict] = {}

    for feature in geojson["features"]:
        code8 = str(feature["properties"]["EMD_CD"])
        row = legal_by_code8.get(code8)
        if row is None or str(row["시군구명"]) != gu:
            continue

        score = clean_number(row.get("우선지원점수"))
        fill_color = interpolate_color_hex(score)
        legal_code = str(row["법정동코드"])
        dong_name = str(row.get("표시명") or row.get("법정동명"))
        risk_type = str(row.get("위험유형") or "자료없음")

        pts = list(iter_lonlat(feature["geometry"]))
        if pts:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            if legal_code not in dong_centers:
                dong_centers[legal_code] = {
                    "lon": cx, "lat": cy, "name": dong_name,
                    "score": score, "risk": risk_type, "code": legal_code,
                    "rank": "-" if pd.isna(row.get("우선지원순위")) else str(int(row.get("우선지원순위"))),
                }

        for ring in iter_rings(feature["geometry"]):
            if len(ring) < 3:
                continue
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines", fill="toself",
                    fillcolor=fill_color,
                    line=dict(color="white", width=0.8),
                    hoverinfo="skip", showlegend=False,
                )
            )

    # 법정동 라벨 & 클릭용 invisible markers
    click_x, click_y, click_codes, click_texts = [], [], [], []
    for code, info in dong_centers.items():
        score_txt = format_score(info["score"])
        fig.add_annotation(
            x=info["lon"], y=info["lat"],
            text=f"<b>{info['name']}</b>",
            showarrow=False,
            font=dict(color="#1E293B", size=9),
            bgcolor="rgba(255,255,255,0.7)",
            borderpad=2, opacity=0.95,
        )
        click_x.append(info["lon"])
        click_y.append(info["lat"])
        click_codes.append(info["code"])
        click_texts.append(
            f"<b>{info['name']}</b><br>"
            f"우선지원 점수: {score_txt}<br>"
            f"순위: {info['rank']}<br>"
            f"위험유형: {info['risk']}"
        )

    fig.add_trace(
        go.Scatter(
            x=click_x, y=click_y, mode="markers",
            marker=dict(size=30, color="rgba(0,0,0,0.01)"),
            customdata=click_codes, hovertext=click_texts,
            hoverinfo="text", showlegend=False,
        )
    )

    # 뷰포트 계산
    all_pts = []
    for feature in geojson["features"]:
        row = legal_by_code8.get(str(feature["properties"]["EMD_CD"]))
        if row and str(row["시군구명"]) == gu:
            all_pts.extend(iter_lonlat(feature["geometry"]))
    if all_pts:
        lons = [p[0] for p in all_pts]
        lats = [p[1] for p in all_pts]
        x_range = [min(lons) - 0.003, max(lons) + 0.003]
        y_range = [min(lats) - 0.002, max(lats) + 0.002]
    else:
        x_range, y_range = None, None

    layout_kwargs = dict(
        height=580,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(visible=False, constrain="domain"),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    if x_range:
        layout_kwargs["xaxis"]["range"] = x_range
        layout_kwargs["yaxis"]["range"] = y_range
    fig.update_layout(**layout_kwargs)

    event = st.plotly_chart(
        fig, use_container_width=True,
        key=f"dong-map-{gu}", on_select="rerun",
        selection_mode="points", config={"displayModeBar": False},
    )

    selection = getattr(event, "selection", None) if event else None
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    points = (selection.get("points", []) if isinstance(selection, dict)
              else getattr(selection, "points", []))
    if points:
        pt = points[0]
        if isinstance(pt, dict):
            cd = pt.get("customdata")
            if isinstance(cd, str):
                return cd
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
    tab_indicator, tab_rent = st.tabs(["📊 세부 지표", "📈 전월세 상승률"])

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
            st.plotly_chart(fig, use_container_width=True)


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
    # ── 커스텀 스타일 ──
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        h1 { letter-spacing: 0; font-size: 2.3rem; line-height: 1.16; }
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
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            padding: 0.8rem;
            border-radius: 8px;
        }
        .back-btn-area { margin-bottom: 0.4rem; }
        /* 사이드바 리스크 범례 */
        .legend-item {
            display: flex; align-items: center; gap: 0.4rem;
            margin-bottom: 0.3rem; font-size: 0.85rem;
        }
        .legend-dot {
            width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    data = cached_data()
    legal = data["legal"]
    quality = data["quality"]

    # ── 헤더 ──
    st.title("🏙️ 서울 청년·임차가구 생활권 이탈 위험 대시보드")
    st.caption("법정동 분석 → 행정동 정책 번역 | 자치구를 클릭하면 법정동 상세 지도로 진입합니다")

    # ── 사이드바 ──
    with st.sidebar:
        st.header("🔎 탐색 조건")

        metric_label = st.selectbox(
            "서울 지도 지표",
            list(SEOUL_MAP_METRICS.keys()),
            help="서울 지도에서 자치구별로 표시할 핵심 수치를 선택합니다.",
        )

        st.divider()
        st.markdown("**위험유형 범례**")
        for risk, color in RISK_COLORS.items():
            st.markdown(
                f"<div class='legend-item'><div class='legend-dot' style='background:{color}'></div>{risk}</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption("색상은 우선지원 점수 기준입니다. 진할수록 위험도가 높습니다.")

    # ── 상단 핵심 요약 (접을 수 있는 지표) ──
    with st.expander("📋 전체 분석 요약 지표", expanded=False):
        top = legal.sort_values("우선지원순위").head(1).iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("법정동 분석 단위", f"{quality['법정동수']:,}개")
        c2.metric("행정동 정책 단위", f"{quality['행정동수']:,}개")
        c3.metric("연결쌍", f"{quality['연결쌍수']:,}개")
        c4.metric("다중 행정동 법정동", f"{quality['다중행정동_법정동수']:,}개")
        c5.metric("최우선 후보", "1위", f"{top['법정동명']} · {format_score(top['우선지원점수'])}점")
        if quality["Top30_매핑누락수"]:
            st.warning(f"Top30 법정동 중 행정동 매핑 누락이 {quality['Top30_매핑누락수']}개 있습니다.")

    # ── 세션 상태 초기화 ──
    gu_list = sorted(legal["시군구명"].dropna().unique().tolist())
    if "drill_gu" not in st.session_state:
        st.session_state["drill_gu"] = None  # None = 서울 전체 뷰
    if "selected_dong_code" not in st.session_state:
        st.session_state["selected_dong_code"] = None

    drill_gu = st.session_state["drill_gu"]

    # ─────────────────────────────────────────────────────
    # 뷰 분기: 서울 전체 vs 자치구 드릴다운
    # ─────────────────────────────────────────────────────
    if drill_gu is None:
        # ── 서울 전체 자치구 지도 ──
        st.subheader("서울 전체 자치구 지도")
        st.caption("자치구 버블을 클릭하면 해당 자치구의 법정동 상세 지도로 이동합니다.")

        clicked_gu = render_seoul_gu_map(data["geojson"], legal, None, metric_label)
        if clicked_gu and clicked_gu in gu_list:
            st.session_state["drill_gu"] = clicked_gu
            st.session_state["selected_dong_code"] = None
            st.rerun()

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
        col_back, col_title = st.columns([1, 5])
        with col_back:
            if st.button("← 서울 전체로", use_container_width=True):
                st.session_state["drill_gu"] = None
                st.session_state["selected_dong_code"] = None
                st.rerun()
        with col_title:
            st.subheader(f"📍 {gu} 법정동 상세")

        # 자치구 요약 지표
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("법정동 수", f"{len(gu_legal):,}개")
        mc2.metric("최고 점수", format_score(gu_legal["우선지원점수"].max()))
        mc3.metric("Top30 포함", f"{int(gu_legal['is_top30'].sum()):,}개")
        mc4.metric("주거비 관측", f"{int(gu_legal['관측품질'].eq('주거비관측').sum()):,}개")

        # ── 법정동 지도 ──
        st.caption("법정동을 클릭하면 아래에 상세 정보가 표시됩니다. 색이 진할수록 우선지원 점수가 높습니다.")
        clicked_dong = render_gu_dong_map(data["geojson"], legal, gu)
        if clicked_dong and clicked_dong in set(gu_legal["법정동코드"]):
            st.session_state["selected_dong_code"] = clicked_dong

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

            selected_code = st.session_state.get("selected_dong_code")
            if not selected_code or selected_code not in set(gu_legal["법정동코드"]):
                selected_code = options[0] if options else None

            if options:
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
                fig = px.scatter(
                    scatter,
                    x="concentration_index", y="주거비_압박지수",
                    size="우선지원점수", color="위험유형",
                    color_discrete_map=RISK_COLORS,
                    hover_name="표시명",
                    hover_data=["우선지원점수", "우선지원순위", "관측품질"],
                )
                fig.update_layout(
                    height=420,
                    margin=dict(l=8, r=8, t=28, b=8),
                    xaxis_title="취약계층 밀집도 (CI)",
                    yaxis_title="주거비 압박지수",
                )
                st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fig_pie, use_container_width=True)


if __name__ == "__main__":
    main()
