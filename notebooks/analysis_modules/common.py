from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from .config import PROJECT_ROOT


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[저장] {path.relative_to(PROJECT_ROOT)} shape={df.shape}")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "데이터 없음"

    table = df.copy().astype(str).replace({"\n": " "}, regex=True)
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(row[col]).replace("|", "/") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def configure_matplotlib() -> None:
    try:
        import matplotlib.font_manager as fm

        available_fonts = {font.name for font in fm.fontManager.ttflist}
        for candidate in ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"]:
            if candidate in available_fonts:
                plt.rcParams["font.family"] = candidate
                break
    except Exception:
        pass

    plt.rcParams["axes.unicode_minus"] = False
