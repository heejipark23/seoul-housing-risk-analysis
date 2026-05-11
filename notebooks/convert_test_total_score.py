from __future__ import annotations

import csv
import math
from pathlib import Path


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
INPUT_PATH = PROJECT_ROOT / "output" / "ldri" / "Test.csv"
OUTPUT_PATH = PROJECT_ROOT / "output" / "ldri" / "Test_total_100_sorted.csv"
TOTAL_COL = "총점"
RAW_TOTAL_COL = "총점_raw"


def parse_float(value: str) -> float:
    try:
        number = float(str(value).strip().replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"숫자로 변환할 수 없는 값입니다: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"유한한 숫자가 아닙니다: {value!r}")
    return number


def parse_optional_float(value: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return parse_float(value)


def format_number(value: float, digits: int = 6) -> str:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def convert_total_to_100(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if TOTAL_COL not in fieldnames:
        raise ValueError(f"'{TOTAL_COL}' 컬럼이 없습니다: {input_path}")

    totals = [parse_optional_float(row[TOTAL_COL]) for row in rows]
    valid_totals = [total for total in totals if total is not None]
    if not valid_totals:
        raise ValueError(f"'{TOTAL_COL}' 컬럼에 환산할 숫자 값이 없습니다.")

    max_total = max(valid_totals)
    if max_total <= 0:
        raise ValueError(f"'{TOTAL_COL}' 컬럼의 최대값이 0보다 커야 합니다.")

    result: list[dict[str, str]] = []
    for row, total in zip(rows, totals):
        converted = dict(row)
        converted[RAW_TOTAL_COL] = row[TOTAL_COL]
        converted[TOTAL_COL] = "" if total is None else format_number(total / max_total * 100)
        result.append(converted)

    result.sort(
        key=lambda row: parse_optional_float(row[TOTAL_COL]) or float("-inf"),
        reverse=True,
    )
    for rank, row in enumerate(result, start=1):
        row["순위"] = str(rank)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fieldnames = ["순위"]
    for fieldname in fieldnames:
        if fieldname == TOTAL_COL:
            output_fieldnames.extend([RAW_TOTAL_COL, TOTAL_COL])
        else:
            output_fieldnames.append(fieldname)

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(result)

    return result


def main() -> None:
    result = convert_total_to_100()
    print(f"saved: {OUTPUT_PATH.relative_to(PROJECT_ROOT)} rows={len(result):,}")


if __name__ == "__main__":
    main()
