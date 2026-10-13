#!/usr/bin/env python3


from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple


try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "[错误] 缺少依赖 openpyxl。\n"
        "请先安装：pip install openpyxl\n"
        "（openpyxl 已列于 requirements.txt）\n"
    )
    sys.exit(1)


_MODES: Tuple[str, ...] = ("hard", "soft", "free")
_COMPONENT_KEYS: Tuple[str, ...] = ("conjunction", "tenor", "vehicle")
_COMPONENT_SHORT: Dict[str, str] = {
    "conjunction": "C",
    "tenor": "T",
    "vehicle": "V",
}
_CLASSIFICATION_METRICS: Tuple[Tuple[str, str], ...] = (
    ("accuracy", "ACC"),
    ("precision", "PRE"),
    ("recall", "REC"),
    ("f1", "F1"),
)


_THIN = Side(style="thin", color="9E9E9E")
_MED = Side(style="medium", color="424242")
_LEFT_BORDER = Border(left=_THIN)
_HEADER_BORDER = Border(bottom=_MED)
_HEADER_BORDER_BLOCK = Border(bottom=_MED, left=_THIN)
_CAPTION_FONT = Font(bold=True, size=12, color="1A237E")
_HEADER_FONT = Font(bold=True, size=10, color="FFFFFF")
_HEADER_FILL = PatternFill(start_color="3949AB", end_color="3949AB", fill_type="solid")
_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
_LEFT = Alignment(horizontal="left", vertical="center")
_MODEL_COL_WIDTH = 30
_DATA_COL_WIDTH = 10.5


_FNAME_RE = re.compile(r"^metrics_(?P<model>.+?)_seed(?P<rest>.*)$")
_SEED_RE = re.compile(r"(?:^|_)seed([0-9]+)")
_TEMP_RE = re.compile(r"(?:^|_)t([0-9]+(?:\.[0-9]+)?)(?:_|$)")


def _parse_metrics_filename(stem: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    """从文件名（不含扩展名）解析出模型名与实验后缀。

    命名规则::

        metrics_{model_name}_seed{seed}_init{n}_few{n}_t{temp}_it{itemp}

    :param stem: 文件名去掉 ``.json`` 后的部分
    :return: ``(model_name, suffix, seed_str, temp_str)``；
        解析失败时回退为 ``(stem, "", None, None)``
    """
    m = _FNAME_RE.match(stem)
    if not m:

        return stem, "", None, None
    model = m.group("model")
    suffix = "seed" + m.group("rest")
    seed_m = _SEED_RE.search(suffix)
    temp_m = _TEMP_RE.search(suffix)
    seed_str = seed_m.group(1) if seed_m else None
    temp_str = temp_m.group(1) if temp_m else None
    return model, suffix, seed_str, temp_str


def _build_display_label(
    model: str, suffix: str, seed_str: Optional[str], temp_str: Optional[str],
    duplicated: bool,
) -> str:
    """构造 Model 列展示名。

    - 单一实验：直接用 ``model``。
    - 同模型多次实验：追加区分后缀，优先 ``(seedX, tY)``；
      若仍冲突则回退到完整 suffix。
    """
    if not duplicated:
        return model
    parts: List[str] = []
    if seed_str is not None:
        parts.append(f"seed{seed_str}")
    if temp_str is not None:
        parts.append(f"t{temp_str}")
    if parts:
        return f"{model} ({', '.join(parts)})"
    if suffix:
        return f"{model} ({suffix})"
    return model


class Column:
    """单个数据列的规格：表头、取值函数、是否为模式块起始列。"""

    __slots__ = ("header", "getter", "block_start")

    def __init__(
        self,
        header: str,
        getter: Callable[[Dict[str, Any]], Optional[float]],
        block_start: bool = False,
    ) -> None:
        self.header = header
        self.getter = getter
        self.block_start = block_start


def _make_location_columns() -> List[Column]:
    cols: List[Column] = []
    for i, mode in enumerate(_MODES):
        cols.append(Column(
            header=f"IoU-{mode}",
            getter=(lambda r: r.get("location")),
            block_start=(i == 0 or True),
        ))
    return cols


def _make_classification_columns(level: str) -> List[Column]:
    def getter_factory(metric_key: str) -> Callable[[Dict[str, Any]], Optional[float]]:
        def g(mode_results: Dict[str, Any]) -> Optional[float]:
            cls = mode_results.get("classification") or {}
            sub = cls.get(level) or {}
            val = sub.get(metric_key)
            return val if isinstance(val, (int, float)) else None
        return g

    cols: List[Column] = []
    for mode in _MODES:
        for j, (metric_key, short) in enumerate(_CLASSIFICATION_METRICS):
            cols.append(Column(
                header=f"{short}-{mode}",
                getter=getter_factory(metric_key),
                block_start=(j == 0),
            ))
    return cols


def _make_component_columns(field: str) -> List[Column]:
    def getter_factory(comp_key: str) -> Callable[[Dict[str, Any]], Optional[float]]:
        def g(mode_results: Dict[str, Any]) -> Optional[float]:
            comp = mode_results.get("component") or {}
            sub = comp.get(comp_key) or {}
            val = sub.get(field)
            return val if isinstance(val, (int, float)) else None
        return g

    cols: List[Column] = []
    for mode in _MODES:
        for j, comp_key in enumerate(_COMPONENT_KEYS):
            cols.append(Column(
                header=f"{_COMPONENT_SHORT[comp_key]}-{mode}",
                getter=getter_factory(comp_key),
                block_start=(j == 0),
            ))
    return cols


_TASK_SHEETS: List[Tuple[str, str, Callable[[], List[Column]]]] = [
    (
        "Location",
        "Results (in %) of Rhetorical Sentence Group Location Task",
        _make_location_columns,
    ),
    (
        "Classification-Rhetoric",
        "Results (in %) of Coarse-grained Rhetoric Classification Task",
        lambda: _make_classification_columns("rhetoric"),
    ),
    (
        "Classification-Form",
        "Results (in %) of Form-oriented Fine-grained Rhetoric Classification Task",
        lambda: _make_classification_columns("form"),
    ),
    (
        "Classification-Content",
        "Results (in %) of Content-oriented Fine-grained Rhetoric Classification Task",
        lambda: _make_classification_columns("content"),
    ),
    (
        "Component-Content",
        "Results (in %) of Rhetorical Component Understanding Task",
        lambda: _make_component_columns("content_sim"),
    ),
    (
        "Component-Location",
        "Results (in %) of Rhetorical Component Location Task",
        lambda: _make_component_columns("position_iou"),
    ),
]


class ExperimentRow:
    """一个 metrics JSON 文件对应的实验记录。"""

    __slots__ = ("filename", "label", "data")

    def __init__(self, filename: str, label: str, data: Dict[str, Any]) -> None:
        self.filename = filename
        self.label = label
        self.data = data

    def mode_results(self, mode: str) -> Dict[str, Any]:
        results = self.data.get("results") or {}
        entry = results.get(mode)
        return entry if isinstance(entry, dict) else {}


def load_experiments(metrics_dir: str) -> List[ExperimentRow]:
    """扫描目录加载全部 metrics JSON，按文件名字母序返回。"""
    if not os.path.isdir(metrics_dir):
        sys.stderr.write(f"[错误] metrics 目录不存在：{metrics_dir}\n")
        sys.exit(1)

    pattern = os.path.join(metrics_dir, "metrics_*.json")
    files = sorted(glob.glob(pattern))

    files = [f for f in files if not os.path.basename(f).startswith("._")]

    if not files:
        sys.stderr.write(f"[警告] 目录中未找到 metrics_*.json 文件：{metrics_dir}\n")
        return []


    parsed: List[Tuple[str, str, str, Optional[str], Optional[str]]] = []
    model_counts: Dict[str, int] = {}
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        model, suffix, seed_str, temp_str = _parse_metrics_filename(stem)
        parsed.append((path, stem, model, seed_str, temp_str))
        model_counts[model] = model_counts.get(model, 0) + 1


    rows: List[ExperimentRow] = []
    seen_labels: Dict[str, int] = {}
    for path, stem, model, seed_str, temp_str in parsed:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            sys.stderr.write(f"[警告] 跳过无法解析的文件 {os.path.basename(path)}：{exc}\n")
            continue
        if not isinstance(data, dict):
            sys.stderr.write(f"[警告] 跳过非对象 JSON：{os.path.basename(path)}\n")
            continue

        suffix = stem[len("metrics_"):] if stem.startswith("metrics_") else stem
        duplicated = model_counts.get(model, 0) > 1
        label = _build_display_label(model, suffix, seed_str, temp_str, duplicated)

        if label in seen_labels:
            label = f"{model} ({suffix})"
        seen_labels[label] = seen_labels.get(label, 0) + 1
        rows.append(ExperimentRow(os.path.basename(path), label, data))

    return rows


def _scale(value: Optional[float]) -> Optional[float]:
    """指标值 ×100 保留 2 位小数；None 透传。"""
    if value is None:
        return None
    try:
        return round(float(value) * 100.0, 2)
    except (TypeError, ValueError):
        return None


def _compute_emphasis(values: List[Optional[float]]) -> Tuple[set, set]:
    """逐列判定强调格式。

    :param values: 该列所有行的（已 ×100 的）数值，可能含 None
    :return: ``(bold_idx, underline_idx)`` 两个行索引集合
    """
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return set(), set()
    max_val = max(v for _, v in valid)
    bold_idx = {i for i, v in valid if v == max_val}
    underline_idx: set = set()

    if len(bold_idx) == 1:
        rest = [v for _, v in valid if v != max_val]
        if rest:
            second_val = max(rest)
            underline_idx = {i for i, v in valid if v == second_val}
    return bold_idx, underline_idx


def _write_caption(ws, caption: str, ncols: int) -> None:

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    cell = ws.cell(row=1, column=1, value=caption)
    cell.font = _CAPTION_FONT
    cell.alignment = _CENTER
    ws.row_dimensions[1].height = 26


def _write_task_sheet(ws, caption: str, columns: List[Column], rows: List[ExperimentRow]) -> None:

    ncols = len(columns) + 1
    _write_caption(ws, caption, ncols)


    header_row = 2
    hcell = ws.cell(row=header_row, column=1, value="Model")
    hcell.font = _HEADER_FONT
    hcell.fill = _HEADER_FILL
    hcell.alignment = _CENTER
    hcell.border = _HEADER_BORDER
    for c, col in enumerate(columns, start=2):
        cell = ws.cell(row=header_row, column=c, value=col.header)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = _CENTER
        cell.border = _HEADER_BORDER_BLOCK if col.block_start else _HEADER_BORDER


    data_start_row = 3
    for c, col in enumerate(columns, start=2):
        raw_values = [_scale(col.getter(_mode_results_for(r, col))) for r in rows]
        bold_idx, underline_idx = _compute_emphasis(raw_values)
        for ridx, val in enumerate(raw_values):
            cell = ws.cell(row=data_start_row + ridx, column=c)
            if val is not None:
                cell.value = val
                cell.number_format = "0.00"
            cell.alignment = _CENTER
            if col.block_start:
                cell.border = _LEFT_BORDER
            if ridx in bold_idx:
                cell.font = Font(bold=True)
            elif ridx in underline_idx:
                cell.font = Font(underline="single")


    for ridx, r in enumerate(rows):
        cell = ws.cell(row=data_start_row + ridx, column=1, value=r.label)
        cell.alignment = _LEFT


    ws.column_dimensions[get_column_letter(1)].width = _MODEL_COL_WIDTH
    for c in range(2, ncols + 1):
        ws.column_dimensions[get_column_letter(c)].width = _DATA_COL_WIDTH
    ws.freeze_panes = f"B{data_start_row}"


def mode_of(col: Column) -> str:  # pragma: no cover
    for m in _MODES:
        if col.header.endswith(f"-{m}"):
            return m
    return "hard"


def _mode_results_for(row: ExperimentRow, col: Column) -> Dict[str, Any]:
    """根据列表头后缀选择对应模式的 results 子字典。"""
    return row.mode_results(mode_of(col))


def _write_overview_sheet(ws, rows: List[ExperimentRow]) -> None:
    """写入实验概览：基本信息 + free 模式 Krippendorff's alpha。"""
    headers = [
        "Model", "File", "Total Samples", "Parse Successes", "Parse Failures",
        "Parse Failure Idxes",
        "Krippendorff α (Rhetoric)", "Krippendorff α (Form)", "Krippendorff α (Content)",
    ]
    ncols = len(headers)
    _write_caption(
        ws,
        "Experiment Overview (Krippendorff's α in %, free mode)",
        ncols,
    )

    header_row = 2
    alpha_start_col = 7
    for c, name in enumerate(headers, start=1):
        cell = ws.cell(row=header_row, column=c, value=name)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = _CENTER
        cell.border = _HEADER_BORDER_BLOCK if c == alpha_start_col else _HEADER_BORDER

    data_start_row = 3
    for ridx, r in enumerate(rows):
        row_i = data_start_row + ridx
        data = r.data
        free = r.mode_results("free")
        alpha = free.get("krippendorff_alpha") or {}

        failures_idxes = data.get("parse_failures_idxes") or []
        if isinstance(failures_idxes, list):
            idx_str = ", ".join(str(x) for x in failures_idxes)
        else:
            idx_str = str(failures_idxes)

        basic_cells = [
            (1, r.label, _LEFT, None),
            (2, r.filename, _LEFT, None),
            (3, data.get("total_samples"), _CENTER, "0"),
            (4, data.get("parse_successes"), _CENTER, "0"),
            (5, data.get("parse_failures"), _CENTER, "0"),
            (6, idx_str if idx_str else "-", _CENTER, None),
        ]
        for col, val, align, numfmt in basic_cells:
            cell = ws.cell(row=row_i, column=col, value=val)
            cell.alignment = align
            if numfmt:
                cell.number_format = numfmt


        for j, level in enumerate(("rhetoric", "form", "content")):
            col = alpha_start_col + j
            cell = ws.cell(row=row_i, column=col)
            aval = _scale(alpha.get(level)) if isinstance(alpha, dict) else None
            if aval is not None:
                cell.value = aval
                cell.number_format = "0.00"
            cell.alignment = _CENTER
            if j == 0:
                cell.border = _LEFT_BORDER


    widths = {1: _MODEL_COL_WIDTH, 2: 46, 3: 13, 4: 15, 5: 13, 6: 22, 7: 22, 8: 20, 9: 20}
    for c, w in widths.items():
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.freeze_panes = f"A{data_start_row}"


def build_workbook(rows: List[ExperimentRow]) -> Workbook:
    """根据实验记录构建完整工作簿。"""
    wb = Workbook()

    default_ws = wb.active
    wb.remove(default_ws)


    ov = wb.create_sheet("Overview")
    _write_overview_sheet(ov, rows)


    for sheet_name, caption, builder in _TASK_SHEETS:
        ws = wb.create_sheet(sheet_name)
        _write_task_sheet(ws, caption, builder(), rows)

    return wb


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="扫描 metrics JSON 生成实验结果汇总 Excel（对齐旧论文 Table 1-6）。",
    )
    parser.add_argument(
        "--metrics-dir",
        default="results/exp/metrics/",
        help="metrics 归档文件夹路径（默认 results/exp01/metrics/）",
    )
    parser.add_argument(
        "-o", "--output",
        default="results/exp/summary.xlsx",
        help="输出 xlsx 路径（默认 results/exp/summary.xlsx）",
    )
    args = parser.parse_args(argv)

    metrics_dir = args.metrics_dir
    output_path = args.output

    rows = load_experiments(metrics_dir)
    if not rows:
        sys.stderr.write("[错误] 无可用实验数据，未生成 Excel。\n")
        return 1

    wb = build_workbook(rows)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    wb.save(output_path)

    print(f"[完成] 已汇总 {len(rows)} 个实验 → {output_path}")
    print(f"       Sheets: {wb.sheetnames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
