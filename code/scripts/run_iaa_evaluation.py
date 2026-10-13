#!/usr/bin/env python3
"""人工标注三阶段 IAA（Inter-Annotator Agreement）复算脚本。

背景:
    两名标注者的结果已被合并进 ``data_for_IAA/`` 下的三个阶段文件：

        - data-pre-annotation-stage.json    预标注阶段
        - data-formal-annotation-stage.json 正式标注阶段
        - data-double-check-stage.json      交叉复核阶段

    每条记录中 ``rhetoricItems`` 存放标注者 A 的结构化结果，``rhetoricItems_pred``
    存放标注者 B 结果的 JSON 字符串（字段名沿用模型预测格式，纯属命名巧合，
    与任何模型输出无关）。

本脚本职责:
    - 只读加载三个阶段文件
    - 复用 ``new-code/src/evaluator.py`` 的 ``evaluate_all``（不修改该文件），把
      标注者 B 的字符串视为 predictions、标注者 A 的结构化结果视为 gold，
      在 hard / soft / free 三种模式下计算 location / classification / component
      指标；free 模式额外输出 Krippendorff's alpha
    - 汇总三阶段结果为 JSON + Markdown 表格，写入 new-code/results/iaa/

用法::

    cd new-code
    python scripts/run_iaa_evaluation.py \
        [--data-dir PATH] [--output-dir PATH] [--stages a,b,c] [--modes hard,soft,free]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.evaluator import CLASSIFICATION_LEVELS, evaluate_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("iaa")


STAGES: Dict[str, str] = {
    "pre-annotation": "data-pre-annotation-stage.json",
    "formal-annotation": "data-formal-annotation-stage.json",
    "double-check": "data-double-check-stage.json",
}

_VALID_MODES: Sequence[str] = ("hard", "soft", "free")

_DEFAULT_DATA_DIR = os.path.normpath(os.path.join(
    _PROJECT_ROOT, os.pardir,
    "data", "IAA", "data_for_IAA",
))
_DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "iaa")


def load_stage_file(path: str) -> List[Dict[str, Any]]:
    """只读加载单个阶段文件，并做最小结构校验。

    要求每条记录包含：``idx``、``sentenceList``、``rhetoricItems``（标注者 A，list）、
    ``rhetoricItems_pred``（标注者 B，JSON 字符串）。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} 应为非空 JSON 数组，实际: {type(data).__name__}")

    required_keys = {"idx", "sentenceList", "rhetoricItems", "rhetoricItems_pred"}
    for i, rec in enumerate(data):
        missing = required_keys - set(rec.keys())
        if missing:
            raise ValueError(f"{path} 第 {i} 条缺少字段: {sorted(missing)}")
        if not isinstance(rec["rhetoricItems"], list):
            raise ValueError(f"{path} 第 {i} 条 rhetoricItems 非 list（应为标注者 A 的结构化结果）")
        if not isinstance(rec["rhetoricItems_pred"], str):
            raise ValueError(f"{path} 第 {i} 条 rhetoricItems_pred 非 str（应为标注者 B 结果的 JSON 字符串）")

    idx_list = [rec["idx"] for rec in data]
    if len(set(idx_list)) != len(idx_list):
        logger.warning("%s 存在重复 idx（%d/%d），评测仍按原样逐条处理",
                       os.path.basename(path), len(set(idx_list)), len(idx_list))
    return data


def run_stage(
    stage_name: str,
    data: List[Dict[str, Any]],
    modes: Sequence[str],
) -> Dict[str, Any]:
    """对单个阶段跑 ``evaluate_all``。

    同一份记录同时充当 predictions（读取 ``rhetoricItems_pred`` 字段，即标注者 B）
    与 gold（读取 ``rhetoricItems`` / ``sentenceList`` 字段，即标注者 A）。
    """
    eval_modes = {
        "location": list(modes),
        "classification": list(modes),
        "component": list(modes),
    }
    metrics = evaluate_all(
        predictions=data,
        gold_data=data,
        eval_modes=eval_modes,
    )
    metrics["stage"] = stage_name
    metrics["modes"] = list(modes)
    return metrics


def _fmt(value: Any) -> str:
    """数值保留 4 位小数，None 显示为 N/A。"""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_markdown(summaries: Dict[str, Dict[str, Any]]) -> str:
    """把各阶段 metrics 组织成 Markdown 摘要表。"""
    lines: List[str] = []
    lines.append("# 人工标注三阶段 IAA 结果")
    lines.append("")
    lines.append("> 说明：本结果由 `new-code/scripts/run_iaa_evaluation.py` 复用 "
                 "`new-code/src/evaluator.py` 的 `evaluate_all` 计算得到。"
                 "每条记录的 `rhetoricItems` 为标注者 A 的结果，`rhetoricItems_pred` "
                 "为标注者 B 的结果（字段命名沿用模型预测格式，与模型输出无关）。")
    lines.append("")

    all_modes: List[str] = []
    for sm in summaries.values():
        for m in sm.get("modes", []):
            if m not in all_modes:
                all_modes.append(m)


    lines.append("## 总览")
    lines.append("")
    lines.append("| 阶段 | 样本数 | 解析成功 | 解析失败 |")
    lines.append("|---|---|---|---|")
    for stage, sm in summaries.items():
        lines.append(
            f"| {stage} | {sm['total_samples']} | "
            f"{sm['parse_successes']} | {sm['parse_failures']} |"
        )
    lines.append("")


    lines.append("## 句组定位 IoU")
    lines.append("")
    lines.append("| 阶段 | " + " | ".join(all_modes) + " |")
    lines.append("|---" * (len(all_modes) + 1) + "|")
    for stage, sm in summaries.items():
        row = [_fmt(sm["results"].get(m, {}).get("location")) for m in all_modes]
        lines.append(f"| {stage} | " + " | ".join(row) + " |")
    lines.append("")


    lines.append("## 修辞分类（三粒度 × hard/soft/free）")
    lines.append("")
    for level in CLASSIFICATION_LEVELS:
        lines.append(f"### {level}")
        lines.append("")
        header = ["阶段", "模式", "accuracy", "precision", "recall", "f1"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---" * len(header) + "|")
        for stage, sm in summaries.items():
            for m in sm.get("modes", []):
                cell = sm["results"].get(m, {}).get("classification", {}).get(level)
                if cell is None:
                    continue
                lines.append(
                    f"| {stage} | {m} | {_fmt(cell['accuracy'])} | "
                    f"{_fmt(cell['precision'])} | {_fmt(cell['recall'])} | {_fmt(cell['f1'])} |"
                )
        lines.append("")


    has_alpha = any(
        "krippendorff_alpha" in sm["results"].get("free", {})
        for sm in summaries.values()
    )
    if has_alpha:
        lines.append("## Krippendorff's alpha（free 模式，分类标签一致性）")
        lines.append("")
        header = ["阶段"] + list(CLASSIFICATION_LEVELS)
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---" * len(header) + "|")
        for stage, sm in summaries.items():
            alpha = sm["results"].get("free", {}).get("krippendorff_alpha", {})
            lines.append(
                f"| {stage} | " + " | ".join(_fmt(alpha.get(lv)) for lv in CLASSIFICATION_LEVELS) + " |"
            )
        lines.append("")


    lines.append("## 修辞成分（conjunction / tenor / vehicle）")
    lines.append("")
    header = ["阶段", "模式", "成分", "content_sim", "position_iou"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|---" * len(header) + "|")
    for stage, sm in summaries.items():
        for m in sm.get("modes", []):
            comp = sm["results"].get(m, {}).get("component")
            if not comp:
                continue
            for key in ("conjunction", "tenor", "vehicle"):
                cell = comp.get(key, {})
                lines.append(
                    f"| {stage} | {m} | {key} | {_fmt(cell.get('content_sim'))} | "
                    f"{_fmt(cell.get('position_iou'))} |"
                )
    lines.append("")

    return "\n".join(lines)


def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="复算人工标注三阶段的 IAA 结果（复用 new-code/src/evaluator.py）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=_DEFAULT_DATA_DIR,
                        help="三阶段数据目录（只读）")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR,
                        help="结果输出目录")
    parser.add_argument("--stages", default=None,
                        help="逗号分隔的阶段名子集，默认全部："
                             + ",".join(STAGES.keys()))
    parser.add_argument("--modes", default="hard,soft,free",
                        help="逗号分隔的评测模式子集（hard/soft/free）")
    args = parser.parse_args(argv)

    modes = _split_csv(args.modes) or list(_VALID_MODES)
    invalid = [m for m in modes if m not in _VALID_MODES]
    if invalid:
        raise ValueError(f"--modes 含非法模式 {invalid}，合法取值: {list(_VALID_MODES)}")

    stages = _split_csv(args.stages) or list(STAGES.keys())
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"--stages 含未知阶段 {unknown}，可选: {list(STAGES.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("数据目录: %s", args.data_dir)
    logger.info("输出目录: %s", args.output_dir)
    logger.info("阶段: %s | 模式: %s", stages, modes)

    summaries: Dict[str, Dict[str, Any]] = {}
    for stage in stages:
        path = os.path.join(args.data_dir, STAGES[stage])
        if not os.path.exists(path):
            logger.warning("跳过阶段 %s：文件不存在 %s", stage, path)
            continue
        data = load_stage_file(path)
        logger.info("<<<--- %s 阶段 (%d 条) --->>>", stage, len(data))
        metrics = run_stage(stage, data, modes)

        stage_metrics_path = os.path.join(args.output_dir, f"metrics_iaa_{stage}.json")
        with open(stage_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info("已保存单阶段指标: %s", stage_metrics_path)

        summaries[stage] = metrics

        free = metrics["results"].get("free", {})
        logger.info(
            "[%s] location(free)=%s | krippendorff_alpha=%s",
            stage,
            free.get("location"),
            json.dumps(free.get("krippendorff_alpha"), ensure_ascii=False),
        )

    if not summaries:
        logger.error("没有任何阶段成功评测，请检查 --data-dir / --stages")
        return 1

    combined_path = os.path.join(args.output_dir, "iaa_summary.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(args.output_dir, "iaa_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown(summaries))

    logger.info("三阶段汇总: %s | %s", combined_path, md_path)
    logger.info("IAA 复算完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
