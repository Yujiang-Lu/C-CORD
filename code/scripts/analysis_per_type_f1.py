#!/usr/bin/env python3


from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(1, os.path.join(_PROJECT_ROOT, "scripts"))


from src.evaluator import (  # noqa: E402
    RHETORIC_LABELS,
    _LABEL_INDEX,
    _extract_rhetoric_items,
    _flatten_rhetoric_list,
    _match_sentence_groups,
    _multi_hot,
    compute_classification,
    merge_rhetoric_items,
    parse_prediction,
)

from scripts.analysis_ch3_failure_cases import (  # noqa: E402
    RHET_EN,
    dedupe_labels,
    parse_model_meta,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("per_type_f1")

MODES: Tuple[str, ...] = ("hard", "free")
LEVEL = "rhetoric"


DISPLAY_ROWS: List[Tuple[str, str]] = [
    ("ZHIPU_GLM-5.3-FlashX|think|1.0", "\\model{GLM}"),
    ("claude-sonnet-5-5|think|1.0", "\\model{Claude}"),
    ("deepseek-flash|nothink|0.7", "\\model{DS-Flash}"),
    ("deepseek-v4-pro|nothink|0.7", "\\model{DS-V4-Pro}"),
    ("gemini-3.8-flash|think|1.0", "\\model{Gemini}"),
    ("gpt-6-luna|nothink|1.0", "\\model{GPT-Luna-NT}"),
    ("gpt-6-luna|think|1.0", "\\model{GPT-Luna-T}"),
    ("kimi-k3|think|1.0", "\\model{Kimi}"),
    ("qwen3.8-flash|nothink|0.7", "\\model{Qwen-Flash}"),
    ("qwen3.8-max|nothink|0.7", "\\model{Qwen-Max-t0.7}"),
    ("qwen3.8-max|nothink|1.0", "\\model{Qwen-Max-t1.0}"),
]


def row_key_of(meta: Dict[str, Any]) -> str:
    """把一条预测文件的 meta 映射到 DISPLAY_ROWS 的键。"""
    return f"{meta['model']}|{meta['mode']}|{meta['temp']}"


def type_counts(vec: np.ndarray) -> List[int]:
    """multi-hot 向量 → 每个类型位上的 1 的个数（multi-hot 每维至多 1）。"""
    return [int(x) for x in np.asarray(vec, dtype=int)]


def accumulate(doc_items: List[Tuple[Any, ...]], mode: str, acc: Dict[str, Dict[str, int]]) -> None:

    for gold_items, pred_items in doc_items:
        if mode == "free":
            gv = _multi_hot(_flatten_rhetoric_list(gold_items), LEVEL)
            pv = _multi_hot(_flatten_rhetoric_list(pred_items), LEVEL)
            _add_pair(gv, pv, acc)
            continue

        matched, unmatched_gold, leftover_pred = _match_sentence_groups(gold_items, pred_items, mode)
        for gold, preds, _mtype in (matched or []):
            gvec = _multi_hot(gold.get("rhetoricList") or [], LEVEL)
            for pred in preds:
                pvec = _multi_hot(pred.get("rhetoricList") or [], LEVEL)
                _add_pair(gvec, pvec, acc)

        for gold in (unmatched_gold or []):
            for lab, one in zip(RHETORIC_LABELS, _multi_hot(gold.get("rhetoricList") or [], LEVEL)):
                if one:
                    acc[lab]["fn"] += 1

        for pred in (leftover_pred or []):
            for lab, one in zip(RHETORIC_LABELS, _multi_hot(pred.get("rhetoricList") or [], LEVEL)):
                if one:
                    acc[lab]["fp"] += 1


def _add_pair(gvec: np.ndarray, pvec: np.ndarray, acc: Dict[str, Dict[str, int]]) -> None:
    """一对 multi-hot 向量的类型级 TP/FP/FN 累加。"""
    for i, lab in enumerate(RHETORIC_LABELS):
        g, p = int(gvec[i]), int(pvec[i])
        if g and p:
            acc[lab]["tp"] += 1
        elif p and not g:
            acc[lab]["fp"] += 1
        elif g and not p:
            acc[lab]["fn"] += 1


def load_records(pred_path: str) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]]:
    """解析一条预测文件 → [(gold_merged, pred_merged, raw_record)]，与 evaluate_all 同预处理。"""
    with open(pred_path, encoding="utf-8") as fh:
        records = json.load(fh)
    out = []
    for rec in records:
        raw = rec.get("rhetoricItems_pred")
        parsed = parse_prediction(raw) if isinstance(raw, str) else None
        pred_items = _extract_rhetoric_items(parsed) if parsed is not None else []
        gold_items = rec.get("rhetoricItems") or []
        if not isinstance(gold_items, list):
            gold_items = []
        out.append((merge_rhetoric_items(gold_items), merge_rhetoric_items(pred_items), rec))
    return out


def gold_n_per_type(gold_items: List[Dict[str, Any]]) -> Dict[str, int]:
    """一篇文档里每个粗粒度类型出现的 gold 句组数（merge 后）。"""
    n: Dict[str, int] = {lab: 0 for lab in RHETORIC_LABELS}
    for it in gold_items:
        vec = _multi_hot(it.get("rhetoricList") or [], LEVEL)
        for lab, one in zip(RHETORIC_LABELS, vec):
            if one:
                n[lab] += 1
    return n


def finalise(acc: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
    """TP/FP/FN → precision/recall/f1/n。"""
    out: Dict[str, Dict[str, Any]] = {}
    for lab in RHETORIC_LABELS:
        c = acc[lab]
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lab] = {
            "n_gold_cells": tp + fn,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
        }
    return out


def reconcile(doc_pairs: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
              metrics_json: Dict[str, Any], label: str) -> Dict[str, Any]:
    """对账：重放的"文档均值 F1"必须等于 metrics JSON 存的标量。"""
    report: Dict[str, Any] = {}
    for mode in MODES:
        vals = [compute_classification(g, p, mode, LEVEL)["f1"] for g, p in doc_pairs]
        replayed = float(np.mean(vals)) * 100 if vals else 0.0
        stored = metrics_json["results"][mode]["classification"][LEVEL]["f1"] * 100
        delta = abs(replayed - stored)
        if delta > 0.05:
            raise SystemExit(f"对账失败：{label} mode={mode} 重放 {replayed:.2f} vs 存量 {stored:.2f}")
        report[mode] = {"replayed_doc_mean_f1": round(replayed, 2),
                        "stored_f1": round(stored, 2),
                        "abs_delta": round(delta, 6)}
    return report


def build_figure(agg: Dict[str, Dict[str, Any]], n_by_mode: Dict[str, Dict[str, int]],
                 out_dir: str) -> str:


    labels = [RHET_EN.get(t, t) for t in RHETORIC_LABELS]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, mode, title in zip(axes, ("free", "hard"),
                               ("Free mode (flattened)", "Hard mode (exact sentence group)")):
        ns = [n_by_mode[mode][t] for t in RHETORIC_LABELS]
        mean_f1 = [agg[mode][t]["mean_f1"] for t in RHETORIC_LABELS]
        sd = [agg[mode][t]["sd_f1"] for t in RHETORIC_LABELS]
        x = np.arange(len(labels))
        ax.bar(x, mean_f1, yerr=sd, capsize=2, color="#4C72B0",
               error_kw={"elinewidth": 0.9, "capthick": 0.9,
                         "ecolor": "#7F7F7F", "alpha": 0.8})
        for xi, v, s, n in zip(x, mean_f1, sd, ns):
            ax.text(xi, v + s + 1.5, f"n={n}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel(r"per-type $F_1$ (%)", fontsize=8.5)
    fig.suptitle("Per-coarse-type $F_1$, averaged over the 11 configurations (bars) "
                 "with across-configuration s.d. (whiskers)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, "fig_pertype_f1.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def tex_rows(table: Dict[str, Dict[str, Any]], n_row: Dict[str, int], mode: str) -> str:


    keys = [k for k, _ in DISPLAY_ROWS if k in table]
    disp_of = dict(DISPLAY_ROWS)


    head = (r"    \textbf{Type} & \textbf{$n$} & "
            + " & ".join(r"\rotatebox[origin=c]{90}{" + disp_of[k] + "}"
                          for k in keys) + r" \\")
    lines = [head, r"    \midrule"]
    for t in RHETORIC_LABELS:
        vals = [table[k][t]["f1"] for k in keys]
        mx = max(vals)
        cells = []
        for v in vals:
            s = f"{v:.2f}"
            if abs(v - mx) < 1e-9:
                s = r"\best{" + s + "}"
            cells.append(s)
        lines.append(f"    {RHET_EN.get(t, t)} & {n_row[t]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-type F1 breakdown (Appendix D)")
    ap.add_argument("--predictions-dir", default=os.path.join(_PROJECT_ROOT, "results", "exp01", "predictions"))
    ap.add_argument("--metrics-dir", default=os.path.join(_PROJECT_ROOT, "results", "exp01", "metrics"))
    ap.add_argument("--output-dir", default=os.path.join(_PROJECT_ROOT, "results", "per_type_f1"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(args.predictions_dir, "predictions_*.json")))
    if not pred_files:
        raise SystemExit(f"未找到预测文件：{args.predictions_dir}")
    metas = dedupe_labels([parse_model_meta(p) for p in pred_files])

    table: Dict[str, Dict[str, Dict[str, Any]]] = {}
    per_config: Dict[str, Any] = {}
    gold_ns: List[Dict[str, int]] = []
    reconcile_all: Dict[str, Any] = {}

    for path, meta in zip(pred_files, metas):
        rows = load_records(path)
        pairs = [(g, p) for g, p, _ in rows]
        row_key = row_key_of(meta)
        if row_key not in {k for k, _ in DISPLAY_ROWS}:
            raise SystemExit(f"未在 DISPLAY_ROWS 中登记该配置：{row_key}\n"
                             f"已登记：{sorted(k for k, _ in DISPLAY_ROWS)}")
        per_config[meta["label"]] = {}
        for mode in MODES:
            acc = {lab: {"tp": 0, "fp": 0, "fn": 0} for lab in RHETORIC_LABELS}
            accumulate(pairs, mode, acc)
            fin = finalise(acc)
            table.setdefault(mode, {})[row_key] = fin
            per_config[meta["label"]][mode] = fin
        gold_ns.append({lab: 0 for lab in RHETORIC_LABELS})

        for _g, _p, rec in rows:
            for lab, cnt in gold_n_per_type(_g).items():
                gold_ns[-1][lab] += cnt

        mpath = os.path.join(args.metrics_dir,
                             os.path.basename(path).replace("predictions_", "metrics_"))
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as fh:
                reconcile_all[meta["label"]] = reconcile(pairs, json.load(fh), meta["label"])

    if any(g != gold_ns[0] for g in gold_ns):
        raise SystemExit("各配置的 gold 类型计数不一致，请检查数据源")
    gold_n = gold_ns[0]

    agg: Dict[str, Dict[str, Any]] = {}
    for mode in MODES:
        agg[mode] = {}
        for lab in RHETORIC_LABELS:
            vals = [table[mode][k][lab]["f1"] for k in table[mode]]
            agg[mode][lab] = {
                "mean_f1": round(float(np.mean(vals)), 2),
                "sd_f1": round(float(np.std(vals)), 2),
                "min_f1": round(float(np.min(vals)), 2),
                "max_f1": round(float(np.max(vals)), 2),
            }


    any_key = next(iter(table["hard"]))
    n_by_mode: Dict[str, Dict[str, int]] = {
        mode: {lab: table[mode][any_key][lab]["n_gold_cells"] for lab in RHETORIC_LABELS}
        for mode in MODES}

    fig = build_figure(agg, n_by_mode, args.output_dir)
    with open(os.path.join(args.output_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "meta": {"modes": list(MODES), "level": LEVEL,
                     "aggregation": "pooled one-vs-rest TP/FP/FN over all 511 documents "
                                    "and all matched/unmatched groups; NOT the document-mean "
                                    "used by Table 2",
                     "n_denominator": "hard: gold sentence groups carrying the type; "
                                      "free: documents carrying the type after flattening; "
                                      "identical across configurations",
                     "n_predictions_files": len(pred_files)},
            "gold_group_type_cells": gold_n,
            "gold_cells_by_mode": n_by_mode,
            "per_configuration": per_config,
            "aggregate_by_type": agg,
            "reconciliation_document_mean": reconcile_all,
        }, fh, ensure_ascii=False, indent=2)
    for mode in MODES:
        with open(os.path.join(args.output_dir, f"tab_pertype_{mode}.tex"), "w", encoding="utf-8") as fh:
            fh.write(tex_rows(table[mode], n_by_mode[mode], mode) + "\n")
    write_markdown(args.output_dir, table, agg, n_by_mode, reconcile_all, fig)

    print("\n===== per-type F1 摘要 =====")
    for mode in MODES:
        print(f"\n{mode} 模式（n = 含该类型的 gold "
              f"{'句组' if mode == 'hard' else '文档'}数）按类型均值 F1：")
        for lab in RHETORIC_LABELS:
            a = agg[mode][lab]
            print(f"  {RHET_EN.get(lab, lab):<15} n={n_by_mode[mode][lab]:<5} "
                  f"mean={a['mean_f1']:6.2f} sd={a['sd_f1']:5.2f} "
                  f"range=[{a['min_f1']:.2f}, {a['max_f1']:.2f}]")
    print(f"\n对账最大绝对偏差："
          f"{max(v['abs_delta'] for d in reconcile_all.values() for v in d.values()):.6f} pt")
    print(f"输出目录：{args.output_dir}")


def write_markdown(out_dir: str, table: Dict[str, Dict[str, Dict[str, Any]]],
                   agg: Dict[str, Dict[str, Any]], n_by_mode: Dict[str, Dict[str, int]],
                   reconcile_all: Dict[str, Any], fig: str) -> None:
    L: List[str] = ["# Appendix D 补充：粗粒度修辞类型的 per-type F1", ""]
    L.append("> 口径：**pooled one-vs-rest**——把每个类型自己的 TP/FP/FN 跨 511 篇文档、")
    L.append("> 跨所有已匹配/未匹配句组累加后再算 P/R/F1。这与正文 Table 2 的")
    L.append("> \"逐文档算分再对文档取均值\"**不是同一个聚合方式**（后者不保存类型维度，")
    L.append("> 故 per-type 无法从它反解）；两者只应趋势一致，不应逐格相等。")
    L.append("")
    L.append(f"![per-type F1]({os.path.basename(fig)})")
    L.append("")
    for mode in MODES:
        L.append(f"## {mode} 模式")
        L.append("")
        head = " | ".join(f"{RHET_EN.get(t, t)} ({n_by_mode[mode][t]})" for t in RHETORIC_LABELS)
        L.append(f"| Model | {head} |")
        L.append("|---" * (len(RHETORIC_LABELS) + 1) + "|")
        for key, disp in DISPLAY_ROWS:
            if key not in table[mode]:
                continue
            cells = " | ".join(f"{table[mode][key][t]['f1']:.2f}" for t in RHETORIC_LABELS)
            L.append(f"| {disp.replace(chr(92) + 'model{', '').rstrip('}')} | {cells} |")
        means = " | ".join(f"**{agg[mode][t]['mean_f1']:.2f}**" for t in RHETORIC_LABELS)
        L.append(f"| *mean (sd)* | {means} |")
        L.append("")
    L.append("## 对账（重放文档均值 vs metrics JSON 存量）")
    L.append("")
    L.append("| Model | hard 重放 | hard 存量 | free 重放 | free 存量 |")
    L.append("|---|---|---|---|---|")
    for label, rep in reconcile_all.items():
        L.append(f"| {label} | {rep['hard']['replayed_doc_mean_f1']:.2f} | "
                 f"{rep['hard']['stored_f1']:.2f} | {rep['free']['replayed_doc_mean_f1']:.2f} | "
                 f"{rep['free']['stored_f1']:.2f} |")
    L.append("")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
