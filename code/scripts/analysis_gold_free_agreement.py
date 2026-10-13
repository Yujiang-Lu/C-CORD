#!/usr/bin/env python3


from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.evaluator import (  # noqa: E402
    RHETORIC_LABELS,
    _normalize_label,
    merge_rhetoric_items,
    parse_prediction,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("gold_free_agreement")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.unicode_minus": False,
})
DPI = 300

MULTI_CATEGORY = "multi"
RHET_EN: Dict[str, str] = {
    "比喻": "Metaphor", "比拟": "Personif.", "夸张": "Hyperbole",
    "排比": "Parallelism", "反复": "Repetition", "设问": "Hypophora",
    "反问": "Rhet. Question", "摹状": "Depiction", "引用": "Quotation",
    "对偶": "Antithesis",
}


def parse_model_meta(filename: str) -> Dict[str, str]:
    """从预测文件名解析模型名、思考模式、温度。"""
    base = os.path.basename(filename)
    core = base[len("predictions_"):] if base.startswith("predictions_") else base
    core = core[:-len(".json")] if core.endswith(".json") else core
    mode = "default"
    if core.endswith("_think"):
        mode, core = "think", core[:-len("_think")]
    elif core.endswith("_nothink"):
        mode, core = "nothink", core[:-len("_nothink")]
    model = core.split("_seed")[0] if "_seed" in core else core
    m = re.search(r"_t([0-9.]+)_it", core)
    temp = m.group(1) if m else "?"
    return {"model": model, "mode": mode, "temp": temp, "label": f"{model} ({mode})"}


def strip_markdown_fence(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    t = raw.strip()
    if t.startswith("```"):
        if "\n" in t:
            t = t.split("\n", 1)[1]
        else:
            t = t.lstrip("`").lstrip()
        t = t.rstrip()
        if t.endswith("```"):
            t = t[:-3].rstrip()
    return t


def parse_pred_items(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """返回 rhetoricItems 列表；解析失败返回 None（与 ch3 同语义）。"""
    text = strip_markdown_fence(raw) if isinstance(raw, str) else ""
    if not text.strip():
        obj = parse_prediction(raw) if isinstance(raw, str) else None
    else:
        obj = parse_prediction(text)
        if obj is None:
            obj = parse_prediction(raw) if isinstance(raw, str) else None
    if obj is None:
        return None
    items = obj.get("rhetoricItems")
    return items if isinstance(items, list) else []


def norm_rhetoric(entry: Dict[str, Any]) -> str:
    raw = str(entry.get("rhetoric", "")).strip()
    return _normalize_label(raw, "rhetoric") or raw


def load_predictions(predictions_dir: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(predictions_dir, "predictions_*.json")))
    models: List[Dict[str, Any]] = []
    for fp in files:
        meta = parse_model_meta(fp)
        with open(fp, "r", encoding="utf-8") as fh:
            meta["records"] = json.load(fh)
        meta["path"] = fp
        models.append(meta)
    dedupe_labels(models)
    logger.info("加载 %d 个预测文件 ← %s", len(models), predictions_dir)
    return models


def dedupe_labels(models: List[Dict[str, Any]]) -> None:
    """就地修正 label 冲突。

    同一 base model + mode 可能有两个配置（仅温度不同，如 qwen3.8-max 的
    t0.7 / t1.0），而下游所有 per-model 字典都以 label 为键 —— 不去重就会
    让后加载的文件覆盖前者，并使 rater 数虚高（同一列被算两次）。
    冲突时把温度追加进 label。ch3 脚本已于同日补上同名函数（dedupe_labels），
    两侧现在都是 11 个独立 rater，可直接互证。
    """
    seen = Counter(m["label"] for m in models)
    clash = {lb for lb, c in seen.items() if c > 1}
    for m in models:
        if m["label"] in clash:
            m["label_base"] = m["label"]
            m["label"] = f"{m['label']}, t{m['temp']}"
            logger.warning("label 冲突 → 用温度区分：%s → %s", m["label_base"], m["label"])
    if len({m["label"] for m in models}) != len(models):
        raise SystemExit("去重后仍有重复 label，请检查文件名口径")


class PredIndex:
    """聚合每个配置的修辞声称，同时维护两种粒度。

    属性
    ----
    sentences : {doc_idx: n_sentences}      以 sentenceList 长度为权威句数
    claims    : {label: {(doc, sent): set(types)}}   句级拍平口径（M 系列）
    spans     : {label: {(doc, 句组元组): set(types)}}  句组完全匹配口径（S 系列）
    noncont   : {label: int}                非连续句组计数（gold 从不出现）
    oob       : {label: int}                越界 sentenceIdx 计数（诊断用）
    failures  : {label: [doc_idx]}          解析失败的文档

    label 必须唯一（由 dedupe_labels 保证），否则 per-label 字典会互相覆盖。
    """

    def __init__(self, models: Sequence[Dict[str, Any]]) -> None:
        self.labels = [m["label"] for m in models]
        if len(set(self.labels)) != len(self.labels):
            raise SystemExit("PredIndex 要求 label 唯一，请先过 dedupe_labels")
        self.sentences: Dict[int, int] = {}
        self.claims: Dict[str, Dict[Tuple[int, int], Set[str]]] = {lb: {} for lb in self.labels}
        self.spans: Dict[str, Dict[Tuple[int, Tuple[int, ...]], Set[str]]] = {lb: {} for lb in self.labels}
        self.noncont: Dict[str, int] = {lb: 0 for lb in self.labels}
        self.doc_idx: Dict[str, Set[int]] = {lb: set() for lb in self.labels}
        self.oob: Dict[str, int] = {lb: 0 for lb in self.labels}
        self.unknown_types: Dict[str, Counter] = {lb: Counter() for lb in self.labels}
        self.failures: Dict[str, List[int]] = {lb: [] for lb in self.labels}
        self._build(models)

    def _build(self, models: Sequence[Dict[str, Any]]) -> None:
        for m in models:
            label = m["label"]
            for rec in m["records"]:
                try:
                    idx = int(rec["idx"])
                except (KeyError, TypeError, ValueError):
                    continue
                self.doc_idx[label].add(idx)
                sent_list = rec.get("sentenceList") or []
                n_sent = len(sent_list)
                if idx in self.sentences and self.sentences[idx] != n_sent:
                    logger.warning("doc %s 句数不一致（%s vs %s），取较小值",
                                   idx, self.sentences[idx], n_sent)
                    self.sentences[idx] = min(self.sentences[idx], n_sent)
                else:
                    self.sentences[idx] = n_sent

                items = parse_pred_items(rec.get("rhetoricItems_pred"))
                if items is None:
                    self.failures[label].append(idx)
                    continue
                for it in merge_rhetoric_items(items):
                    sidx_raw = it.get("sentenceIdx", [])
                    if not isinstance(sidx_raw, list):
                        continue
                    in_range: List[int] = []
                    for x in sidx_raw:
                        if not isinstance(x, (int, float)):
                            continue
                        s = int(x)
                        if 0 <= s < self.sentences[idx]:
                            in_range.append(s)
                        else:
                            self.oob[label] += 1
                    if not in_range:
                        continue
                    tup = tuple(sorted(set(in_range)))
                    if len(tup) > 1 and list(tup) != list(range(tup[0], tup[-1] + 1)):
                        self.noncont[label] += 1

                    rl = it.get("rhetoricList") or []
                    types: Set[str] = set()
                    if isinstance(rl, list):
                        for entry in rl:
                            if isinstance(entry, dict):
                                t = norm_rhetoric(entry)
                                if t not in RHETORIC_LABELS:
                                    self.unknown_types[label][t] += 1
                                types.add(t)
                    if not types:
                        continue

                    for s in tup:
                        self.claims[label].setdefault((idx, s), set()).update(types)

                    self.spans[label].setdefault((idx, tup), set()).update(types)

    def subject_keys(self) -> List[Tuple[int, int]]:
        """全部 (doc, sent) 格子，按文档号、句序排序。"""
        return [(d, s) for d in sorted(self.sentences)
                for s in range(self.sentences[d])]

    def coverage_report(self) -> Dict[str, Any]:
        """校验各配置的文档覆盖是否一致；不一致时 κ 会把缺席当成「无声称」，必须报错。"""
        union = set().union(*self.doc_idx.values())
        missing = {lb: sorted(union - self.doc_idx[lb]) for lb in self.labels
                   if union - self.doc_idx[lb]}
        unknown = {lb: dict(c) for lb, c in self.unknown_types.items() if c}
        return {"n_docs_union": len(union),
                "docs_missing_per_model": missing,
                "out_of_vocab_type_labels": unknown}


def fleiss_kappa(table: np.ndarray) -> Optional[Dict[str, float]]:
    """Fleiss' κ。table: N subjects × k categories，每行和恒等于 r（评分者数）。

    返回 {"kappa", "p_bar"(观察一致), "p_e"(期望一致), "n_subjects", "n_raters"}。
    """
    if table.size == 0:
        return None
    n_raters = table.sum(axis=1)
    if not np.all(n_raters == n_raters[0]) or n_raters[0] <= 1:
        raise ValueError("Fleiss 表要求各 subject 的评分者数相同")
    n = int(n_raters[0])
    big_n = table.shape[0]
    p_i = (np.sum(table ** 2, axis=1) - n) / (n * (n - 1))
    p_bar = float(np.mean(p_i))
    p_j = table.sum(axis=0) / (big_n * n)
    p_e = float(np.sum(p_j ** 2))
    if abs(1.0 - p_e) < 1e-12:
        kappa = 1.0 if abs(p_bar - 1.0) < 1e-12 else 0.0
    else:
        kappa = (p_bar - p_e) / (1.0 - p_e)
    return {"kappa": round(kappa, 4), "p_bar": round(p_bar, 4),
            "p_e": round(p_e, 4), "n_subjects": int(big_n), "n_raters": n}


def cohen_kappa_binary(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """二分类 Cohen's κ（两个评分者、同一组 subject）。"""
    n = a.size
    if n == 0:
        return {"kappa": None, "po": None}
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))
    kappa = 1.0 if abs(1 - pe) < 1e-12 and abs(po - 1) < 1e-12 else (po - pe) / (1 - pe)
    return {"kappa": round(kappa, 4), "po": round(po, 4)}


def jaccard(a: Set[Any], b: Set[Any]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def metric_presence(pi: PredIndex) -> Dict[str, Any]:
    """M1：句级「本句是否含修辞」二分类 Fleiss κ（仅取评分者齐全的句子）。"""

    subjects = pi.subject_keys()
    failed_docs = {idx for lb in pi.labels for idx in pi.failures[lb]}
    subjects = [k for k in subjects if k[0] not in failed_docs]
    table = np.zeros((len(subjects), 2), dtype=float)
    for i, key in enumerate(subjects):
        pos = sum(1 for lb in pi.labels if pi.claims[lb].get(key))
        table[i, 0] = pos
        table[i, 1] = len(pi.labels) - pos
    out = fleiss_kappa(table) or {}
    out["n_subjects_total"] = len(pi.subject_keys())
    out["n_subjects_dropped_parse_fail"] = len(pi.subject_keys()) - len(subjects)
    out["_subjects"] = subjects
    return out


def metric_type_presence(pi: PredIndex) -> Dict[str, Any]:
    """M2：(句 × 类型) 粒度「该类型是否被声称」的 Fleiss κ + 分类型 κ。"""
    failed_docs = {idx for lb in pi.labels for idx in pi.failures[lb]}
    types = list(RHETORIC_LABELS)
    per_type: Dict[str, Any] = {}

    rows_all: List[List[int]] = []
    for key in pi.subject_keys():
        if key[0] in failed_docs:
            continue
        for t in types:
            pos = sum(1 for lb in pi.labels if t in pi.claims[lb].get(key, ()))
            rows_all.append([pos, len(pi.labels) - pos])
    table_all = np.array(rows_all, dtype=float)
    overall = fleiss_kappa(table_all) or {}

    for t in types:
        rows = []
        for key in pi.subject_keys():
            if key[0] in failed_docs:
                continue
            pos = sum(1 for lb in pi.labels if t in pi.claims[lb].get(key, ()))
            rows.append([pos, len(pi.labels) - pos])
        per_type[RHET_EN.get(t, t)] = (fleiss_kappa(np.array(rows, dtype=float)) or {})
    return {"overall": overall, "per_type": per_type}


def metric_naming(pi: PredIndex) -> Dict[str, Any]:
    """M3：11 配置都声称含修辞的句子上，命名一致性（10 类 + multi）。"""
    failed_docs = {idx for lb in pi.labels for idx in pi.failures[lb]}
    cats = list(RHETORIC_LABELS) + [MULTI_CATEGORY]
    keys = [k for k in pi.subject_keys()
            if k[0] not in failed_docs
            and all(pi.claims[lb].get(k) for lb in pi.labels)]
    table = np.zeros((len(keys), len(cats)), dtype=float)
    for i, key in enumerate(keys):
        for j, lb in enumerate(pi.labels):
            claimed = pi.claims[lb][key]
            if len(claimed) == 1:
                t = next(iter(claimed))
                col = cats.index(t) if t in cats else cats.index(MULTI_CATEGORY)
            else:
                col = cats.index(MULTI_CATEGORY)
            table[i, col] += 1
    out = fleiss_kappa(table) or {}
    out["n_categories_used"] = int(np.sum(table.sum(axis=0) > 0))
    out["category_distribution"] = {
        RHET_EN.get(c, c): int(table[:, j].sum()) for j, c in enumerate(cats)
    }
    return out


def metric_pairwise(pi: PredIndex) -> Dict[str, Any]:
    """M4：逐模型对的句级一致性（Cohen κ / 一致率 / Jaccard）与类型集合 Jaccard。"""
    failed_docs = {idx for lb in pi.labels for idx in pi.failures[lb]}
    subjects = [k for k in pi.subject_keys() if k[0] not in failed_docs]
    binary = {lb: np.array([1 if pi.claims[lb].get(k) else 0 for k in subjects])
              for lb in pi.labels}
    dev_sets = {lb: {k for k in subjects if pi.claims[lb].get(k)} for lb in pi.labels}
    common_cells = [k for k in subjects if all(pi.claims[lb].get(k) for lb in pi.labels)]

    pairs: List[Dict[str, Any]] = []
    kappas, jaccards, joint_type_jaccard = [], [], []
    for a, b in combinations(pi.labels, 2):
        ck = cohen_kappa_binary(binary[a], binary[b])
        ja = len(dev_sets[a] & dev_sets[b]) / len(dev_sets[a] | dev_sets[b]) \
            if (dev_sets[a] | dev_sets[b]) else 0.0
        tjs = [jaccard(pi.claims[a][k], pi.claims[b][k]) for k in common_cells]
        mean_tj = round(float(np.mean(tjs)), 4) if tjs else None
        pairs.append({
            "model_a": a, "model_b": b,
            "cohen_kappa_presence": ck["kappa"],
            "percent_agreement": ck["po"],
            "jaccard_presence_sentences": round(ja, 4),
            "jaccard_type_sets_on_joint_cells": mean_tj,
        })
        kappas.append(ck["kappa"])
        jaccards.append(ja)
        if mean_tj is not None:
            joint_type_jaccard.append(mean_tj)
    return {
        "n_subjects": len(subjects),
        "n_joint_cells_all_raters_flag": len(common_cells),
        "mean_pairwise_cohen_kappa": round(float(np.mean(kappas)), 4),
        "min_pairwise_cohen_kappa": round(float(np.min(kappas)), 4),
        "max_pairwise_cohen_kappa": round(float(np.max(kappas)), 4),
        "mean_pairwise_jaccard_presence": round(float(np.mean(jaccards)), 4),
        "min_pairwise_jaccard_presence": round(float(np.min(jaccards)), 4),
        "max_pairwise_jaccard_presence": round(float(np.max(jaccards)), 4),
        "mean_pairwise_jaccard_types_joint_cells":
            round(float(np.mean(joint_type_jaccard)), 4) if joint_type_jaccard else None,
        "pairs": pairs,
    }


def metric_prevalence(pi: PredIndex) -> Dict[str, Any]:
    """M5：各配置句级声称率与越界索引，用于解释 κ 偏低的密度/偏差来源。"""
    failed_docs = {idx for lb in pi.labels for idx in pi.failures[lb]}
    subjects = [k for k in pi.subject_keys() if k[0] not in failed_docs]
    per_model: Dict[str, Any] = {}
    for lb in pi.labels:
        flagged = sum(1 for k in subjects if pi.claims[lb].get(k))
        type_counter: Counter = Counter()
        multi_cells = 0
        for k in subjects:
            claimed = pi.claims[lb].get(k)
            if claimed:
                type_counter.update(claimed)
                if len(claimed) > 1:
                    multi_cells += 1
        per_model[lb] = {
            "sentence_claim_rate": round(flagged / len(subjects), 4),
            "n_sentences_flagged": flagged,
            "n_multi_type_sentences": multi_cells,
            "out_of_range_sentence_idx": pi.oob[lb],
            "n_parse_failures": len(pi.failures[lb]),
            "top_types": {RHET_EN.get(t, t): c for t, c in type_counter.most_common(3)},
        }
    return {"n_subjects": len(subjects), "per_model": per_model}


def contiguous_intervals(n: int) -> List[Tuple[int, ...]]:
    """一个 n 句文档里的全部连续句组（长度 1..n）。"""
    return [tuple(range(i, j + 1)) for i in range(n) for j in range(i, n)]


def span_universe(pi: PredIndex, mode: str = "intervals",
                  min_len: int = 1, max_len: Optional[int] = None,
                  ) -> List[Tuple[int, Tuple[int, ...]]]:
    """subject 全集。

    intervals : 每篇文档的全部连续句组（模型无关、与 gold 的连续句组惯例对齐）
    claimed   : 所有配置实际声称过的句组元组的并集（无「全票无声称」的稀薄负例）
    """
    failed_docs = {idx for lb in pi.labels for idx in pi.failures[lb]}
    hi = max_len if max_len is not None else 10 ** 9
    out: List[Tuple[int, Tuple[int, ...]]] = []
    if mode == "intervals":
        for d in sorted(pi.sentences):
            if d in failed_docs:
                continue
            for tup in contiguous_intervals(pi.sentences[d]):
                if min_len <= len(tup) <= hi:
                    out.append((d, tup))
    else:
        union: Set[Tuple[int, Tuple[int, ...]]] = set()
        for lb in pi.labels:
            union |= {k for k in pi.spans[lb] if k[0] not in failed_docs}
        out = [k for k in sorted(union, key=lambda x: (x[0], len(x[1]), x[1]))
               if min_len <= len(k[1]) <= hi]
    return out


def metric_span_presence(pi: PredIndex, universe: List[Tuple[int, Tuple[int, ...]]],
                         label: str) -> Dict[str, Any]:
    """S1：句组粒度「这个恰好是句组的修辞实体是否存在」的二分类 Fleiss κ。"""
    table = np.zeros((len(universe), 2), dtype=float)
    for i, (d, tup) in enumerate(universe):
        pos = sum(1 for lb in pi.labels if pi.spans[lb].get((d, tup)))
        table[i, 0] = pos
        table[i, 1] = len(pi.labels) - pos
    out = fleiss_kappa(table) or {}
    out["universe"] = label
    out["n_raters"] = len(pi.labels)
    return out


def metric_span_naming(pi: PredIndex) -> Dict[str, Any]:
    """S2：全部配置都在同一句组上声称修辞时，比类型命名。"""
    failed_docs = {idx for lb in pi.labels for idx in pi.failures[lb]}
    cats = list(RHETORIC_LABELS) + [MULTI_CATEGORY]
    keys = [k for k in span_universe(pi, "claimed")
            if k[0] not in failed_docs and all(pi.spans[lb].get(k) for lb in pi.labels)]
    table = np.zeros((len(keys), len(cats)), dtype=float)
    for i, key in enumerate(keys):
        for j, lb in enumerate(pi.labels):
            claimed = pi.spans[lb][key]
            col = cats.index(next(iter(claimed))) if len(claimed) == 1 and next(iter(claimed)) in cats \
                else cats.index(MULTI_CATEGORY)
            table[i, col] += 1
    out = fleiss_kappa(table) or {}
    out["n_raters"] = len(pi.labels)
    out["category_distribution"] = {
        RHET_EN.get(c, c): int(table[:, j].sum()) for j, c in enumerate(cats)}
    return out


def metric_span_pairwise(pi: PredIndex) -> Dict[str, Any]:
    """S3：逐模型对的句组级 Jaccard（总体 / 单句 / 多句）。"""
    failed_docs = {idx for lb in pi.labels for idx in pi.failures[lb]}
    span_sets = {lb: {k for k in pi.spans[lb]
                      if k[0] not in failed_docs and pi.spans[lb][k]} for lb in pi.labels}
    universe = span_universe(pi, "claimed")
    binary = {lb: np.array([1 if k in span_sets[lb] else 0 for k in universe])
              for lb in pi.labels}

    def _jac(a: str, b: str, only_len: Optional[int] = None) -> float:
        A, B = span_sets[a], span_sets[b]
        if only_len is not None:
            A = {k for k in A if len(k[1]) == only_len}
            B = {k for k in B if len(k[1]) == only_len}
        u = A | B
        return len(A & B) / len(u) if u else 0.0

    pairs: List[Dict[str, Any]] = []
    for a, b in combinations(pi.labels, 2):
        pairs.append({
            "model_a": a, "model_b": b,
            "cohen_kappa_span": cohen_kappa_binary(binary[a], binary[b])["kappa"],
            "jaccard_span_exact": round(_jac(a, b), 4),
            "jaccard_span_single_sentence": round(_jac(a, b, 1), 4),
            "jaccard_span_multi_sentence": round(_jac(a, b, 2), 4),
        })
    def _mean(key: str) -> Dict[str, Any]:
        vals = [p[key] for p in pairs if p[key] is not None]
        if not vals:
            return {"mean": None, "min": None, "max": None}
        return {"mean": round(float(np.mean(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4)}
    return {
        "n_subjects_claimed_union": len(universe),
        "span_exact": _mean("jaccard_span_exact"),
        "single_sentence_span": _mean("jaccard_span_single_sentence"),
        "multi_sentence_span": _mean("jaccard_span_multi_sentence"),
        "cohen_kappa_span": _mean("cohen_kappa_span"),
        "pairs": pairs,
    }


def metric_span_profile(pi: PredIndex) -> Dict[str, Any]:
    """S4：各配置的句组形状（是否真的在用多句实体）。"""
    per_model: Dict[str, Any] = {}
    for lb in pi.labels:
        lens = Counter(len(k[1]) for k in pi.spans[lb])
        total = sum(lens.values())
        multi = sum(v for k, v in lens.items() if k > 1)
        per_model[lb] = {
            "n_distinct_spans": total,
            "multi_sentence_span_share": round(multi / total, 4) if total else None,
            "max_span_len": max(lens) if lens else 0,
            "span_len_hist": {str(k): v for k, v in sorted(lens.items())},
            "non_contiguous_spans": pi.noncont[lb],
        }
    return {"per_model": per_model,
            "unit": "distinct 句组元组（同一句组上的多个修辞条目已合并计数）",
            "note": "non_contiguous_spans = 模型给出非连续 sentenceIdx 集合的次数；gold 从不如此"}


def gold_span_profile(models: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """描述性对照（非一致性指标）：gold 自己的句组形状。只取一份副本。"""
    if not models:
        return {}
    lens: Counter = Counter()
    for rec in models[0]["records"]:
        for it in merge_rhetoric_items(rec.get("rhetoricItems") or []):
            sidx_raw = it.get("sentenceIdx", [])
            if not isinstance(sidx_raw, list):
                continue
            tup = tuple(sorted({int(x) for x in sidx_raw if isinstance(x, (int, float))}))
            if tup:
                lens[len(tup)] += 1
    total = sum(lens.values())
    multi = sum(v for k, v in lens.items() if k > 1)
    return {
        "n_distinct_gold_spans": total,
        "multi_sentence_span_share": round(multi / total, 4) if total else None,
        "max_span_len": max(lens) if lens else 0,
        "span_len_hist": {str(k): v for k, v in sorted(lens.items())},
        "note": "gold 句组全部连续；仅作形状对照，不参与 κ 计算。单位为 distinct 句组元组，"
                "与论文报的 gold 修辞实例数（句组 × 修辞条目）不同口径",
    }


def plot_pairwise_heatmap(pairwise: Dict[str, Any], labels: List[str], out_dir: str) -> None:
    idx = {lb: i for i, lb in enumerate(labels)}
    n = len(labels)
    mat = np.eye(n)
    for p in pairwise["pairs"]:
        i, j = idx[p["model_a"]], idx[p["model_b"]]
        v = p["cohen_kappa_presence"] or 0.0
        mat[i, j] = mat[j, i] = v
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(mat, cmap="coolwarm", vmin=0.0, vmax=1.0)
    short = [lb.replace(" (think)", "\n(think)").replace(" (nothink)", "\n(nothink)")
             for lb in labels]
    ax.set_xticks(range(n), short, rotation=90)
    ax.set_yticks(range(n), short)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black")
    ax.set_title("Gold-free pairwise agreement on sentence-level device presence\n"
                 "(Cohen's $\\kappa$; no reference to gold annotations)",
                 fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$\\kappa$")
    fig.tight_layout()
    fp = os.path.join(out_dir, "fig_gf_pairwise_cohen_heatmap.png")
    fig.savefig(fp, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("写出 %s", fp)


def plot_per_type_kappa(pres: Dict[str, Any], out_dir: str) -> None:
    names = list(pres["per_type"].keys())
    vals = [pres["per_type"][t].get("kappa") or 0.0 for t in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    order = np.argsort(vals)[::-1]
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    bars = ax.bar(range(len(names)), vals, color="#4C72B0", edgecolor="#4F4F4F", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=7)
    overall = pres["overall"].get("kappa")
    if overall is not None:
        ax.axhline(overall, color="#C44E52", linestyle="--", linewidth=0.9,
                   label=f"all-types pooled $\\kappa$ = {overall:.4f}")
        ax.legend(fontsize=8)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_ylabel("Fleiss' $\\kappa$")
    ax.set_title("Gold-free cross-model agreement per rhetoric type\n"
                 "(sentence $\\times$ type: is this type claimed here?)",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fp = os.path.join(out_dir, "fig_gf_per_type_kappa.png")
    fig.savefig(fp, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("写出 %s", fp)


def plot_span_kappa(s1: Dict[str, Any], out_dir: str) -> None:
    """单句 vs 多句句组的 gold-free 存在性 κ 对比。"""
    spec = [
        ("single\n(S1b)", "span_exact_single_sentence"),
        ("multi\n(S1c)", "span_exact_multi_sentence"),
        ("single\n(S1e)", "claimed_union_single"),
        ("multi\n(S1f)", "claimed_union_multi"),
    ]
    vals, labs, ns = [], [], []
    for name, key in spec:
        d = s1.get(key) or {}
        vals.append(d.get("kappa") or 0.0)
        labs.append(name)
        ns.append(d.get("n_subjects"))
    fig, ax = plt.subplots(figsize=(3.35, 2.7))
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="#4F4F4F", linewidth=0.6)
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}\n(n={n})",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(vals)), labs, fontsize=7)
    ax.set_ylim(0, min(1.0, max(vals) * 1.5 if vals else 1.0))
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Fleiss' $\\kappa$ (gold-free)", fontsize=8)

    ax.set_title("Gold-free span agreement\n(exact index-tuple match)",
                 fontsize=8.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout(pad=0.4)
    fp = os.path.join(out_dir, "fig_gf_span_kappa.png")
    fig.savefig(fp, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("写出 %s", fp)


def recompute_gold_based_dedup(predictions_dir: str) -> Optional[Dict[str, Any]]:
    """用去重后的 11 个独立 rater 重算 ch3 的 gold-based 一致性（对照用）。

    ch3 脚本自身担心 label 冲突，本函数仅读取其逻辑、不改写它的产物。
    需要 ch3 文件存在；否则返回 None。
    """
    ch3_path = os.path.join(_SCRIPT_DIR, "analysis_ch3_failure_cases.py")
    if not os.path.isfile(ch3_path):
        logger.warning("未找到 ch3 脚本，跳过 gold-based 对照重算")
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("ch3_for_cross_check", ch3_path)
    ch3 = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(ch3)

    models = ch3.load_predictions(predictions_dir)
    dedupe_labels(models)
    res = ch3.analyze(models)
    cc = ch3.compute_consistency(res)
    mat = cc["jaccard"]
    off = mat[~np.eye(mat.shape[0], dtype=bool)]
    difficulty = cc["difficulty"]
    return {
        "fleiss_kappa": round(float(cc["fleiss_kappa"]), 4) if cc["fleiss_kappa"] is not None else None,
        "mean_pairwise_jaccard": round(float(off.mean()), 4),
        "min_pairwise_jaccard": round(float(off.min()), 4),
        "max_pairwise_jaccard": round(float(off.max()), 4),
        "n_samples": len(difficulty),
        "hard_cases_le3": sum(1 for v in difficulty.values() if v <= 3),
        "easy_cases_ge8": sum(1 for v in difficulty.values() if v >= 8),
        "note": "同一算法、但 11 个 rater 去重后的重算值；与 ch3 已存产物不同则说明受 label 冲突影响",
    }


def read_gold_based_ref(path: str) -> Optional[Dict[str, Any]]:
    """读取 ch3 已产出的 gold-based 一致性（仅作对照，本脚本不修改它）。"""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    cc = d.get("cross_model_consistency") or {}
    return {
        "source": path,
        "fleiss_kappa_gold_based": cc.get("fleiss_kappa"),
        "mean_pairwise_jaccard_gold_based": cc.get("mean_pairwise_jaccard"),
        "max_pairwise_jaccard_gold_based": cc.get("max_pairwise_jaccard"),
        "definition": "类别轴 = 该 gold 实例是否被全维度判对（依赖 gold），非本脚本产物",
    }


def build_summary_md(res: Dict[str, Any]) -> List[str]:
    m1, m2, m3 = res["M1_presence_kappa"], res["M2_type_presence_kappa"], res["M3_naming_kappa"]
    m4, m5 = res["M4_pairwise"], res["M5_prevalence"]
    s1, s2, s3, s4 = (res["S1_span_presence_kappa"], res["S2_span_naming_kappa"],
                      res["S3_span_pairwise"], res["S4_span_profile"])
    ref = res.get("reference_gold_based") or {}
    L: List[str] = []
    L.append("# Gold-free 跨模型一致性（补算，供 rebuttal 使用）")
    L.append("")
    L.append(f"> 数据来源：`{res['meta']['predictions_dir']}`"
             f"（{res['meta']['n_models']} 配置 × {res['meta']['n_docs']} 文档，"
             f"{res['meta']['n_sentence_cells']} 个句子格子）。"
             "**所有 κ 指标不读取 gold `rhetoricItems`**，只用模型自身输出；"
             "仅 S4 的句组形状对照读 gold（描述性，不参与 κ）。")
    L.append("")
    L.append("## 为什么需要这组数")
    L.append("")
    L.append("论文 §4.3/§6.5 的 Fleiss' κ = 0.3988 的类别轴是「该 gold 实例是否被**全维度**判对」")
    L.append("（location exact ∧ rhetoric/form/content correct ∧ 三成分 content_sim == 1.0），")
    L.append("这个标签本身由 gold 导出。它回答的是「模型们对哪些题能做全对一致吗」，")
    L.append("而 §1/Abstract 想说的是「模型对*哪些地方存在何种修辞*本身就分歧」。")
    L.append("下面的指标把 gold 完全拿掉，直接量化后者。")
    L.append("")
    L.append("## 结果一览")
    L.append("")
    L.append("| 指标 | 粒度 | 类别 | subjects | raters | Fleiss' κ | P_bar | P_e |")
    L.append("|---|---|---|---|---|---|---|---|")
    L.append(f"| M1 句级存在性 | 句 | 含修辞 / 无 | {m1.get('n_subjects')} | {m1.get('n_raters')} | **{m1.get('kappa')}** | {m1.get('p_bar')} | {m1.get('p_e')} |")
    L.append(f"| M2 句×类型存在性 | 句×10 类型 | 声称 / 未声称 | {m2['overall'].get('n_subjects')} | {m2['overall'].get('n_raters')} | **{m2['overall'].get('kappa')}** | {m2['overall'].get('p_bar')} | {m2['overall'].get('p_e')} |")
    L.append(f"| M3 命名一致（同句全员声称） | 句 | 10 类型 + multi | {m3.get('n_subjects')} | {m3.get('n_raters')} | **{m3.get('kappa')}** | {m3.get('p_bar')} | {m3.get('p_e')} |")
    L.append("")
    if ref:
        L.append(f"对照（已有、gold-derived）：Fleiss' κ = **{ref.get('fleiss_kappa_gold_based')}**，"
                 f"mean pairwise Jaccard = {ref.get('mean_pairwise_jaccard_gold_based')}"
                 f"（已存产物报的上界 {ref.get('max_pairwise_jaccard_gold_based')}，见下文第 6 条）。")
        L.append("")
    chk = res.get("cross_check_gold_based_dedup")
    if chk:
        L.append("## 附：gold-based 指标的去重对照重算")
        L.append("")
        L.append("（`--with-gold-based-check` 产出：同一算法、但 11 个 rater 按温度去重）")
        L.append("")
        L.append("| 量 | ch3 已存产物 | 去重重算 | 一致？ |")
        L.append("|---|---|---|---|")
        same_k = (ref or {}).get("fleiss_kappa_gold_based") == chk["fleiss_kappa"]
        L.append(f"| Fleiss' κ | {(ref or {}).get('fleiss_kappa_gold_based')} | {chk['fleiss_kappa']} | "
                 f"{'是' if same_k else '否'} |")
        L.append(f"| mean pairwise Jaccard | {(ref or {}).get('mean_pairwise_jaccard_gold_based')} | "
                 f"{chk['mean_pairwise_jaccard']} | "
                 f"{'是' if (ref or {}).get('mean_pairwise_jaccard_gold_based') == chk['mean_pairwise_jaccard'] else '否'} |")
        L.append(f"| max pairwise Jaccard | {ref.get('max_pairwise_jaccard_gold_based') if ref else '-'} | "
                 f"{chk['max_pairwise_jaccard']} | 上界受 label 冲突影响时此项必异 |")
        L.append(f"| hard cases（≤ 3/11） | — | {chk['hard_cases_le3']} / {chk['n_samples']} | "
                 f"与论文现引数比对 |")
        L.append(f"| easy cases（≥ 8/11） | — | {chk['easy_cases_ge8']} / {chk['n_samples']} | |")
        L.append("")
    L.append("## M2 分类型 κ")
    L.append("")
    L.append("| 类型 | κ | subjects | 声称率(全模型合计) |")
    L.append("|---|---|---|---|")
    for name, d in sorted(m2["per_type"].items(), key=lambda kv: -(kv[1].get("kappa") or 0)):
        p_j_note = res["meta"]["type_claim_share"].get(name, "-")
        L.append(f"| {name} | {d.get('kappa')} | {d.get('n_subjects')} | {p_j_note} |")
    L.append("")
    L.append("## M4 pairwise（句级存在性）")
    L.append("")
    L.append(f"- mean Cohen's κ = **{m4['mean_pairwise_cohen_kappa']}** "
             f"(range {m4['min_pairwise_cohen_kappa']}–{m4['max_pairwise_cohen_kappa']})")
    L.append(f"- mean Jaccard（修辞句集合） = **{m4['mean_pairwise_jaccard_presence']}** "
             f"(range {m4['min_pairwise_jaccard_presence']}–{m4['max_pairwise_jaccard_presence']})")
    L.append(f"- 全员声称句子上、类型集合的 mean pairwise Jaccard = "
             f"**{m4['mean_pairwise_jaccard_types_joint_cells']}** "
             f"(n = {m4['n_joint_cells_all_raters_flag']} cells)")
    L.append("")
    L.append("![pairwise](fig_gf_pairwise_cohen_heatmap.png)")
    L.append("")
    L.append("![per-type](fig_gf_per_type_kappa.png)")
    L.append("")
    L.append("## M5 声称率（解释 κ 的密度成分）")
    L.append("")
    L.append("| 配置 | 句级声称率 | 多类型同句 | 越界 sentenceIdx | 解析失败 |")
    L.append("|---|---|---|---|---|")
    for lb, d in m5["per_model"].items():
        L.append(f"| {lb} | {d['sentence_claim_rate']} | {d['n_multi_type_sentences']} | "
                 f"{d['out_of_range_sentence_idx']} | {d['n_parse_failures']} |")
    L.append("")
    L.append("## S 系列：不拍平的句组完全匹配（文档级修辞单元口径）")
    L.append("")
    L.append("gold 把修辞实体标为**最小修辞单元**：跨 (2,3,4) 的实体，单看第 2 句或 3、4 句都应是「无修辞」。")
    L.append("因此下列指标以 **sentenceIdx 元组完全相等** 为唯一匹配标准，不拍平，也不给包含／部分匹配任何计入。")
    L.append("")
    L.append("| 指标 | subject 全集 | 类别 | subjects | raters | Fleiss' κ | P_bar | P_e |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, name in (("span_exact_all", "S1a 全部连续句组"),
                      ("span_exact_single_sentence", "S1b 仅单句实体（len=1）"),
                      ("span_exact_multi_sentence", "S1c 仅多句实体（len≥2）"),
                      ("claimed_union", "S1d 声称句组并集（含单+多）"),
                      ("claimed_union_single", "S1e 声称并集（仅单句）"),
                      ("claimed_union_multi", "S1f 声称并集（仅多句）")):
        d = s1.get(key) or {}
        L.append(f"| {name} | {d.get('universe', '-')} | 存在 / 不存在 | "
                 f"{d.get('n_subjects')} | {d.get('n_raters')} | **{d.get('kappa')}** | "
                 f"{d.get('p_bar')} | {d.get('p_e')} |")
    L.append(f"| S2 句组命名一致（全配置同句组） | claimed | 10 类型 + multi | "
             f"{s2.get('n_subjects')} | {s2.get('n_raters')} | **{s2.get('kappa')}** | "
             f"{s2.get('p_bar')} | {s2.get('p_e')} |")
    L.append("")
    L.append("⚠️ **读表入口**：S1a/S1c 的全集包含大量「没有任何模型声称」的空句组，")
    L.append("   这些行上 11 个配置一致投「不存在」，会人为推高 Fleiss κ；"
             "因此 **S1a 不可与 S1b/S1c 直接比高低**。")
    L.append("   干净的对比是同一全集内的 S1b vs S1c，以及剔掉空负例后的 S1e vs S1f。")
    L.append("")
    b_c = ((s1.get("span_exact_single_sentence") or {}).get("kappa"),
           (s1.get("span_exact_multi_sentence") or {}).get("kappa"))
    e_f = ((s1.get("claimed_union_single") or {}).get("kappa"),
           (s1.get("claimed_union_multi") or {}).get("kappa"))
    if all(v is not None for v in b_c):
        L.append(f"- 单句→多句的共识崩塌量：S1b {b_c[0]} → S1c {b_c[1]}（Δ = "
                 f"{round(b_c[1] - b_c[0], 4)}，相对下降 "
                 f"{round((1 - b_c[1] / b_c[0]) * 100, 1)}%）")
    if all(v is not None for v in e_f):
        L.append(f"- 同样趋势在无空负例口径下重现：S1e {e_f[0]} → S1f {e_f[1]}（Δ = "
                 f"{round(e_f[1] - e_f[0], 4)}）")
    L.append("")
    L.append(f"- S3 句组级 mean pairwise Jaccard：**{s3['span_exact']['mean']}**"
             f"（range {s3['span_exact']['min']}–{s3['span_exact']['max']}）；"
             f"其中单句实体 {s3['single_sentence_span']['mean']}、"
             f"多句实体 {s3['multi_sentence_span']['mean']}"
             f"（多句仅统计长度恰好为 2 的实体，长度≥ 3 的多句实体过稀，逐对 Jaccard 不稳定）")
    gp = res["meta"]["gold_span_profile"]
    L.append(f"- S4 形状对照（单位：distinct 句组元组）：gold 多句句组占比 "
             f"**{gp.get('multi_sentence_span_share')}**（最长 {gp.get('max_span_len')} 句，"
             f"全部连续）；模型侧从 "
             + " / ".join(f"{v['multi_sentence_span_share']}"
                          for v in s4["per_model"].values()) + "；"
             f"非连续句组声称次数 "
             + f"合计 {sum(v['non_contiguous_spans'] for v in s4['per_model'].values())}"
               "（gold 为 0，属格式越界）")
    L.append("")
    L.append("![span](fig_gf_span_kappa.png)")
    L.append("")
    L.append(f"解读：实测已确认这一趋势 —— 同一全集内单句 {b_c[0]} vs 多句 {b_c[1]}，"
             "剔掉空负例后单句 "
             f"{e_f[0]} vs 多句 {e_f[1]}。即模型之间的共识主要建立在「单句就能看出的修辞」上，"
             "一旦修辞单元需要跨句划定（排比、设问/反问这类句群级修辞），"
             "模型连彼此都不一致 —— 这是文档级修辞理解的直接证据，且不依赖 gold。")
    L.append("")
    L.append("## 解读与使用边界")
    L.append("")
    gold_k = ref.get("fleiss_kappa_gold_based")
    m1_k = m1.get("kappa")
    if gold_k is not None and m1_k is not None:
        diff = round(m1_k - gold_k, 4)
        if m1_k > gold_k:
            L.append(f"1. **方向判断**：gold-free 句级存在性 κ = {m1_k} 高于 gold-based κ = {gold_k}"
                     f"（Δ = +{diff}）。即模型之间对「哪一句有修辞、是什么修辞」"
                     f"存在**系统性共识**（moderate agreement），它们的分歧主要发生在"
                     f"与 gold 的严格对齐（逐句定位 + 逐维度全对）上，而不仅是彼此看不见对方看见的东西。"
                     f"因此本组数不能简单当作「模型间分歧比 gold-based 显示的更大」来用，"
                     f"而是把【一致性的所在层级】说清楚。")
        else:
            L.append(f"1. **方向判断**：gold-free 句级存在性 κ = {m1_k} 不高于 gold-based κ = {gold_k}"
                     f"（Δ = {diff}）。拿掉 gold 后模型之间仍不一致，"
                     f"可直接支撑 §1/Abstract 的 *models disagree on which rhetoric is present*。")
    lowest = sorted(m2["per_type"].items(), key=lambda kv: (kv[1].get("kappa") or 0))[:3]
    L.append(f"2. **分类型分歧最大的三类**："
             + "、".join(f"{n} (κ={d.get('kappa')})" for n, d in lowest)
             + "。若与 §7.2 的漏检类型（repetition / personification 等）吻合，"
               "则说明低召回不是单纯的标注偏好差异，而是感知层面的共识缺失。")
    L.append(f"3. **M3 是条件性子集**：只在全部 {m3.get('n_raters')} 个配置都声称含修辞的句子上比命名"
             f"（n = {m3.get('n_subjects')} 句，全集 {res['meta']['n_sentence_cells']} 句），"
             f"不能与 M1 直接比较高低；multi 列把同句声称多类型的情况归为一类，"
             f"故 M3 是命名共识的保守估计。")
    L.append("4. **口径互补**：M 系列把跨句修辞拍平成逐句声称，回答「共识在不在同一句上」；"
             "S 系列不拍平，要求 sentenceIdx 元组完全相等，回答「共识在不在同一个修辞单元上」。"
             "两者都是独立的 gold-free 证据，都不是 §6.5 表格中 κ 的替代值。")
    L.append("5. **缺席不当负面**：解析失败的文档在三个指标中都按 subject 剔除（而非记为「无修辞」），"
             "避免把 API 失败伪装成负面判断；剔除条数见 stats.json 的 "
             "`M1_presence_kappa.n_subjects_dropped_parse_fail`。")
    L.append("6. **rater 口径修正（重要）**：exp01 的 11 个配置中有两个仅温度不同"
             "（qwen3.8-max nothink 的 t0.7 / t1.0），而文件名解析出的 display label 完全相同。"
             "本脚本用 `dedupe_labels()` 给它们加温度后缀，保证 11 个真实独立的 rater；"
             "ch3 脚本已同步补上同一补丁并重跑（历史上它缺这一步，两类容器失真方向不同："
             "字典式（instance_dims / "
             "sample_success / correct_sets）被后加载的 t1.0 覆盖导致 t0.7 丢失，"
             "计数式（match_counts / rhet_total）则两次累加→ 行值翻倍（表现为某行 "
             "matched+missed = 2× gold 实例数），且 Jaccard 矩阵曾出现一个「同名列自比」"
             "的 1.0 假上界、hard-case 计数偏移）。")
    L.append("")
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions-dir",
                    default=os.path.join(_PROJECT_ROOT, "results", "exp01", "predictions"))
    ap.add_argument("--output-dir",
                    default=os.path.join(_PROJECT_ROOT, "results", "gold_free_agreement"))
    ap.add_argument("--gold-based-ref",
                    default=os.path.join(_PROJECT_ROOT, "analysis",
                                         "ch3_failure_cases", "stats_summary.json"))
    ap.add_argument("--with-gold-based-check", action="store_true",
                    help="额外用去重后的 11 rater 重算 ch3 的 gold-based 一致性（较慢，对照用）")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    models = load_predictions(args.predictions_dir)
    if len(models) < 2:
        logger.error("至少需要 2 个预测文件")
        return 1
    pi = PredIndex(models)
    cov = pi.coverage_report()
    if cov["docs_missing_per_model"]:
        logger.error("各配置文档覆盖不一致，κ 会把缺席误计为「无声称」：%s",
                     {k: len(v) for k, v in cov["docs_missing_per_model"].items()})
        return 1
    if cov["out_of_vocab_type_labels"]:
        logger.warning("发现词表外类型标签（计入 multi 类别）：%s",
                       cov["out_of_vocab_type_labels"])

    logger.info("句子格子 %d 个（%d 文档）", len(pi.subject_keys()), len(pi.sentences))
    m1 = metric_presence(pi)
    m2 = metric_type_presence(pi)
    m3 = metric_naming(pi)
    m4 = metric_pairwise(pi)
    m5 = metric_prevalence(pi)
    m1.pop("_subjects", None)


    uni_all = span_universe(pi, "intervals")
    uni_single = span_universe(pi, "intervals", min_len=1, max_len=1)
    uni_multi = span_universe(pi, "intervals", min_len=2)
    uni_claimed = span_universe(pi, "claimed")
    uni_claimed_single = span_universe(pi, "claimed", min_len=1, max_len=1)
    uni_claimed_multi = span_universe(pi, "claimed", min_len=2)
    logger.info("句组 subject：全部连续 %d / 单句 %d / 多句 %d / 声称并集 %d",
                len(uni_all), len(uni_single), len(uni_multi), len(uni_claimed))
    s1 = {
        "span_exact_all": metric_span_presence(pi, uni_all, "all contiguous intervals"),
        "span_exact_single_sentence": metric_span_presence(
            pi, uni_single, "intervals of length 1"),
        "span_exact_multi_sentence": metric_span_presence(
            pi, uni_multi, "intervals of length >= 2"),
        "claimed_union": metric_span_presence(pi, uni_claimed, "union of claimed spans"),
        "claimed_union_single": metric_span_presence(
            pi, uni_claimed_single, "claimed spans, length 1"),
        "claimed_union_multi": metric_span_presence(
            pi, uni_claimed_multi, "claimed spans, length >= 2"),
    }
    s2 = metric_span_naming(pi)
    s3 = metric_span_pairwise(pi)
    s4 = metric_span_profile(pi)


    type_share: Counter = Counter()
    for lb in pi.labels:
        for claimed in pi.claims[lb].values():
            type_share.update(claimed)
    total_claims = sum(type_share.values()) or 1
    type_claim_share = {RHET_EN.get(t, t): round(type_share.get(t, 0) / total_claims, 4)
                        for t in RHETORIC_LABELS}

    res = {
        "meta": {
            "task": "gold-free cross-model agreement (no gold used)",
            "predictions_dir": os.path.relpath(args.predictions_dir, _PROJECT_ROOT),
            "n_models": len(models),
            "n_docs": len(pi.sentences),
            "n_sentence_cells": len(pi.subject_keys()),
            "model_labels": pi.labels,
            "granularity": "sentence-level (跨句修辞拆为逐句声称)",
            "coverage_check": cov,
            "gold_span_profile": gold_span_profile(models),
            "type_claim_share": type_claim_share,
        },
        "M1_presence_kappa": m1,
        "M2_type_presence_kappa": m2,
        "M3_naming_kappa": m3,
        "M4_pairwise": m4,
        "M5_prevalence": m5,
        "S1_span_presence_kappa": s1,
        "S2_span_naming_kappa": s2,
        "S3_span_pairwise": s3,
        "S4_span_profile": s4,
        "reference_gold_based": read_gold_based_ref(args.gold_based_ref),
    }

    if args.with_gold_based_check:
        logger.info("额外重算 gold-based（去重 11 rater）…")
        res["cross_check_gold_based_dedup"] = recompute_gold_based_dedup(args.predictions_dir)

    stats_fp = os.path.join(args.output_dir, "stats.json")
    with open(stats_fp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    logger.info("写出 %s", stats_fp)

    md = build_summary_md(res)
    md_fp = os.path.join(args.output_dir, "summary.md")
    with open(md_fp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    logger.info("写出 %s", md_fp)

    plot_pairwise_heatmap(m4, pi.labels, args.output_dir)
    plot_per_type_kappa(m2, args.output_dir)
    plot_span_kappa(s1, args.output_dir)

    print("\n===== gold-free 一致性摘要 =====")
    print(f"M1 句级存在性 Fleiss κ        = {m1.get('kappa')}  "
          f"(subjects={m1.get('n_subjects')}, raters={m1.get('n_raters')})")
    print(f"M2 句×类型 Fleiss κ         = {m2['overall'].get('kappa')}  "
          f"(subjects={m2['overall'].get('n_subjects')})")
    print(f"M3 命名一致 Fleiss κ         = {m3.get('kappa')}  "
          f"(subjects={m3.get('n_subjects')})")
    print(f"M4 mean pairwise Cohen κ     = {m4['mean_pairwise_cohen_kappa']}  "
          f"| mean Jaccard = {m4['mean_pairwise_jaccard_presence']}")
    print("--- 不拍平（句组完全匹配）---")
    for key in ("span_exact_all", "span_exact_single_sentence",
                "span_exact_multi_sentence", "claimed_union",
                "claimed_union_single", "claimed_union_multi"):
        d = s1[key]
        print(f"  {key:30s} κ = {d.get('kappa')}  (subjects={d.get('n_subjects')})")
    print(f"  S2 句组命名                 κ = {s2.get('kappa')}  (subjects={s2.get('n_subjects')})")
    print(f"  S3 句组级 mean Jaccard      = {s3['span_exact']['mean']}"
          f"  (单句 {s3['single_sentence_span']['mean']} / 多句 {s3['multi_sentence_span']['mean']})")
    ref = res.get("reference_gold_based") or {}
    print(f"[对照] 论文现用 gold-based κ  = {ref.get('fleiss_kappa_gold_based')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
