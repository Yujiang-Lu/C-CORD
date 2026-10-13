"""
评测指标计算模块。

职责:
    - 五级容错解析模型原始字符串输出
    - 修辞条目预处理（合并相同句组、去重）
    - 句组定位 IoU（hard / soft / free 三种模式）
    - 修辞分类 multi-hot 指标（hard / soft / free 三种模式 × rhetoric / form / content 三粒度）
    - 修辞成分（conjunction / tenor / vehicle）内容相似度 + 位置 IoU（三种模式）
    - free 模式下的 Krippendorff's alpha 一致性
    - 端到端 ``evaluate_all`` 调度

三种评测模式的苛刻程度从高到低:
    - **hard**: 修辞句组定位必须与标准答案完全一致，才能参与后续分类/成分评估
    - **soft**: 修辞句组定位只要有重合部分就可以匹配，参与后续评估
    - **free**: 忽略句组边界，拍平后直接比较整篇文档的修辞细分答案
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


try:
    from Levenshtein import distance as levenshtein_distance  # type: ignore
    _HAS_LEVENSHTEIN = True
except ImportError:  # pragma: no cover
    import difflib
    _HAS_LEVENSHTEIN = False

    def levenshtein_distance(a: str, b: str) -> int:  # type: ignore
        """difflib 降级实现（基于 SequenceMatcher ratio 反推距离）。"""
        if a == b:
            return 0
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        return int(round((1.0 - ratio) * max(len(a), len(b))))


try:
    from krippendorff import alpha as _krippendorff_alpha  # type: ignore
    _HAS_KRIPPENDORFF = True
except ImportError:  # pragma: no cover
    _HAS_KRIPPENDORFF = False


RHETORIC_LABELS: Tuple[str, ...] = (
    "比喻", "比拟", "夸张", "排比", "反复",
    "设问", "反问", "摹状", "引用", "对偶",
)


FORM_LABELS: Tuple[str, ...] = (
    "明喻", "暗喻", "借喻",
    "名词", "动词", "形容词",
    "直接夸张", "间接夸张",
    "成分排比", "句子排比",
    "连续反复", "间隔反复",
    "问答连属", "问答不连属",
    "单句反问", "复句反问",
    "通感", "直感",
    "明引", "暗引", "转引",
    "正对", "反对", "串对",
)


CONTENT_LABELS: Tuple[str, ...] = (
    "实在物", "动作", "抽象概念",
    "拟人", "拟物",
    "扩大夸张", "缩小夸张", "超前夸张",
    "并列", "承接", "递进",
    "词语", "句子",
    "问而不答", "一问一答", "数问一答", "一问数答",
    "肯定反问", "否定反问",
    "摹视", "摹听", "摹嗅", "摹触", "摹味",
    "话语", "引典",
    "工整对", "宽松对",
)

_LABEL_INDEX: Dict[str, Dict[str, int]] = {
    "rhetoric": {lab: i for i, lab in enumerate(RHETORIC_LABELS)},
    "form": {lab: i for i, lab in enumerate(FORM_LABELS)},
    "content": {lab: i for i, lab in enumerate(CONTENT_LABELS)},
}


_COMPOSITE_TO_BARE: Dict[str, Dict[str, str]] = {"form": {}, "content": {}}
for _comp_lab in FORM_LABELS:
    if "_" in _comp_lab:
        _COMPOSITE_TO_BARE["form"][_comp_lab] = _comp_lab.split("_", 1)[1]
for _comp_lab in CONTENT_LABELS:
    if "_" in _comp_lab:
        _COMPOSITE_TO_BARE["content"][_comp_lab] = _comp_lab.split("_", 1)[1]


def _normalize_label(label: str, level: str) -> Optional[str]:
    """将标签归一化为 _LABEL_INDEX 中可查找的形式。

    处理逻辑：
        1. 若 label 直接存在于 _LABEL_INDEX[level]，返回原值
        2. 若 label 为复合格式（如 "比喻_明喻"），尝试提取裸标签部分查找
        3. 若 label 含 "_" 但不在索引中，尝试用 "_" 分割后的后半部分查找
        4. 均失败则返回 None
    """
    idx_map = _LABEL_INDEX.get(level)
    if idx_map is None:
        return None

    if label in idx_map:
        return label

    if level in _COMPOSITE_TO_BARE and label in _COMPOSITE_TO_BARE[level]:
        bare = _COMPOSITE_TO_BARE[level][label]
        if bare in idx_map:
            return bare

    if "_" in label:
        suffix = label.split("_", 1)[1]
        if suffix in idx_map:
            return suffix
    return None


CLASSIFICATION_LEVELS: Tuple[str, ...] = ("rhetoric", "form", "content")

_COMPONENT_KEYS: Tuple[str, ...] = ("conjunction", "tenor", "vehicle")

_EVAL_MODES: Tuple[str, ...] = ("hard", "soft", "free")


_COMPONENT_CLEAN_PATTERN = re.compile(r"（[^）]*）")


def merge_rhetoric_items(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合并 sentenceIdx 相同的修辞条目，并对合并后的 rhetoricList 去重。

        - 以 ``tuple(sentenceIdx)`` 为键分组（完全相同的句组归为一组）
        - 合并每组的 ``rhetoricList``
        - 对合并结果去重序列化整条
          修辞实体判重，可覆盖 rhetoric+form+content 三元组及成分信息）

    :param items: 原始 rhetoricItems 列表
    :return: 合并去重后的 rhetoricItems 列表
    """
    groups: Dict[Tuple[int, ...], List[Dict[str, Any]]] = {}
    order: List[Tuple[int, ...]] = []
    for item in items:
        if not isinstance(item, dict) or "sentenceIdx" not in item:

            continue
        key = tuple(item["sentenceIdx"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)

    merged: List[Dict[str, Any]] = []
    for key in order:
        group_items = groups[key]
        combined_rhetoric: List[Dict[str, Any]] = []
        for it in group_items:
            rl = it.get("rhetoricList") or []
            if isinstance(rl, list):
                combined_rhetoric.extend(rl)


        seen: set[str] = set()
        unique_rhetoric: List[Dict[str, Any]] = []
        for rhet in combined_rhetoric:
            try:
                rhet_str = json.dumps(rhet, sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError):
                rhet_str = repr(rhet)
            if rhet_str not in seen:
                seen.add(rhet_str)
                unique_rhetoric.append(rhet)

        merged.append({
            "sentenceIdx": list(key),
            "rhetoricList": unique_rhetoric,
        })
    return merged


def _extract_between_braces(text: str) -> str:
    """截取最外层 ``{...}``。"""
    left = text.find("{")
    right = text.rfind("}")
    if left == -1 or right == -1 or right <= left:
        return ""
    return text[left:right + 1]


def _strip_line_comments(text: str) -> str:
    """去掉 ``//`` 单行注释（保留字符串字面量内的 ``//``，如 URL）。"""
    out_lines: List[str] = []
    for line in text.splitlines():
        in_str = False
        quote_char = ""
        cut = -1
        i = 0
        while i < len(line) - 1:
            ch = line[i]
            if in_str:
                if ch == "\\" and i + 1 < len(line):
                    i += 2
                    continue
                if ch == quote_char:
                    in_str = False
            else:
                if ch in ('"', "'"):
                    in_str = True
                    quote_char = ch
                elif ch == "/" and line[i + 1] == "/":
                    cut = i
                    break
            i += 1
        out_lines.append(line[:cut] if cut >= 0 else line)
    return "\n".join(out_lines)


def _repair_unescaped_quotes(text: str) -> str:
    """修复 JSON 字符串值内部未转义的双引号。

    模型（如 Claude Sonnet 5.5）偶尔在文本字段中使用裸 ASCII 双引号作为中文引用
    标记，破坏 JSON 结构。本函数逐字符扫描并跟踪是否处于字符串内部：在字符串内部
    遇到 ``"`` 时，检查其后续字符（跳过空白），若为 JSON 结构字符（``:`` ``,``
    ``]`` ``}``）或文本结尾则视为合法终止符，否则判定为内部未转义引号并补转义。

    :param text: 待修复的候选 JSON 字符串
    :return: 补转义后的字符串
    """
    result: list[str] = []
    i = 0
    in_string = False

    while i < len(text):
        ch = text[i]
        if not in_string:
            result.append(ch)
            if ch == '"':
                in_string = True
            i += 1
        else:
            if ch == '\\':

                result.append(ch)
                if i + 1 < len(text):
                    i += 1
                    result.append(text[i])
                i += 1
            elif ch == '"':

                j = i + 1
                while j < len(text) and text[j] in ' \t\n\r':
                    j += 1
                if j >= len(text) or text[j] in ':,]}':

                    result.append(ch)
                    in_string = False
                else:

                    result.append('\\"')
                i += 1
            else:
                result.append(ch)
                i += 1
    return ''.join(result)


def parse_prediction(raw_str: str) -> Optional[Dict[str, Any]]:
    """五级容错解析模型原始输出字符串。

    解析顺序:
        1. ``json.loads(raw_str)``
        2. 截取最外层 ``{...}`` 再 ``json.loads``
        3. 去掉 ``//`` 单行注释后再 ``json.loads``
        4. ``ast.literal_eval``（兼容 Python 字面量风格输出）
        5. 修复字符串值内部未转义的双引号后再 ``json.loads`` /
           ``raw_decode``（应对模型输出裸 ASCII 引号作为中文引用标记）

    :param raw_str: 模型原始返回字符串
    :return: 解析后的 dict（含 ``rhetoricItems`` 键），失败时返回 None
    """
    if not isinstance(raw_str, str) or not raw_str.strip():
        return None


    try:
        obj = json.loads(raw_str)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass


    candidate = _extract_between_braces(raw_str)
    if candidate:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass


        try:
            obj = json.loads(_strip_line_comments(candidate))
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass


        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return obj
        except (ValueError, SyntaxError, RecursionError, MemoryError, TypeError):
            pass


        try:
            obj = ast.literal_eval(_strip_line_comments(candidate))
            if isinstance(obj, dict):
                return obj
        except (ValueError, SyntaxError, RecursionError, MemoryError, TypeError):
            pass


        repaired = _repair_unescaped_quotes(candidate)
        if repaired != candidate:
            try:
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    logging.debug("parse_prediction Level 5 成功（修复未转义引号）")
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass


            try:
                obj, _ = json.JSONDecoder().raw_decode(repaired.lstrip())
                if isinstance(obj, dict):
                    logging.debug("parse_prediction Level 5b 成功（修复引号 + raw_decode）")
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass

    return None


def _extract_rhetoric_items(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从解析后的 dict 中抽取 rhetoricItems 字段。"""
    items = parsed.get("rhetoricItems")
    if isinstance(items, list):
        return [it for it in items if isinstance(it, dict)]
    return []


def _sentence_idx_set(items: Sequence[Dict[str, Any]]) -> List[Tuple[int, ...]]:
    """抽取每个 rhetoricItem 的 sentenceIdx 元组列表（保持稳定顺序）。"""
    out: List[Tuple[int, ...]] = []
    for it in items:
        if "sentenceIdx" in it and isinstance(it["sentenceIdx"], list):
            out.append(tuple(it["sentenceIdx"]))
    return out


def compute_location_iou(
    pred_items: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
    mode: str = "soft",
) -> float:
    """句组定位 IoU。

    :param pred_items: 预测的 rhetoricItems list
    :param gold_items: gold 的 rhetoricItems list
    :param mode: ``"hard"`` / ``"soft"`` / ``"free"``
        - hard: pred 与 gold 的 sentenceIdx 集合完全匹配才算命中
        - soft: 对每个 gold 句组找与 pred 句组的 IoU（优先完全匹配），平均；未匹配 pred 额外惩罚 0
        - free: 把所有 sentenceIdx 拍平为集合后算 IoU
    :return: [0.0, 1.0] 浮点分数
    """
    if mode not in _EVAL_MODES:
        raise ValueError(f"mode 必须是 hard/soft/free，收到 {mode!r}")

    gold_set = set(_sentence_idx_set(gold_items))
    pred_set = set(_sentence_idx_set(pred_items))

    if mode == "hard":
        if not gold_set and not pred_set:
            return 1.0
        if not gold_set or not pred_set:
            return 0.0
        inter = gold_set & pred_set
        union = gold_set | pred_set
        return len(inter) / len(union) if union else 0.0

    if mode == "free":
        gold_flat = {i for s in gold_set for i in s}
        pred_flat = {i for s in pred_set for i in s}
        if not gold_flat and not pred_flat:
            return 1.0
        if not gold_flat or not pred_flat:
            return 0.0
        inter = gold_flat & pred_flat
        union = gold_flat | pred_flat
        return len(inter) / len(union) if union else 0.0


    if not gold_set and not pred_set:
        return 1.0
    if not gold_set:
        return 0.0
    if not pred_set:
        return 0.0

    scores: List[float] = []
    used_pred: List[Tuple[int, ...]] = []
    for g in gold_set:
        gs = set(g)
        exact = [p for p in pred_set if set(p) == gs]
        if exact:
            scores.append(1.0)
            used_pred.extend(exact)
            continue
        overlaps: List[float] = []
        for p in pred_set:
            ps = set(p)
            inter = gs & ps
            if inter:
                union = gs | ps
                overlaps.append(len(inter) / len(union))
                used_pred.append(p)
        if overlaps:
            scores.append(sum(overlaps) / len(overlaps))
        else:
            scores.append(0.0)

    left = [p for p in pred_set if p not in used_pred]
    scores.extend([0.0] * len(left))

    return sum(scores) / len(scores) if scores else 0.0


def _match_sentence_groups(
    gold_items: Sequence[Dict[str, Any]],
    pred_items: Sequence[Dict[str, Any]],
    mode: str,
) -> Tuple[
    Optional[List[Tuple[Dict[str, Any], List[Dict[str, Any]], str]]],
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
]:
    """按模式对 gold / pred 句组做匹配。

    :return: ``(matched, unmatched_gold, leftover_pred)``
        - matched: ``List[(gold_item, [pred_items], match_type)]``，match_type ∈ {"exact", "overlap"}
        - unmatched_gold: 无任何匹配的 gold 句组
        - leftover_pred: 未被任何 gold 匹配的 pred 句组（用于惩罚项）
        - free 模式返回 ``(None, None, None)``，表示"拍平模式"，调用方需自行处理
    """
    if mode == "free":
        return None, None, None
    if mode not in ("hard", "soft"):
        raise ValueError(f"mode 必须是 hard/soft/free，收到 {mode!r}")

    matched: List[Tuple[Dict[str, Any], List[Dict[str, Any]], str]] = []
    unmatched_gold: List[Dict[str, Any]] = []
    used_pred_idx: set[int] = set()

    for gold in gold_items:
        gset = set(gold.get("sentenceIdx") or [])

        exact = [
            (pi, p) for pi, p in enumerate(pred_items)
            if set(p.get("sentenceIdx") or []) == gset
        ]
        if exact:
            for pi, _ in exact:
                used_pred_idx.add(pi)
            matched.append((gold, [p for _, p in exact], "exact"))
            continue

        if mode == "soft":
            overlap = [
                (pi, p) for pi, p in enumerate(pred_items)
                if set(p.get("sentenceIdx") or []) & gset
            ]
            if overlap:
                for pi, _ in overlap:
                    used_pred_idx.add(pi)
                matched.append((gold, [p for _, p in overlap], "overlap"))
                continue
        unmatched_gold.append(gold)

    leftover_pred = [p for pi, p in enumerate(pred_items) if pi not in used_pred_idx]
    return matched, unmatched_gold, leftover_pred


def _multi_hot(
    rhetoric_list: Sequence[Dict[str, Any]],
    level: str,
) -> np.ndarray:
    """将 rhetoricList 编码为 multi-hot 向量。

    标签兼容：同时支持裸标签（gold 数据格式，如 "明喻"）和复合标签（prompt 中展示给
    模型的格式，如 "比喻_明喻"），通过 _normalize_label 统一归一化后查找索引。

    :param rhetoric_list: 修辞实体列表
    :param level: ``"rhetoric"`` / ``"form"`` / ``"content"``
    :return: shape=(size+1,) 的 0/1 numpy 数组
    """
    if level not in _LABEL_INDEX:
        raise ValueError(f"level 必须是 rhetoric/form/content，收到 {level!r}")
    idx_map = _LABEL_INDEX[level]
    size = len(idx_map)
    vec = np.zeros(size + 1, dtype=np.int8)
    if not rhetoric_list:
        vec[size] = 1
        return vec
    for r in rhetoric_list:
        if not isinstance(r, dict):
            continue
        lab = r.get(level)
        if not isinstance(lab, str):
            continue

        normalized = _normalize_label(lab, level)
        if normalized is not None and normalized in idx_map:
            vec[idx_map[normalized]] = 1
    return vec


def _flatten_rhetoric_list(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将所有句组的 rhetoricList 拍平为单个 list（free 模式使用）。"""
    out: List[Dict[str, Any]] = []
    for it in items:
        rl = it.get("rhetoricList") or []
        if isinstance(rl, list):
            out.extend([r for r in rl if isinstance(r, dict)])
    return out


def _score_multi_hot_pair(gold_vec: np.ndarray, pred_vec: np.ndarray) -> List[float]:
    """对一对 multi-hot 向量计算 [accuracy, precision, recall, f1]。

    """
    yt = np.asarray(gold_vec).reshape(1, -1)
    yp = np.asarray(pred_vec).reshape(1, -1)
    acc = float(accuracy_score(yt, yp))
    prec = float(precision_score(yt, yp, average="micro", zero_division=0))
    rec = float(recall_score(yt, yp, average="micro", zero_division=0))
    f1 = float(f1_score(yt, yp, average="micro", zero_division=0))
    return [acc, prec, rec, f1]


def _to_metric_dict(scores: Sequence[float]) -> Dict[str, float]:
    """把 [acc, prec, rec, f1] 转为结构化 dict。"""
    return {
        "accuracy": float(scores[0]),
        "precision": float(scores[1]),
        "recall": float(scores[2]),
        "f1": float(scores[3]),
    }


_ZERO_CLS = [0.0, 0.0, 0.0, 0.0]
_FULL_CLS = [1.0, 1.0, 1.0, 1.0]


def compute_classification(
    gold_items: Sequence[Dict[str, Any]],
    pred_items: Sequence[Dict[str, Any]],
    mode: str,
    level: str,
) -> Dict[str, float]:
    """单篇文档、单粒度、单模式的修辞分类指标。

    逻辑：
        - **hard**: 仅 sentenceIdx 完全匹配的句组对参与评分；gold 无匹配 → [0,0,0,0]；
          多余 pred 句组 → 追加 [0,0,0,0] 惩罚；文档分数 = 所有句组分数均值
        - **soft**: 优先完全匹配；无完全匹配时取有重叠的 pred 句组分别评分后取均值；
          gold 无任何重叠 → [0,0,0,0]；多余 pred → 惩罚；文档分数 = 均值
        - **free**: 忽略句组边界，将全文档 rhetoricList 拍平后做 multi-hot 编码再比较

    :param gold_items: gold 的 rhetoricItems（已 merge 预处理）
    :param pred_items: 预测的 rhetoricItems（已 merge 预处理）
    :param mode: ``"hard"`` / ``"soft"`` / ``"free"``
    :param level: ``"rhetoric"`` / ``"form"`` / ``"content"``
    :return: dict ``{"accuracy", "precision", "recall", "f1"}``
    """
    if mode not in _EVAL_MODES:
        raise ValueError(f"mode 必须是 hard/soft/free，收到 {mode!r}")


    if mode == "free":
        gv = _multi_hot(_flatten_rhetoric_list(gold_items), level)
        pv = _multi_hot(_flatten_rhetoric_list(pred_items), level)
        return _to_metric_dict(_score_multi_hot_pair(gv, pv))


    matched, unmatched_gold, leftover_pred = _match_sentence_groups(gold_items, pred_items, mode)
    assert matched is not None and unmatched_gold is not None and leftover_pred is not None

    score_list: List[List[float]] = []
    for gold, matched_preds, match_type in matched:
        gold_rl = gold.get("rhetoricList") or []
        if match_type == "exact":

            for pred in matched_preds:
                gv = _multi_hot(gold_rl, level)
                pv = _multi_hot(pred.get("rhetoricList") or [], level)
                score_list.append(_score_multi_hot_pair(gv, pv))
        else:

            pair_scores = []
            for pred in matched_preds:
                gv = _multi_hot(gold_rl, level)
                pv = _multi_hot(pred.get("rhetoricList") or [], level)
                pair_scores.append(_score_multi_hot_pair(gv, pv))
            score_list.append(np.mean(np.array(pair_scores), axis=0).tolist())


    for _ in unmatched_gold:
        score_list.append(list(_ZERO_CLS))

    for _ in leftover_pred:
        score_list.append(list(_ZERO_CLS))


    if gold_items:
        if pred_items and score_list:
            doc = np.mean(np.array(score_list), axis=0).tolist()
        else:
            doc = list(_ZERO_CLS)
    else:
        doc = list(_ZERO_CLS) if pred_items else list(_FULL_CLS)
    return _to_metric_dict(doc)


def compute_classification_f1(
    pred_items: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
    level: str = "rhetoric",
) -> Dict[str, float]:
    """[已弃用] 兼容旧签名，等价于 free 模式的文档级分类指标。

    新代码请使用 :func:`compute_classification`。
    """
    return compute_classification(gold_items, pred_items, mode="free", level=level)


def _locate_from_document(
    idx: Sequence[int],
    offset: int,
) -> List[int]:
    """把句内相对下标转换为文档全局下标。

    逻辑：
    累加句组首句之前所有句子的字符长度（offset），-1 保持不变。
    """
    if not idx:
        return []
    return [int(i) + offset if i != -1 else i for i in idx]


def _aggregate_components(
    rhetoric_list: Sequence[Dict[str, Any]],
    sentence_list: Sequence[str],
    sentence_idx: Sequence[int],
) -> Dict[str, List[Tuple[List[str], List[int], List[int]]]]:
    """修辞成分聚合，每个成分实体形成三元组 (成分内容列表, 起始下标列表, 结束下标列表)。

    逻辑：
        - 成分内容去除全角括号说明并去重保序（保留清洗后可能为空的字符串）
        - 位置统一转换为文档全局下标
        - **保留每个 begin/end 对独立**（不合并为单区间）

    :return: ``{component_key: [(contents, begin_idxs, end_idxs), ...]}``
    """
    result: Dict[str, List[Tuple[List[str], List[int], List[int]]]] = {k: [] for k in _COMPONENT_KEYS}
    if not sentence_idx:
        return result

    first = sentence_idx[0]
    offset = len("".join(sentence_list[:first])) if isinstance(first, int) and first >= 0 else 0

    for rhetoric in rhetoric_list:
        if not isinstance(rhetoric, dict):
            continue
        for key in _COMPONENT_KEYS:

            if key not in rhetoric or f"{key}BeginIdx" not in rhetoric or f"{key}EndIdx" not in rhetoric:
                continue
            raw_contents = rhetoric.get(key)
            if not isinstance(raw_contents, list):
                continue

            cleaned = [_COMPONENT_CLEAN_PATTERN.sub("", c) if isinstance(c, str) else c for c in raw_contents]
            deduped: List[str] = []
            for c in cleaned:
                if c not in deduped:
                    deduped.append(c)
            begin_idxs = _locate_from_document(rhetoric.get(f"{key}BeginIdx") or [], offset)
            end_idxs = _locate_from_document(rhetoric.get(f"{key}EndIdx") or [], offset)
            result[key].append((deduped, begin_idxs, end_idxs))
    return result


def _aggregate_all_components(
    items: Sequence[Dict[str, Any]],
    sentence_list: Sequence[str],
) -> Dict[str, List[Tuple[List[str], List[int], List[int]]]]:
    """free 模式：把全文档所有句组的成分聚合到一起（不区分句组边界）。

    """
    agg: Dict[str, List[Tuple[List[str], List[int], List[int]]]] = {k: [] for k in _COMPONENT_KEYS}
    for item in items:
        if "sentenceIdx" not in item:
            continue
        comp = _aggregate_components(
            item.get("rhetoricList") or [], sentence_list, item.get("sentenceIdx") or []
        )
        for key in _COMPONENT_KEYS:
            agg[key].extend(comp[key])
    return agg


def _list_similarity(a1: List[str], a2: List[str]) -> float:
    """两个字符串列表的内容相似度（内层匈牙利算法）。

    逻辑：
    构建 Levenshtein 距离矩阵 → linear_sum_assignment → 归一化相似度，未匹配项以长度计入分母。
    """
    if not a1 and not a2:
        return 1.0
    sum_a1 = sum(len(s) for s in a1)
    sum_a2 = sum(len(s) for s in a2)
    if not a1 or not a2:
        total = sum_a1 + sum_a2
        return 1.0 - (total / total) if total else 1.0

    cost = np.array(
        [[levenshtein_distance(s1, s2) for s2 in a2] for s1 in a1],
        dtype=float,
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_dist = float(cost[row_ind, col_ind].sum())
    matched_a1 = sum(len(a1[i]) for i in row_ind)
    matched_a2 = sum(len(a2[j]) for j in col_ind)
    unpaired = (sum_a1 - matched_a1) + (sum_a2 - matched_a2)
    total_dist = matched_dist + unpaired
    max_possible = sum_a1 + sum_a2
    return 1.0 - (total_dist / max_possible) if max_possible else 1.0


def _content_similarity(A1: List[List[str]], A2: List[List[str]]) -> float:
    """多个成分实体之间的整体内容相似度（外层匈牙利算法）。

    逻辑：
    构建实体间相似度矩阵 → linear_sum_assignment → 均值，未匹配项以 0 分计入分母。
    """
    if not A1:
        return 1.0 if not A2 else 0.0
    if not A2:
        return 0.0
    m, n = len(A1), len(A2)
    sim_mat = np.array(
        [[_list_similarity(a1, a2) for a2 in A2] for a1 in A1],
        dtype=float,
    )
    row_ind, col_ind = linear_sum_assignment(-sim_mat)
    total = float(sim_mat[row_ind, col_ind].sum())
    return total / max(m, n)


def _position_iou(
    true_locs: Sequence[Tuple[Sequence[int], Sequence[int]]],
    pred_locs: Sequence[Tuple[Sequence[int], Sequence[int]]],
) -> float:
    """成分位置 IoU（字符级）。

    逻辑：
    将每个 (begin, end) 对独立展开为字符索引集合，计算 |intersection| / |union|；
    双空 → 1.0。
    """
    def get_items(loc_list: Sequence[Tuple[Sequence[int], Sequence[int]]]) -> set[Tuple[int, int]]:
        items: set[Tuple[int, int]] = set()
        for begin_l, end_l in loc_list:
            for b, e in zip(begin_l, end_l):
                if b != -1 and e != -1:
                    items.add((min(b, e), max(b, e)))
        return items

    def expand(labels: set[Tuple[int, int]]) -> set[int]:
        s: set[int] = set()
        for start, end in labels:
            s.update(range(start, end + 1))
        return s

    true_set = expand(get_items(true_locs))
    pred_set = expand(get_items(pred_locs))
    union = true_set | pred_set
    if not union:
        return 1.0
    return len(true_set & pred_set) / len(union)


def _component_pair_scores(
    gold_item: Dict[str, Any],
    pred_item: Dict[str, Any],
    sentence_list: Sequence[str],
) -> Tuple[List[float], List[float]]:
    """对一对匹配的句组计算三类成分的内容相似度与位置 IoU。

    :return: ``(content_scores[conj,tenor,veh], location_scores[conj,tenor,veh])``
    """
    gold_comp = _aggregate_components(
        gold_item.get("rhetoricList") or [], sentence_list, gold_item.get("sentenceIdx") or []
    )
    pred_comp = _aggregate_components(
        pred_item.get("rhetoricList") or [], sentence_list, pred_item.get("sentenceIdx") or []
    )
    content_scores: List[float] = []
    location_scores: List[float] = []
    for key in _COMPONENT_KEYS:
        gc = gold_comp[key]
        pc = pred_comp[key]
        content_scores.append(_content_similarity(
            [e[0] for e in gc if e[0]],
            [e[0] for e in pc if e[0]],
        ))
        location_scores.append(_position_iou(
            [(e[1], e[2]) for e in gc if e[1] and e[2]],
            [(e[1], e[2]) for e in pc if e[1] and e[2]],
        ))
    return content_scores, location_scores


_ZERO_COMP = [0.0, 0.0, 0.0]
_FULL_COMP = [1.0, 1.0, 1.0]


def compute_component(
    gold_items: Sequence[Dict[str, Any]],
    pred_items: Sequence[Dict[str, Any]],
    mode: str,
    sentence_list: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """单篇文档、单模式的修辞成分指标。

    逻辑：
        - **hard**: 仅 sentenceIdx 完全匹配的句组对参与；gold 无匹配 → [0,0,0]；
          多余 pred → 惩罚 [0,0,0]；文档分数 = 均值
        - **soft**: 优先完全匹配；无完全匹配时取重叠 pred 分别评分后均值；多余 pred 惩罚
        - **free**: 全文档成分聚合后统一比较（不区分句组边界）

    :param gold_items: gold 的 rhetoricItems（已 merge 预处理）
    :param pred_items: 预测的 rhetoricItems（已 merge 预处理）
    :param mode: ``"hard"`` / ``"soft"`` / ``"free"``
    :param sentence_list: 当前文档的分句列表，用于位置全局下标换算
    :return: ``{"conjunction": {"content_sim", "position_iou"}, "tenor": {...}, "vehicle": {...}}``
    """
    if mode not in _EVAL_MODES:
        raise ValueError(f"mode 必须是 hard/soft/free，收到 {mode!r}")
    s_list = list(sentence_list) if sentence_list else []


    if mode == "free":
        gold_agg = _aggregate_all_components(gold_items, s_list)
        pred_agg = _aggregate_all_components(pred_items, s_list)
        out: Dict[str, Dict[str, float]] = {}
        for key in _COMPONENT_KEYS:
            gc = gold_agg[key]
            pc = pred_agg[key]
            out[key] = {
                "content_sim": float(_content_similarity(
                    [e[0] for e in gc if e[0]], [e[0] for e in pc if e[0]])),
                "position_iou": float(_position_iou(
                    [(e[1], e[2]) for e in gc if e[1] and e[2]],
                    [(e[1], e[2]) for e in pc if e[1] and e[2]])),
            }
        return out


    matched, unmatched_gold, leftover_pred = _match_sentence_groups(gold_items, pred_items, mode)
    assert matched is not None and unmatched_gold is not None and leftover_pred is not None

    content_scores: List[List[float]] = []
    location_scores: List[List[float]] = []
    for gold, matched_preds, match_type in matched:
        if match_type == "exact":
            for pred in matched_preds:
                cs, ls = _component_pair_scores(gold, pred, s_list)
                content_scores.append(cs)
                location_scores.append(ls)
        else:
            cs_list, ls_list = [], []
            for pred in matched_preds:
                cs, ls = _component_pair_scores(gold, pred, s_list)
                cs_list.append(cs)
                ls_list.append(ls)
            content_scores.append(np.mean(np.array(cs_list), axis=0).tolist())
            location_scores.append(np.mean(np.array(ls_list), axis=0).tolist())

    for _ in unmatched_gold:
        content_scores.append(list(_ZERO_COMP))
        location_scores.append(list(_ZERO_COMP))
    for _ in leftover_pred:
        content_scores.append(list(_ZERO_COMP))
        location_scores.append(list(_ZERO_COMP))


    if gold_items:
        if pred_items and content_scores:
            content_doc = np.mean(np.array(content_scores), axis=0).tolist()
            location_doc = np.mean(np.array(location_scores), axis=0).tolist()
        else:
            content_doc = list(_ZERO_COMP)
            location_doc = list(_ZERO_COMP)
    else:
        if pred_items:
            content_doc = list(_ZERO_COMP)
            location_doc = list(_ZERO_COMP)
        else:
            content_doc = list(_FULL_COMP)
            location_doc = list(_FULL_COMP)

    return {
        key: {
            "content_sim": float(content_doc[i]),
            "position_iou": float(location_doc[i]),
        }
        for i, key in enumerate(_COMPONENT_KEYS)
    }


def compute_component_metrics(
    pred_items: Sequence[Dict[str, Any]],
    gold_items: Sequence[Dict[str, Any]],
    sentence_list: Optional[Sequence[str]] = None,
    sentence_idx: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """[已弃用] 兼容旧签名（输入为拍平的 rhetoricList）。

    新代码请使用 :func:`compute_component`。此处把拍平列表包装为单句组伪条目后，
    以 hard 模式调用新实现。
    """
    s_idx = list(sentence_idx) if sentence_idx else [0]
    gold_pseudo = [{"sentenceIdx": s_idx, "rhetoricList": list(gold_items)}]
    pred_pseudo = [{"sentenceIdx": s_idx, "rhetoricList": list(pred_items)}]
    return compute_component(gold_pseudo, pred_pseudo, mode="hard", sentence_list=sentence_list)


def _compute_krippendorff_alpha(
    gold_vectors: Sequence[np.ndarray],
    pred_vectors: Sequence[np.ndarray],
) -> Optional[float]:
    """计算 free 模式下 gold / pred multi-hot 矩阵的 Krippendorff's alpha。
    把每个标注者的标签矩阵展平，将每个 样本-类别 对视为独立数据点。

    :return: alpha 浮点值；若 krippendorff 库不可用或计算失败则返回 None
    """
    if not _HAS_KRIPPENDORFF or not gold_vectors or not pred_vectors:
        return None
    try:
        a1 = np.array(gold_vectors).flatten()
        a2 = np.array(pred_vectors).flatten()
        return float(_krippendorff_alpha(reliability_data=[a1, a2], level_of_measurement="nominal"))
    except Exception as exc:  # pragma: no cover
        logging.warning("Krippendorff's alpha 计算失败: %s", exc)
        return None


def _normalize_modes(value: Any, default: Sequence[str]) -> List[str]:
    """把配置中的 eval_modes 取值规整为合法模式列表。

    兼容：None → 默认全开；bool → True 全开 / False 全关；list → 过滤合法模式。
    """
    if value is None:
        return list(default)
    if isinstance(value, bool):
        return list(default) if value else []
    if isinstance(value, (list, tuple, set)):
        return [m for m in value if m in _EVAL_MODES]
    return list(default)


def evaluate_all(
    predictions: Sequence[Dict[str, Any]],
    gold_data: Sequence[Dict[str, Any]],
    eval_modes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """端到端评测。

    流程:
        1. 解析每条预测的原始字符串 + merge 预处理（gold 与 pred 均执行）
        2. 对每条文档、每种 mode，分别计算 location / classification / component
        3. 汇总为按模式分类的结构化 dict

    :param predictions: list[dict]，每条含 ``rhetoricItems_pred``（原始字符串）
    :param gold_data: list[dict]，每条含 ``rhetoricItems``（gold 标签）与 ``sentenceList``
    :param eval_modes: 可选配置 dict，形如
        ``{"location": ["hard","soft","free"], "classification": [...], "component": [...]}``
        默认三类指标均启用全部三种模式
    :return: 结构化评测结果 dict
    """
    if len(predictions) != len(gold_data):
        raise ValueError(
            f"predictions({len(predictions)}) 与 gold_data({len(gold_data)}) 长度不一致"
        )

    em = eval_modes or {}
    location_modes = _normalize_modes(em.get("location"), _EVAL_MODES)
    classification_modes = _normalize_modes(em.get("classification"), _EVAL_MODES)
    component_modes = _normalize_modes(em.get("component"), _EVAL_MODES)

    total = len(predictions)
    parse_successes = 0
    parse_failures = 0
    parse_failures_idxes: List[int] = []


    docs: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]] = []
    for pred, gold in zip(predictions, gold_data):
        raw = pred.get("rhetoricItems_pred")
        parsed = parse_prediction(raw) if isinstance(raw, str) else None
        if parsed is None:
            parse_failures += 1

            idx_val = pred.get("idx")
            if idx_val is not None:
                parse_failures_idxes.append(int(idx_val))
            pred_items: List[Dict[str, Any]] = []
        else:
            parse_successes += 1
            pred_items = _extract_rhetoric_items(parsed)

        gold_items = gold.get("rhetoricItems") or []
        if not isinstance(gold_items, list):
            gold_items = []


        gold_merged = merge_rhetoric_items(gold_items)
        pred_merged = merge_rhetoric_items(pred_items)
        sentence_list = gold.get("sentenceList") or []
        docs.append((gold_merged, pred_merged, list(sentence_list)))


    results: Dict[str, Any] = {}
    for mode in _EVAL_MODES:
        need_loc = mode in location_modes
        need_cls = mode in classification_modes
        need_comp = mode in component_modes
        if not (need_loc or need_cls or need_comp):
            continue

        entry: Dict[str, Any] = {}


        if need_loc:
            loc = [compute_location_iou(p, g, mode=mode) for g, p, _ in docs]
            entry["location"] = float(np.mean(loc)) if loc else 0.0


        if need_cls:
            cls_acc: Dict[str, Dict[str, List[float]]] = {
                lv: {"accuracy": [], "precision": [], "recall": [], "f1": []}
                for lv in CLASSIFICATION_LEVELS
            }
            for g, p, _ in docs:
                for lv in CLASSIFICATION_LEVELS:
                    m = compute_classification(g, p, mode, lv)
                    for k in cls_acc[lv]:
                        cls_acc[lv][k].append(m[k])
            entry["classification"] = {
                lv: {k: (float(np.mean(v)) if v else 0.0) for k, v in cls_acc[lv].items()}
                for lv in CLASSIFICATION_LEVELS
            }

            if mode == "free":
                alpha_out: Dict[str, Optional[float]] = {}
                for lv in CLASSIFICATION_LEVELS:
                    gvecs = [_multi_hot(_flatten_rhetoric_list(g), lv) for g, _, _ in docs]
                    pvecs = [_multi_hot(_flatten_rhetoric_list(p), lv) for _, p, _ in docs]
                    alpha_out[lv] = _compute_krippendorff_alpha(gvecs, pvecs)
                entry["krippendorff_alpha"] = alpha_out


        if need_comp:
            comp_acc: Dict[str, Dict[str, List[float]]] = {
                key: {"content_sim": [], "position_iou": []} for key in _COMPONENT_KEYS
            }
            for g, p, s_list in docs:
                doc_comp = compute_component(g, p, mode, s_list)
                for key in _COMPONENT_KEYS:
                    comp_acc[key]["content_sim"].append(doc_comp[key]["content_sim"])
                    comp_acc[key]["position_iou"].append(doc_comp[key]["position_iou"])
            entry["component"] = {
                key: {k: (float(np.mean(v)) if v else 0.0) for k, v in comp_acc[key].items()}
                for key in _COMPONENT_KEYS
            }

        results[mode] = entry

    result = {
        "total_samples": total,
        "parse_successes": parse_successes,
        "parse_failures": parse_failures,
        "parse_failures_idxes": parse_failures_idxes,
        "results": results,
    }

    logging.info(
        "evaluate_all 完成: total=%d, parse_success=%d, parse_fail=%d, modes=%s",
        total, parse_successes, parse_failures, list(results.keys()),
    )
    return result


__all__ = [
    "RHETORIC_LABELS",
    "FORM_LABELS",
    "CONTENT_LABELS",
    "CLASSIFICATION_LEVELS",
    "merge_rhetoric_items",
    "parse_prediction",
    "compute_location_iou",
    "compute_classification",
    "compute_classification_f1",
    "compute_component",
    "compute_component_metrics",
    "evaluate_all",
]
