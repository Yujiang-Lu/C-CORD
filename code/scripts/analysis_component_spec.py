#!/usr/bin/env python3


from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT = os.path.dirname(_ROOT)
CORPUS = os.path.join(_PROJECT, "data",
                      "datasets_base", "C-CoRD.json")
TEST_SPLIT = os.path.join(_PROJECT, "data",
                          "datasets_split", "test.json")
OUT_DIR = os.path.join(_ROOT, "results", "component_spec")

R = "required"
O = "optional"
F = "forbidden"


SPEC: Dict[Tuple[str, str], Tuple[str, str, Tuple[str, str], Tuple[str, str], Tuple[str, str]]] = {
    ("比喻", "明喻"): ("metaphor", "explicit simile",
                       (R, "simile marker"), (R, "tenor"), (R, "vehicle")),
    ("比喻", "暗喻"): ("metaphor", "implicit metaphor",
                       (O, "metaphor marker"), (R, "tenor"), (R, "vehicle")),
    ("比喻", "借喻"): ("metaphor", "vehicle-only simile",
                       (F, "\\empty"), (F, "\\empty"), (R, "vehicle")),
    ("比拟", "名词"): ("personification", "nominal projection",
                       (F, "\\empty"), (R, "tenor"), (R, "projected noun")),
    ("比拟", "动词"): ("personification", "verbal projection",
                       (F, "\\empty"), (R, "tenor"), (R, "projected verb")),
    ("比拟", "形容词"): ("personification", "adjectival projection",
                       (F, "\\empty"), (R, "tenor"), (R, "projected adjective")),
    ("夸张", "直接夸张"): ("hyperbole", "direct hyperbole",
                          (F, "\\empty"), (R, "hyperbolised object"), (R, "hyperbolic expression")),
    ("夸张", "间接夸张"): ("hyperbole", "indirect hyperbole",
                          (F, "\\empty"), (R, "hyperbolised object"), (R, "hyperbolic expression")),
    ("排比", "成分排比"): ("parallelism", "constituent-level",
                          (R, "parallel frame"), (F, "\\empty"), (R, "parallel items")),
    ("排比", "句子排比"): ("parallelism", "clause-level",
                          (R, "parallel frame"), (F, "\\empty"), (R, "parallel items")),
    ("反复", "连续反复"): ("repetition", "immediate repetition",
                          (R, "lead-in"), (F, "\\empty"), (R, "repeated content")),
    ("反复", "间隔反复"): ("repetition", "interspersed repetition",
                          (R, "lead-in"), (F, "\\empty"), (R, "repeated content")),
    ("设问", "问答连属"): ("hypophora", "adjacent question--answer",
                          (F, "\\empty"), (R, "question"), (R, "answer")),
    ("设问", "问答不连属"): ("hypophora", "non-adjacent question--answer",
                            (F, "\\empty"), (R, "question"), (O, "answer")),
    ("反问", "单句反问"): ("rhetorical question", "single-clause",
                           (F, "\\empty"), (R, "questioned claim"), (R, "rhetorical span")),
    ("反问", "复句反问"): ("rhetorical question", "multi-clause",
                           (F, "\\empty"), (R, "questioned claim"), (R, "rhetorical span")),
    ("摹状", "直感"): ("depiction", "same-sense",
                       (R, "mimetic expression"),
                       (R, "target sense$^{\\dagger}$"), (R, "evoked sense$^{\\dagger}$")),
    ("摹状", "通感"): ("depiction", "cross-sense",
                       (R, "mimetic expression"),
                       (R, "target sense$^{\\dagger}$"), (R, "evoked sense$^{\\dagger}$")),
    ("引用", "明引"): ("quotation", "explicit quotation",
                       (R, "cited source"), (O, "quoted view"), (R, "quoted content")),
    ("引用", "转引"): ("quotation", "second-hand quotation",
                       (R, "cited source"), (O, "quoted view"), (R, "quoted content")),
    ("引用", "暗引"): ("quotation", "implicit quotation",
                       (F, "\\empty"), (O, "quoted view"), (R, "quoted content")),
    ("对偶", "正对"): ("antithesis", "synonymous pairing",
                       (F, "\\empty"), (R, "first-line tokens"), (R, "second-line tokens")),
    ("对偶", "反对"): ("antithesis", "contrastive pairing",
                       (F, "\\empty"), (R, "first-line tokens"), (R, "second-line tokens")),
    ("对偶", "串对"): ("antithesis", "sequential pairing",
                       (F, "\\empty"), (R, "first-line tokens"), (R, "second-line tokens")),
}

COMPS: Tuple[str, ...] = ("conjunction", "tenor", "vehicle")
IDX_KEY = {"conjunction": "conjunctionBeginIdx", "tenor": "tenorBeginIdx",
           "vehicle": "vehicleBeginIdx"}


CONDITIONAL_BY_CONTENT = {("设问", "问答不连属", "vehicle")}


RATE_BOX = r"\makebox[\compsegw][r]{%s}"


def _as_list(v: Any) -> List[Any]:
    if isinstance(v, str):
        return [v] if v else []
    return list(v or [])


def accumulate(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, int]]:
    """按 (粗类, 形式子类) 统计实例数、各成分非空数、各成分虚体(−1)数。"""
    agg: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"n": 0, **{f"{c}": 0 for c in COMPS}, **{f"{c}_abs": 0 for c in COMPS}})
    for rec in records:
        for item in rec.get("rhetoricItems") or []:
            for rl in item.get("rhetoricList") or []:
                key = (rl.get("rhetoric"), rl.get("form"))
                a = agg[key]
                a["n"] += 1
                for c in COMPS:
                    vals = _as_list(rl.get(c))
                    if not vals:
                        continue
                    a[c] += 1
                    if any(i == -1 for i in _as_list(rl.get(IDX_KEY[c]))):
                        a[f"{c}_abs"] += 1
    return agg


def reconcile(agg: Dict[Tuple[str, str], Dict[str, int]]) -> List[str]:
    """把规范期望与观测率对账，返回不一致清单。"""
    problems: List[str] = []
    for key, spec in SPEC.items():
        a = agg.get(key)
        if a is None or a["n"] == 0:
            problems.append(f"语料中缺少 {key} 的实例")
            continue
        for i, c in enumerate(COMPS):
            state = spec[2 + i][0]
            rate = 100.0 * a[c] / a["n"]
            if state == F and rate != 0.0:
                problems.append(f"{key} {c}: 规范要求留空但观测 {rate:.1f}%")
            if state == R and rate < 99.5:
                problems.append(f"{key} {c}: 规范要求必填但观测 {rate:.1f}%")
            if state == O and not (0.0 < rate < 100.0):
                problems.append(f"{key} {c}: 规范记为可选但观测 {rate:.1f}%")
    return problems


def build_rows(agg: Dict[Tuple[str, str], Dict[str, int]]) -> List[List[str]]:
    """生成表体：填写率列按 "/" 等宽对齐，规范标为 optional 的那一个值加粗。"""
    rows: List[List[str]] = []
    last_type: Dict[str, str] = {}
    for (rh, fm), (en_type, en_form, cs, ts, vs) in SPEC.items():
        a = agg[(rh, fm)]
        n = a["n"]
        cells = []
        for i, c in enumerate(COMPS):
            rate = f"{100.0 * a[c] / n:.1f}"
            if SPEC[(rh, fm)][2 + i][0] == O:
                rate = r"\textbf{" + rate + "}"

            cells.append(RATE_BOX % rate)
        rate_col = "\\,\\,/\\,\\,".join(cells)
        type_cell = "" if en_type == last_type.get("t") else f"\\textbf{{{en_type}}}"
        last_type["t"] = en_type
        roles = []
        for i, c in enumerate(COMPS):
            name = SPEC[(rh, fm)][2 + i][1]
            if (rh, fm, c) in CONDITIONAL_BY_CONTENT:
                name += "$^{\\ddagger}$"
            roles.append(name)
        rows.append([type_cell, en_form] + roles + [str(n), rate_col])
    return rows


def rate_band(values: List[str]) -> str:
    """将三段数字拼成固定宽度的单元格（以 "/" 为基准对齐）。"""
    return "\\,\\,/\\,\\,".join(RATE_BOX % v for v in values)


BAND_REF = rate_band(["100.0"] * 3)

BAND_DEF_TEX = (f"  \\newsavebox{{\\compbandbox}}%\n"
                f"  \\newlength{{\\compsegw}}%\n"
                f"  \\settowidth{{\\compsegw}}{{100.0}}%\n"
                f"  \\sbox{{\\compbandbox}}{{{BAND_REF}}}%\n")


def emit_tex(rows: List[List[str]]) -> str:


    head = (r"    \textbf{Coarse type} & \textbf{Form subtype} & \textbf{C} & \textbf{T} & "
            r"\textbf{V} & \textbf{$n$} & "
            r"\makebox[\wd\compbandbox][c]{\textbf{C/T/V filled (\%)}} \\")
    body: List[str] = []
    for r in rows:
        cells = [c.replace("\\empty", r"$\emptyset$") for c in r]
        body.append("    " + " & ".join(cells) + r" \\")
    return head + "\n" + "    \\midrule\n" + "\n".join(body)


def abstract_stats(agg: Dict[Tuple[str, str], Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    """按成分汇总虚体数与非空数。"""
    out: Dict[str, Dict[str, float]] = {}
    for c in COMPS:
        filled = sum(a[c] for a in agg.values())
        absn = sum(a[f"{c}_abs"] for a in agg.values())
        out[c] = {"non_empty": filled, "abstract": absn,
                  "abstract_share": round(100.0 * absn / filled, 1) if filled else 0.0}
    return out


def content_breakdown(records: List[Dict[str, Any]], rh: str, fm: str,
                      comp: str) -> Dict[str, Dict[str, Any]]:

    out: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        for item in rec.get("rhetoricItems") or []:
            for rl in item.get("rhetoricList") or []:
                if (rl.get("rhetoric"), rl.get("form")) != (rh, fm):
                    continue
                content = str(rl.get("content"))
                d = out.setdefault(content, {"n": 0, "filled": 0})
                d["n"] += 1
                if _as_list(rl.get(comp)):
                    d["filled"] += 1
    for d in out.values():
        d["filled_pct"] = round(100.0 * d["filled"] / d["n"], 1) if d["n"] else 0.0
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    corpus = json.load(open(CORPUS, encoding="utf-8"))
    test = json.load(open(TEST_SPLIT, encoding="utf-8"))

    agg_all = accumulate(corpus)
    agg_test = accumulate(test)

    problems = reconcile(agg_all)
    rows = build_rows(agg_all)
    tex = emit_tex(rows)
    with open(os.path.join(OUT_DIR, "tab_component_spec.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    with open(os.path.join(OUT_DIR, "tab_component_spec_band.def"), "w", encoding="utf-8") as fh:
        fh.write(BAND_DEF_TEX)

    stats = {
        "meta": {
            "corpus_records": len(corpus), "test_records": len(test),
            "expectation_legend": "required = filled by every instance of that form subtype; "
                                  "forbidden = empty by definition; optional = may be empty",
            "abstract_note": "depiction tenor/vehicle are sense labels with character index -1; "
                             "evaluator._position_iou drops -1 entries and returns 1.0 when both "
                             "sides are empty, so their position IoU degenerates to an "
                             "emptiness check",
            "table_typography": "fill-rate column uses fixed-width boxes aligned on '/'; "
                                "values for norm-optional fields are \\textbf{}-bolded; "
                                "$^{\\ddagger}$ marks an optional field that is fully decided "
                                "by the content label",
        },
        "per_form_subtype": {
            f"{rh}/{fm}": {
                "n": agg_all[(rh, fm)]["n"],
                "filled": {c: agg_all[(rh, fm)][c] for c in COMPS},
                "filled_pct": {c: round(100.0 * agg_all[(rh, fm)][c] / agg_all[(rh, fm)]["n"], 1)
                               for c in COMPS},
                "abstract": {c: agg_all[(rh, fm)][f"{c}_abs"] for c in COMPS},
                "test_n": agg_test.get((rh, fm), {"n": 0})["n"],
                "expectation": {c: SPEC[(rh, fm)][2 + i][0] for i, c in enumerate(COMPS)},
                "role_names": {c: SPEC[(rh, fm)][2 + i][1] for i, c in enumerate(COMPS)},
            } for (rh, fm) in SPEC
        },
        "abstract_summary": {"corpus": abstract_stats(agg_all), "test": abstract_stats(agg_test)},
        "optional_fields": {
            f"{rh}/{fm}/{comp}": {
                "filled": agg_all[(rh, fm)][comp],
                "n": agg_all[(rh, fm)]["n"],
                "filled_pct": round(100.0 * agg_all[(rh, fm)][comp] / agg_all[(rh, fm)]["n"], 1),
                "decided_by_content_label": (rh, fm, comp) in CONDITIONAL_BY_CONTENT,
                "by_content_subtype": content_breakdown(corpus, rh, fm, comp),
            } for (rh, fm) in SPEC for i, comp in enumerate(COMPS)
            if SPEC[(rh, fm)][2 + i][0] == O
        },
        "norm_vs_data_mismatches": problems,
    }
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    md = ["# Appendix B 补充：C-CoRD 修辞成分划分体系", "",
          f"> 语料：`datasets_base/C-CoRD.json`（{len(corpus)} 记录 / "
          f"{sum(a['n'] for a in agg_all.values())} 实例）；"
          f"测试集：`datasets_split/test.json`（{len(test)} 记录 / "
          f"{sum(a['n'] for a in agg_test.values())} 实例）。",
          "", "规范期望与实测填写率的自动对账结果：**"
          + ("无不一致" if not problems else f"{len(problems)} 处不一致") + "**", ""]
    if problems:
        md += [f"- {p}" for p in problems] + [""]
    md += ["| 粗类 | 形式子类 | C | T | V | n | C/T/V 填写率 (%) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| " + " | ".join(c.replace("\\textbf{", "").replace("}", "").replace("\\empty", "∅")
                                    for c in r) + " |")
    md += ["", "## 虚体（字符索引 −1）占比", ""]
    for split, s in (("全库", stats["abstract_summary"]["corpus"]),
                     ("测试集", stats["abstract_summary"]["test"])):
        md.append(f"- {split}：" + "；".join(
            f"{c} 非空 {v['non_empty']} 条、虚体 {v['abstract']} 条（{v['abstract_share']}%）"
            for c, v in s.items()))
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    print("\n===== 成分划分体系表：对账与摘要 =====")
    print(f"形式子类行数：{len(rows)}")
    print(f"规范 vs 实测不一致：{len(problems)} 处")
    for p in problems:
        print("  !", p)
    for name, s in (("全库", stats["abstract_summary"]["corpus"]),
                    ("测试集", stats["abstract_summary"]["test"])):
        print(f"{name}虚体占比：" + "  ".join(
            f"{c} {v['abstract']}/{v['non_empty']}={v['abstract_share']}%" for c, v in s.items()))
    print(f"输出：{OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
