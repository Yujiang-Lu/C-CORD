"""
CLI 入口模块。

职责:
    - 解析命令行参数
    - 编排 data_loader / model_api / evaluator 完成端到端评测
    - 支持 ``--dry-run`` 模式（用 gold 数据模拟预测，验证管线正确性）
    - 支持 **分批测试**（``--batch-id`` / ``--batch-size`` / ``--batch-file``）
    - 支持 **逐条落盘**（每完成一条预测立即追加到 ``predictions.json``）
    - 支持 **断点续跑**（``--resume`` 跳过已完成样本，``--rerun-idx`` 强制重跑）
    - 支持 **排序输出**（``--sort-output`` 写盘前按 ``idx`` 升序排序）
    - 支持 **按配置归档**（评测完成后自动写入 ``metrics/`` 和 ``predictions/``
      子目录，文件名编码模型/参数信息，保留当前快照 ``metrics.json`` /
      ``predictions.json`` 不变）
    - 支持 **criteria 缓存**（首次生成后持久化到 ``{output_dir}/cache/``，
      相同配置的实验直接加载复用，跳过 API 调用；``--regenerate-criteria`` 强制重建）
    - 输出 ``metrics.json`` / ``predictions.json`` / ``run.log``

用法示例::

    # dry-run 全量（不调用 API，验证管线）
    python src/main.py --config experiments/baseline/config.yaml --dry-run

    # 只跑第 0 批（50 条），启用断点续跑
    python src/main.py --config experiments/baseline/config.yaml \\
        --split test --output-dir results/exp \\
        --batch-id 0 --resume

    # 强制重跑若干 idx（会覆盖 predictions.json 中的旧结果）
    python src/main.py --config experiments/baseline/config.yaml \\
        --split test --output-dir results/exp --rerun-idx 42,107,3

    # 自动重跑 metrics.json 中 parse_failures_idxes 列出的 idx
    python src/main.py --config experiments/baseline/config.yaml \\
        --split test --output-dir results/exp --resume --rerun-idx auto

    # 限制样本数（冒烟测试，跳过批次划分逻辑）
    python src/main.py --config experiments/baseline/config.yaml --max-samples 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_loader import (  # noqa: E402
    create_or_load_batches,
    extract_shots_by_indices,
    get_few_shot_indices,
    get_init_shot_indices,
    load_config,
    load_split,
    resolve_data_dir,
    resolve_output_dir,
    sample_few_shots,
    sample_init_shots,
)
from src.evaluator import evaluate_all  # noqa: E402
from src.model_api import build_init_messages, build_messages, detect_provider, generate_criteria, predict_single  # noqa: E402


def _setup_logging(output_dir: str, log_filename: str = "run.log") -> str:
    """配置 logging：控制台 + 文件双输出。

    日志目录选择优先级：
        1. ``{output_dir}/logs/``（推荐：与结果同目录，便于打包归档）
        2. ``{output_dir}`` 同级的 ``logs/``（回退：多个实验共享日志目录时使用）

    若两者均不存在，则创建第一个候选目录 ``{output_dir}/logs/``。

    :return: 实际日志文件路径
    """
    abs_output_dir = os.path.abspath(output_dir)
    candidate_dirs = [
        os.path.join(abs_output_dir, "logs"),
        os.path.join(os.path.dirname(abs_output_dir), "logs"),
    ]
    log_dir = next((d for d in candidate_dirs if os.path.isdir(d)), None)
    if log_dir is None:
        log_dir = candidate_dirs[0]
        os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return log_path


def _load_existing_predictions(pred_path: str) -> Dict[int, Dict[str, Any]]:
    """加载已有 predictions.json 到 ``idx -> record`` 的 dict。

    若文件不存在或格式损坏，返回空 dict（并写日志告警）。
    """
    if not os.path.exists(pred_path):
        return {}
    try:
        with open(pred_path, "r", encoding="utf-8") as f:
            arr = json.load(f)
    except json.JSONDecodeError as exc:
        logging.warning("已有 %s 解析失败（%s），按空 predictions 处理", pred_path, exc)
        return {}
    if not isinstance(arr, list):
        logging.warning("已有 %s 顶层非 list，按空 predictions 处理", pred_path)
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for item in arr:
        if isinstance(item, dict) and "idx" in item:
            out[item["idx"]] = item
    return out


def _save_predictions(
    existing: Dict[int, Dict[str, Any]],
    pred_path: str,
    sort_output: bool = False,
) -> None:
    """将 ``idx -> record`` 的 dict 以 JSON 数组形式整体写回 predictions.json。

    数据集规模仅 ~500 条，整体重写简单可靠；相比 jsonlines 中间文件，
    无需二次汇总，也保证 predictions.json 始终为合法的 JSON 数组，与 evaluator
    及后续分析脚本兼容。

    :param sort_output: 若为 True，写入前按 ``idx`` 升序排序。
    """
    os.makedirs(os.path.dirname(os.path.abspath(pred_path)) or ".", exist_ok=True)
    records = list(existing.values())
    if sort_output:
        records.sort(key=lambda x: x['idx'])
    tmp_path = pred_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, pred_path)


def _predict_one_dry_run(item: Dict[str, Any]) -> str:
    """dry-run 单条预测：把 gold 的 rhetoricItems 序列化为字符串。

    evaluator 解析后应得到与 gold 完全一致的预测，用于验证解析/IoU/分类/成分链路。
    """
    gold_items = item.get("rhetoricItems") or []
    return json.dumps({"rhetoricItems": gold_items}, ensure_ascii=False)


def _make_api_client(config: Dict[str, Any]) -> Any:
    """延迟创建 openai client，避免 dry-run 模式下强制依赖 openai 包。"""
    import openai

    api_key = os.environ.get("RHETORIC_API_KEY") or config.get("api_key")
    base_url = os.environ.get("RHETORIC_BASE_URL") or config.get("base_url")
    if not api_key or api_key == "YOUR_API_KEY":
        raise ValueError(
            "API Key 未配置：请设置环境变量 RHETORIC_API_KEY 或在 config['api_key'] 中填入；"
            "如需冒烟测试请使用 --dry-run"
        )
    return openai.OpenAI(api_key=api_key, base_url=base_url)


DRY_RUN_PLACEHOLDER_CRITERIA = "[DRY-RUN] placeholder criteria"


def _build_experiment_suffix(
    model_name: Any,
    seed: int,
    num_init_shots: int,
    num_few_shots: int,
    temperature: float,
    init_temperature: float,
    enable_thinking: Optional[bool] = None,
) -> str:
    """构造实验配置后缀字符串，编码关键参数，一看即知来源。

    返回值示例::

        qwen-plus_seed42_init30_few15_t0.7_it1.0
        qwen3.8-flash_seed42_init30_few15_t0.7_it1.0_think

    若 ``enable_thinking`` 不为 None，则追加思考模式标识：
        - ``True``  → ``_think``
        - ``False`` → ``_nothink``
        - ``None``  → 不追加（保持现有行为）

    用于 criteria 缓存、metrics 归档、predictions 归档等文件命名。
    """

    safe_model = str(model_name).replace("/", "_").replace("\\", "_").replace(":", "_")
    suffix = (
        f"{safe_model}_seed{seed}_init{num_init_shots}_few{num_few_shots}"
        f"_t{temperature}_it{init_temperature}"
    )
    if enable_thinking is not None:
        suffix += "_think" if enable_thinking else "_nothink"
    return suffix


def _build_criteria_cache_filename(
    model_name: Any,
    seed: int,
    num_init_shots: int,
    num_few_shots: int,
    temperature: float,
    init_temperature: float,
    enable_thinking: Optional[bool] = None,
) -> str:
    """构造 criteria 缓存文件名。

    命名规则::

        criteria_{model_name}_seed{seed}_init{num_init_shots}_few{num_few_shots}\
_t{temperature}_it{init_temperature}.json

    示例::

        criteria_qwen-plus_seed42_init30_few15_t0.7_it1.0.json
        criteria_qwen3.8-flash_seed42_init30_few15_t0.7_it1.0_think.json
    """
    suffix = _build_experiment_suffix(
        model_name, seed, num_init_shots, num_few_shots, temperature, init_temperature,
        enable_thinking=enable_thinking,
    )
    return f"criteria_{suffix}.json"


def _build_config_snapshot(
    model_name: Any,
    seed: int,
    num_init_shots: int,
    num_few_shots: int,
    temperature: float,
    init_temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """组装写入缓存文件的 config_snapshot（也用于加载时的一致性校验）。"""
    return {
        "model_name": model_name,
        "seed": seed,
        "num_init_shots": num_init_shots,
        "num_few_shots": num_few_shots,
        "temperature": temperature,
        "init_temperature": init_temperature,
        "max_tokens": max_tokens,
    }


def _verify_cache_config(cache: Dict[str, Any], expected: Dict[str, Any]) -> None:
    """校验缓存的 config_snapshot 与当前配置一致，防止文件名碰撞导致误用。

    不一致时仅告警（仍按文件名复用），提示用户可用 ``--regenerate-criteria`` 重建。
    """
    snap = cache.get("config_snapshot", {})
    if not isinstance(snap, dict):
        logging.warning("缓存 config_snapshot 格式异常，跳过一致性校验")
        return
    for key, cur_val in expected.items():
        if key in snap and snap[key] != cur_val:
            logging.warning(
                "缓存 config_snapshot[%s]=%r 与当前配置 %r 不一致（仍按文件名复用，"
                "如需重建请加 --regenerate-criteria）",
                key, snap[key], cur_val,
            )


def _load_parse_failures_idxes(output_dir: str) -> Set[int]:
    """从 ``{output_dir}/metrics.json`` 读取 ``parse_failures_idxes`` 字段。

    当文件不存在、读取失败或字段为空时，返回空集合（不抛异常）。
    """
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        logging.info("metrics.json 不存在（%s），无解析失败 idx", metrics_path)
        return set()
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("读取 metrics.json 失败：%s", exc)
        return set()
    idxes = metrics.get("parse_failures_idxes") or []
    out: Set[int] = set()
    for i in idxes:
        try:
            out.add(int(i))
        except (TypeError, ValueError):
            logging.warning("parse_failures_idxes 含非法值 %r，已跳过", i)
    return out


def _parse_rerun_idx(raw: Optional[str], output_dir: Optional[str] = None) -> Set[int]:
    """解析 ``--rerun-idx`` 取值为 int 集合。

    - ``42,107,3``：逗号分隔的整数列表
    - ``auto``（不区分大小写）：从 ``{output_dir}/metrics.json`` 的
      ``parse_failures_idxes`` 字段自动读取
    """
    if not raw:
        return set()
    if raw.strip().lower() == "auto":
        if output_dir is None:
            raise ValueError("--rerun-idx auto 需要 output_dir 以定位 metrics.json")
        idxes = _load_parse_failures_idxes(output_dir)
        if idxes:
            logging.info(
                "从 metrics.json 读取到 %d 个解析失败的 idx，将强制重跑：%s",
                len(idxes), sorted(idxes),
            )
        return idxes
    out: Set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError as exc:
            raise ValueError(f"--rerun-idx 需为逗号分隔的整数，收到 {part!r}") from exc
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="中文作文修辞识别评测管线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", required=True,
        help="YAML 配置文件路径（如 experiments/baseline/config.yaml）",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "validation", "test"],
        help="评测数据切分",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="结果输出目录；默认读取 config['output_dir']",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="不调用 API，用 gold 数据模拟预测（验证管线正确性）",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="仅对前 N 条样本进行预测（冒烟测试；指定后忽略批次划分）",
    )

    parser.add_argument(
        "--batch-id", type=int, default=None,
        help="只跑第 N 批（从 0 开始）；不指定则跑全部",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="每批样本数（生成批次文件时使用；已存在文件时以文件为准）",
    )
    parser.add_argument(
        "--batch-file", default=None,
        help="批次划分文件路径；默认 <config_dir>/<split>_batches.json",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="断点续跑：跳过 predictions.json 中已有 rhetoricItems_pred 的样本",
    )
    parser.add_argument(
        "--rerun-idx", default=None,
        help="强制重跑的样本 idx（逗号分隔，如 42,107,3），覆盖已有结果。"
             " 传入 'auto' 则自动从 metrics.json 的 parse_failures_idxes 读取",
    )
    parser.add_argument(
        "--sort-output", action="store_true",
        help="写入 predictions.json 时按 idx 升序排序",
    )
    parser.add_argument(
        "--regenerate-criteria", action="store_true",
        help="强制重新生成 criteria 并覆盖已有缓存（忽略 {output_dir}/cache/ 下的缓存文件）",
    )
    return parser


def _log_messages_once(
    init_shots: List[Dict[str, Any]],
    few_shots: List[Dict[str, Any]],
    todo: List[Dict[str, Any]],
    criteria: Optional[str],
) -> None:
    """在实验开始时一次性打印完整的 messages 结构到日志，方便检查 prompt 正确性。

    打印内容：
        1. Criteria 生成 Prompt（init messages）
        2. 第一条测试样本的任务推理 Prompt（task messages）
    """
    sep = "=" * 60


    init_msgs = build_init_messages(init_shots)
    logging.info(sep)
    logging.info("[MESSAGE LOG] Criteria 生成 Prompt（init_shots: %d 条）", len(init_shots))
    logging.info(sep)
    for msg in init_msgs:
        logging.info("--- %s ---", msg["role"])
        logging.info("%s", msg["content"])
    logging.info(sep)


    if todo:
        first_item = todo[0]
        task_msgs = build_messages(few_shots=few_shots, current_item=first_item, criteria=criteria)
        logging.info("")
        logging.info(sep)
        logging.info(
            "[MESSAGE LOG] 任务推理 Prompt（第一条样本 idx=%s）",
            first_item.get("idx", "unknown"),
        )
        logging.info(sep)
        for j, msg in enumerate(task_msgs):
            role = msg["role"]
            if role == "system":
                label = "system"
            elif j == len(task_msgs) - 1:
                label = "user (current)"
            elif role == "user":
                shot_num = (j + 1) // 2
                label = f"user (few-shot {shot_num})"
            else:
                shot_num = j // 2
                label = f"assistant (few-shot {shot_num})"
            logging.info("--- %s ---", label)
            logging.info("%s", msg["content"])
        logging.info(sep)
    else:
        logging.info("[MESSAGE LOG] 无待预测样本，跳过任务推理 Prompt 打印")


def main(argv: Optional[List[str]] = None) -> int:
    """主入口。返回 0 表示成功。"""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)


    cfg = load_config(args.config)


    data_dir = resolve_data_dir(args.config, cfg["data_dir"])
    output_dir = args.output_dir or resolve_output_dir(
        args.config, cfg.get("output_dir", "results/exp")
    )
    os.makedirs(output_dir, exist_ok=True)


    config_dir = os.path.dirname(os.path.abspath(args.config))
    batch_file = args.batch_file or os.path.join(
        config_dir, f"{args.split}_batches.json"
    )


    log_path = _setup_logging(output_dir)
    logging.info("=" * 60)
    logging.info("中文作文修辞识别评测管线启动")
    logging.info(
        "config=%s | split=%s | dry_run=%s | max_samples=%s",
        args.config, args.split, args.dry_run, args.max_samples,
    )
    logging.info(
        "batch_id=%s | batch_size=%d | resume=%s | rerun_idx=%s | batch_file=%s",
        args.batch_id, args.batch_size, args.resume, args.rerun_idx, batch_file,
    )
    logging.info("sort_output=%s", args.sort_output)
    logging.info("regenerate_criteria=%s", args.regenerate_criteria)
    logging.info("data_dir=%s", data_dir)
    logging.info("output_dir=%s", output_dir)
    logging.info("log_path=%s", log_path)
    logging.info("=" * 60)


    train_data = load_split(data_dir, "train")
    eval_data = load_split(data_dir, args.split)


    n_shots = int(cfg.get("num_few_shots", 15))
    n_init_shots = int(cfg.get("num_init_shots", 30))
    seed = int(cfg.get("seed", 42))


    model_name = cfg.get("model_name")
    temperature = float(cfg.get("temperature", 0.7))
    max_tokens = int(cfg.get("max_tokens", 8192))
    max_retries = int(cfg.get("max_retries", 3))
    init_temperature = float(cfg.get("init_temperature", 1.0))

    enable_thinking = cfg.get("enable_thinking")

    base_url = os.environ.get("RHETORIC_BASE_URL") or cfg.get("base_url", "")
    provider = detect_provider(base_url, model_name)
    logging.info("模型厂商: %s", provider)
    logging.info(
        "思考模式: %s",
        enable_thinking if enable_thinking is not None else "模型默认",
    )


    init_shot_indices = get_init_shot_indices(train_data, n=n_init_shots, seed=seed)
    init_shots = sample_init_shots(train_data, n=n_init_shots, seed=seed)


    idx_to_item: Dict[int, Dict[str, Any]] = {item["idx"]: item for item in eval_data}

    use_batching = args.max_samples is None
    batch_meta: Optional[Dict[str, Any]] = None
    if use_batching:
        logging.info("批次划分文件: %s", batch_file)
        batch_meta = create_or_load_batches(
            data_dir=data_dir,
            split_name=args.split,
            batch_size=args.batch_size,
            seed=seed,
            batch_file=batch_file,
        )
        total_samples = int(batch_meta["total_samples"])
        num_batches = int(batch_meta["num_batches"])
        if args.batch_id is None:

            target_indices: List[int] = []
            for b in batch_meta["batches"]:
                target_indices.extend(b["indices"])
            logging.info(
                "总样本数: %d, 批次数: %d, 当前批次: ALL (%d条)",
                total_samples, num_batches, len(target_indices),
            )
        else:
            if not (0 <= args.batch_id < num_batches):
                raise ValueError(
                    f"--batch-id={args.batch_id} 超出范围 [0, {num_batches - 1}]"
                )
            target_indices = list(batch_meta["batches"][args.batch_id]["indices"])
            logging.info(
                "总样本数: %d, 批次数: %d, 当前批次: %d (%d条)",
                total_samples, num_batches, args.batch_id, len(target_indices),
            )
    else:
        subset = eval_data[: args.max_samples]
        target_indices = [item["idx"] for item in subset]
        logging.info(
            "冒烟测试模式：--max-samples=%d，跳过批次划分", args.max_samples,
        )

    subset = [idx_to_item[i] for i in target_indices if i in idx_to_item]
    if len(subset) != len(target_indices):
        logging.warning(
            "有 %d 个 idx 在 %s 数据中未找到，已忽略",
            len(target_indices) - len(subset), args.split,
        )


    pred_path = os.path.join(output_dir, "predictions.json")
    existing: Dict[int, Dict[str, Any]] = _load_existing_predictions(pred_path)
    rerun_set = _parse_rerun_idx(args.rerun_idx, output_dir)


    if (
        args.rerun_idx
        and args.rerun_idx.strip().lower() == "auto"
        and not rerun_set
    ):
        logging.info("无需重跑")
        return 0

    todo: List[Dict[str, Any]] = []
    skipped = 0
    for item in subset:
        idx = item["idx"]
        if idx in rerun_set:
            todo.append(item)
            continue
        if args.resume:
            prev = existing.get(idx)
            if prev is not None and prev.get("rhetoricItems_pred"):
                skipped += 1
                continue
        todo.append(item)

    if args.resume or rerun_set:
        logging.info("断点续跑: 已跳过 %d 条已有结果", skipped)
        if rerun_set:
            logging.info("强制重跑 idx: %s", sorted(rerun_set))
    logging.info("待预测: %d 条", len(todo))


    client = None
    criteria: Optional[str] = None


    few_shot_indices = get_few_shot_indices(
        train_data, n=n_shots, seed=seed, exclude_indices=init_shot_indices
    )
    few_shots = sample_few_shots(
        train_data, n=n_shots, seed=seed, exclude_indices=init_shot_indices
    )


    cache_filename = _build_criteria_cache_filename(
        model_name=model_name,
        seed=seed,
        num_init_shots=n_init_shots,
        num_few_shots=n_shots,
        temperature=temperature,
        init_temperature=init_temperature,
        enable_thinking=enable_thinking,
    )
    cache_path = os.path.join(output_dir, "cache", cache_filename)
    config_snapshot = _build_config_snapshot(
        model_name=model_name,
        seed=seed,
        num_init_shots=n_init_shots,
        num_few_shots=n_shots,
        temperature=temperature,
        init_temperature=init_temperature,
        max_tokens=max_tokens,
    )

    cache_hit = os.path.exists(cache_path) and not args.regenerate_criteria
    if cache_hit:

        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        criteria = cache.get("criteria")

        _verify_cache_config(cache, config_snapshot)
        logging.info("加载已有 criteria 缓存: %s", cache_path)
        logging.info("  generated_at: %s", cache.get("generated_at"))
        logging.info("  base_url: %s", cache.get("base_url"))
        logging.info("  dry_run: %s", cache.get("dry_run", False))
        logging.info("  init_shot_indices: %s", cache.get("init_shot_indices"))
        logging.info("  few_shot_indices: %s", cache.get("few_shot_indices"))

        if cache.get("few_shot_indices"):
            few_shots = extract_shots_by_indices(train_data, cache["few_shot_indices"])
            few_shot_indices = list(cache["few_shot_indices"])
    else:

        should_save = False
        if args.dry_run:

            criteria = DRY_RUN_PLACEHOLDER_CRITERIA
            should_save = True
            logging.info("dry-run 模式：使用占位 criteria（不调用 API）")
        elif todo:
            client = _make_api_client(cfg)
            criteria = generate_criteria(
                client=client,
                model_name=model_name,
                init_shots=init_shots,
                temperature=init_temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                enable_thinking=enable_thinking,
                base_url=base_url,
            )
            should_save = True
            logging.info("criteria 生成完成，长度=%d 字符", len(criteria))
        else:
            logging.info("无待预测样本且无缓存，跳过 criteria 生成")

        if should_save:

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            cache = {
                "config_snapshot": config_snapshot,
                "init_shot_indices": sorted(init_shot_indices),
                "few_shot_indices": few_shot_indices,
                "criteria": criteria,
                "generated_at": datetime.now().isoformat(),
                "base_url": cfg.get("base_url", ""),
            }
            if args.dry_run:
                cache["dry_run"] = True
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            logging.info("criteria 已缓存到: %s", cache_path)


    _log_messages_once(
        init_shots=init_shots,
        few_shots=few_shots,
        todo=todo,
        criteria=criteria,
    )


    if not args.dry_run and todo and client is None:
        client = _make_api_client(cfg)

    completed = 0
    total_todo = len(todo)
    for i, item in enumerate(todo, 1):
        idx = item["idx"]
        if args.dry_run:
            raw = _predict_one_dry_run(item)
        else:
            messages = build_messages(few_shots=few_shots, current_item=item, criteria=criteria)
            raw = predict_single(
                client=client,
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                enable_thinking=enable_thinking,
                base_url=base_url,
            )
        new_item = deepcopy(item)
        new_item["rhetoricItems_pred"] = raw

        if criteria is not None:
            new_item["criteria"] = criteria
        existing[idx] = new_item
        _save_predictions(existing, pred_path, sort_output=args.sort_output)
        completed += 1
        logging.info("进度: [%d/%d] idx=%d ... 完成", i, total_todo, idx)

    logging.info(
        "本批次完成，共预测 %d 条，结果已保存到 %s", completed, pred_path,
    )


    total_split_size = len(eval_data)
    all_done = total_split_size > 0 and len(existing) >= total_split_size
    if existing:

        predictions: List[Dict[str, Any]] = []
        gold_for_eval: List[Dict[str, Any]] = []
        for it in eval_data:
            if it["idx"] in existing:
                predictions.append(existing[it["idx"]])
                gold_for_eval.append(it)

        eval_modes = cfg.get("eval_modes")
        metrics = evaluate_all(predictions, gold_for_eval, eval_modes=eval_modes)

        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logging.info(
            "metrics 已保存: %s (%s)",
            metrics_path,
            "全部样本完成" if all_done else f"部分完成 {len(existing)}/{total_split_size}",
        )


        exp_suffix = _build_experiment_suffix(
            model_name=model_name,
            seed=seed,
            num_init_shots=n_init_shots,
            num_few_shots=n_shots,
            temperature=temperature,
            init_temperature=init_temperature,
            enable_thinking=enable_thinking,
        )

        metrics_archive_dir = os.path.join(output_dir, "metrics")
        os.makedirs(metrics_archive_dir, exist_ok=True)
        metrics_archive_path = os.path.join(
            metrics_archive_dir, f"metrics_{exp_suffix}.json"
        )
        with open(metrics_archive_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logging.info("metrics 归档副本: %s", metrics_archive_path)


        pred_archive_dir = os.path.join(output_dir, "predictions")
        os.makedirs(pred_archive_dir, exist_ok=True)
        pred_archive_path = os.path.join(
            pred_archive_dir, f"predictions_{exp_suffix}.json"
        )

        shutil.copy2(pred_path, pred_archive_path)
        logging.info("predictions 归档副本: %s", pred_archive_path)


        logging.info("-" * 60)
        logging.info("评测摘要:")
        logging.info("  total_samples   = %d", metrics["total_samples"])
        logging.info("  parse_successes = %d", metrics["parse_successes"])
        logging.info("  parse_failures  = %d", metrics["parse_failures"])
        if metrics.get("parse_failures_idxes"):
            logging.info("  parse_failures_idxes = %s", metrics["parse_failures_idxes"])
        for mode, mode_metrics in metrics.get("results", {}).items():
            logging.info("  [%s] location     = %s", mode,
                         json.dumps(mode_metrics.get("location"), ensure_ascii=False))
            logging.info("  [%s] classification = %s", mode,
                         json.dumps(mode_metrics.get("classification"), ensure_ascii=False))
            logging.info("  [%s] component    = %s", mode,
                         json.dumps(mode_metrics.get("component"), ensure_ascii=False))
            if "krippendorff_alpha" in mode_metrics:
                logging.info("  [%s] krippendorff_alpha = %s", mode,
                             json.dumps(mode_metrics["krippendorff_alpha"], ensure_ascii=False))
        logging.info("-" * 60)
    else:
        logging.info("当前无预测结果，跳过评测")

    return 0


if __name__ == "__main__":
    sys.exit(main())
