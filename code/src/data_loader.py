"""
数据加载模块。

职责:
    - 从 datasets_split/ 读取 train/validation/test 三种切分的 JSON 数据
    - 为 few-shot prompt 抽样训练集样本，并清理无关字段
    - 读取 YAML 配置文件

数据 schema (单条样本)::

    {
        "idx": int,                       # 全局样本下标
        "essay_idx": int,                 # 原作文下标
        "document": str,                  # 作文全文
        "sentenceList": list[str],        # 分句列表
        "grade": int,                     # 年级
        "teacherReview": dict,            # 教师评分
        "rhetoricItems": list[dict]       # 修辞标注，详见下
    }

rhetoricItems 单条::

    {
        "sentenceIdx": list[int],         # 句组下标
        "rhetoricList": list[dict]        # 该句组内的修辞实体
    }

rhetoricList 单条::

    {
        "rhetoric": str, "form": str, "content": str,
        "conjunction": list[str], "conjunctionBeginIdx": list[int], "conjunctionEndIdx": list[int],
        "tenor": list[str], "tenorBeginIdx": list[int], "tenorEndIdx": list[int],
        "vehicle": list[str], "vehicleBeginIdx": list[int], "vehicleEndIdx": list[int]
    }

"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

import yaml


DROP_FIELDS_FOR_FEWSHOT: tuple[str, ...] = ("idx", "essay_idx", "grade", "teacherReview")


def _clean_fewshot_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """剔除单条样本中不需要送入 prompt 的字段，返回新的 dict。

    统一 :func:`sample_few_shots` / :func:`sample_init_shots` /
    :func:`extract_shots_by_indices` 的字段清理逻辑，避免重复代码。
    """
    return {k: v for k, v in item.items() if k not in DROP_FIELDS_FOR_FEWSHOT}


def load_split(data_dir: str, split_name: str) -> List[Dict[str, Any]]:
    """读取指定切分的数据集 JSON。

    :param data_dir: 数据目录，应包含 ``train.json`` / ``validation.json`` / ``test.json``
    :param split_name: 切分名，如 ``"train"`` / ``"validation"`` / ``"test"``
    :return: list[dict]，每条 dict 满足上述数据 schema
    :raises FileNotFoundError: 当目标 JSON 文件不存在时
    """
    if split_name not in ("train", "validation", "test"):
        logging.warning(
            "load_split 收到非标准 split_name=%r，仍尝试按 {data_dir}/{split_name}.json 读取", split_name
        )

    path = os.path.join(data_dir, f"{split_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据集文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"数据集顶层应为 list，实际为 {type(data).__name__}: {path}")

    logging.info("已加载 %s 切分: %d 条样本 (path=%s)", split_name, len(data), path)
    return data


def sample_few_shots(
    train_data: List[Dict[str, Any]],
    n: int = 15,
    seed: int = 42,
    exclude_indices: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """从训练集中随机抽 n 条作为 few-shot 示例，按 idx 排序后剔除无关字段。

    采样策略：
        - 使用 ``random.sample`` 在原始 list 上做无放回采样
        - 采样后按 idx 排序
        - 通过 seed 控制可复现性

    :param train_data: 训练集 list[dict]
    :param n: 采样条数，默认 15（对齐旧项目 few-shot 配置）
    :param seed: 随机种子，默认 42
    :param exclude_indices: 可选的 idx 集合，这些样本不参与采样（用于排除 init_shots）
    :return: list[dict]，每条只保留 ``document`` / ``sentenceList`` / ``rhetoricItems``
    """
    if n <= 0:
        raise ValueError(f"n 必须为正整数，收到 {n}")


    pool = train_data
    if exclude_indices:
        pool = [item for item in train_data if item.get("idx") not in exclude_indices]

    if n > len(pool):
        raise ValueError(f"n={n} 超过可用训练集规模 {len(pool)}")

    rng = random.Random(seed)
    sampled = rng.sample(pool, k=n)

    sampled.sort(key=lambda x: x.get("idx", 0))

    cleaned = [_clean_fewshot_fields(item) for item in sampled]

    logging.info("已采样 %d 条 few-shot (seed=%d)，剔除字段 %s", n, seed, DROP_FIELDS_FOR_FEWSHOT)
    return cleaned


def sample_init_shots(
    train_data: List[Dict[str, Any]],
    n: int = 30,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """从训练集中随机抽 n 条作为 init_shots，用于生成 criteria。

        - 从 train_data 中采样 30 条作为 init_shots
        - 采样后按 idx 排序
        - 剔除无关字段（idx / essay_idx / grade / teacherReview）
        - init_shots 与 few_shots 不重叠（通过返回 idx 集合供调用方排除）

    :param train_data: 训练集 list[dict]
    :param n: 采样条数，默认 30
    :param seed: 随机种子，默认 42（使用不同的内部偏移以避免与 few_shots 完全重合）
    :return: list[dict]，每条只保留 ``document`` / ``sentenceList`` / ``rhetoricItems``
    """
    if n <= 0:
        raise ValueError(f"n 必须为正整数，收到 {n}")
    if n > len(train_data):
        raise ValueError(f"n={n} 超过训练集规模 {len(train_data)}")


    rng = random.Random(seed + 1000)
    sampled = rng.sample(train_data, k=n)

    sampled.sort(key=lambda x: x.get("idx", 0))

    cleaned = [_clean_fewshot_fields(item) for item in sampled]

    logging.info("已采样 %d 条 init_shots (seed=%d+1000)，剔除字段 %s", n, seed, DROP_FIELDS_FOR_FEWSHOT)
    return cleaned


def get_init_shot_indices(
    train_data: List[Dict[str, Any]],
    n: int = 30,
    seed: int = 42,
) -> set:
    """获取 init_shots 对应的 idx 集合（用于 few_shots 采样时排除）。

    与 :func:`sample_init_shots` 使用相同的随机策略，保证返回的 idx 集合一致。

    :param train_data: 训练集 list[dict]
    :param n: 采样条数，默认 30
    :param seed: 随机种子
    :return: init_shots 的 idx 集合
    """
    rng = random.Random(seed + 1000)
    sampled = rng.sample(train_data, k=n)
    return {item["idx"] for item in sampled if "idx" in item}


def get_few_shot_indices(
    train_data: List[Dict[str, Any]],
    n: int = 15,
    seed: int = 42,
    exclude_indices: Optional[set] = None,
) -> List[int]:
    """获取 few_shots 对应的 idx 列表（升序），采样策略与 :func:`sample_few_shots` 一致。

    用于在生成 criteria 缓存时记录 few_shots 使用的样本 idx，便于复查与精确恢复。

    :param train_data: 训练集 list[dict]
    :param n: 采样条数，默认 15
    :param seed: 随机种子，默认 42
    :param exclude_indices: 可选的 idx 集合（如 init_shots），这些样本不参与采样
    :return: 升序排列的 few_shots idx 列表
    """
    if n <= 0:
        raise ValueError(f"n 必须为正整数，收到 {n}")

    pool = train_data
    if exclude_indices:
        pool = [item for item in train_data if item.get("idx") not in exclude_indices]
    if n > len(pool):
        raise ValueError(f"n={n} 超过可用训练集规模 {len(pool)}")

    rng = random.Random(seed)
    sampled = rng.sample(pool, k=n)

    indices = sorted(item["idx"] for item in sampled if "idx" in item)
    return indices


def extract_shots_by_indices(
    train_data: List[Dict[str, Any]],
    indices: List[int],
) -> List[Dict[str, Any]]:
    """根据给定的 idx 列表从训练集中精确提取样本，按 idx 升序并剔除无关字段。

    用于加载 criteria 缓存时，根据缓存中记录的 ``few_shot_indices`` 恢复出与
    criteria 生成时完全一致的 few_shots（而非重新随机采样）。

    :param train_data: 训练集 list[dict]
    :param indices: 目标样本 idx 列表
    :return: list[dict]，按 idx 升序，每条只保留 ``document`` / ``sentenceList`` / ``rhetoricItems``
    :raises ValueError: 当某个 idx 在 train_data 中找不到时
    """
    idx_set = set(indices)
    matched = [item for item in train_data if item.get("idx") in idx_set]
    matched.sort(key=lambda x: x.get("idx", 0))

    found = {item["idx"] for item in matched}
    missing = idx_set - found
    if missing:
        raise ValueError(
            f"以下 idx 在训练集中未找到，无法从缓存恢复样本: {sorted(missing)}"
        )

    cleaned = [_clean_fewshot_fields(item) for item in matched]
    logging.info("已从缓存 idx 精确恢复 %d 条样本（剔除字段 %s）", len(cleaned), DROP_FIELDS_FOR_FEWSHOT)
    return cleaned


def load_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件。

    :param config_path: ``config.yaml`` 的绝对或相对路径
    :return: dict，配置内容
    :raises FileNotFoundError: 文件不存在
    :raises yaml.YAMLError: YAML 解析失败
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise TypeError(f"配置文件顶层应为 mapping，实际为 {type(cfg).__name__}: {config_path}")

    logging.info("已加载配置 %s (keys=%s)", config_path, list(cfg.keys()))
    return cfg


def resolve_data_dir(config_path: str, data_dir: str) -> str:
    """将配置中的相对 data_dir 解析为绝对路径。

    规则：若 data_dir 已是绝对路径则原样返回；否则按 config 文件所在目录的上级
    （即 ``new-code/`` 根）拼接。

    :param config_path: 配置文件路径，用于推断项目根
    :param data_dir: 配置中的 data_dir 字段
    :return: 解析后的绝对路径
    """
    if os.path.isabs(data_dir):
        return data_dir
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    return os.path.normpath(os.path.join(project_root, data_dir))


def resolve_output_dir(config_path: str, output_dir: str) -> str:
    """将配置中的相对 output_dir 解析为绝对路径，规则同 :func:`resolve_data_dir`。"""
    if os.path.isabs(output_dir):
        return output_dir
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    return os.path.normpath(os.path.join(project_root, output_dir))


def get_item_signature(item: Dict[str, Any]) -> Optional[str]:
    """为单条样本生成稳定签名字符串，用于日志与去重。"""
    if "idx" not in item:
        return None
    return f"idx={item['idx']}/essay_idx={item.get('essay_idx')}"


def create_or_load_batches(
    data_dir: str,
    split_name: str,
    batch_size: int = 50,
    seed: int = 42,
    batch_file: Optional[str] = None,
) -> Dict[str, Any]:
    """创建或加载测试集批次划分。

    将 ``{data_dir}/{split_name}.json`` 中的样本按固定 seed 随机打乱后，
    每 ``batch_size`` 条划分为一批（尾部不满 batch_size 也自成一组）。

    - 若 ``batch_file`` 已存在，则直接加载返回，保证后续实验复用同一划分；
    - 若不存在，则基于当前 split 数据实时生成，并写入 ``batch_file`` 供后续复用。

    批次文件 JSON 结构::

        {
            "seed": 42,
            "batch_size": 50,
            "total_samples": 511,
            "num_batches": 11,
            "split": "test",
            "batches": [
                {"batch_id": 0, "indices": [<idx>, ...]},
                ...
            ]
        }

    其中 ``indices`` 存放的是数据集中每条样本自带的 ``idx`` 字段值（全局编号），
    便于跨文件/跨运行定位样本。

    :param data_dir: 数据集目录，需包含 ``{split_name}.json``
    :param split_name: 切分名，如 ``"test"`` / ``"validation"`` / ``"train"``
    :param batch_size: 每批样本数，默认 50
    :param seed: 随机种子，默认 42
    :param batch_file: 批次文件路径；为 None 时不生成默认路径，由调用方决定
    :return: dict，批次划分信息（见上述 JSON 结构）
    :raises ValueError: ``batch_size <= 0`` 或已有文件的 seed/batch_size 与本次调用不一致
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，收到 {batch_size}")


    if batch_file and os.path.exists(batch_file):
        with open(batch_file, "r", encoding="utf-8") as f:
            info = json.load(f)

        if int(info.get("seed", seed)) != seed:
            logging.warning(
                "批次文件 %s 的 seed=%s 与本次 seed=%s 不一致，仍按已有文件复用",
                batch_file, info.get("seed"), seed,
            )
        if int(info.get("batch_size", batch_size)) != batch_size:
            logging.warning(
                "批次文件 %s 的 batch_size=%s 与本次 batch_size=%s 不一致，仍按已有文件复用",
                batch_file, info.get("batch_size"), batch_size,
            )
        logging.info(
            "已加载批次划分: %s (total=%d, batches=%d)",
            batch_file, info.get("total_samples"), info.get("num_batches"),
        )
        return info


    data = load_split(data_dir, split_name)
    idx_list = [item["idx"] for item in data]
    rng = random.Random(seed)
    rng.shuffle(idx_list)

    batches: List[Dict[str, Any]] = []
    for batch_id, start in enumerate(range(0, len(idx_list), batch_size)):
        batches.append({
            "batch_id": batch_id,
            "indices": idx_list[start: start + batch_size],
        })

    info = {
        "seed": seed,
        "batch_size": batch_size,
        "total_samples": len(idx_list),
        "num_batches": len(batches),
        "split": split_name,
        "batches": batches,
    }

    if batch_file:
        os.makedirs(os.path.dirname(os.path.abspath(batch_file)) or ".", exist_ok=True)
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        logging.info(
            "已生成批次划分: %s (total=%d, batches=%d, batch_size=%d, seed=%d)",
            batch_file, len(idx_list), len(batches), batch_size, seed,
        )
    return info


__all__ = [
    "DROP_FIELDS_FOR_FEWSHOT",
    "load_split",
    "sample_few_shots",
    "sample_init_shots",
    "get_init_shot_indices",
    "get_few_shot_indices",
    "extract_shots_by_indices",
    "load_config",
    "resolve_data_dir",
    "resolve_output_dir",
    "get_item_signature",
    "create_or_load_batches",
]
