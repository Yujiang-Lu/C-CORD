#!/usr/bin/env python3
"""备份实验结果文件到带时间戳的子目录中，防止被后续操作覆盖。"""

import argparse
import json
import os
import shutil
from datetime import datetime


IGNORE_PATTERNS = shutil.ignore_patterns("._*", ".DS_Store", "__pycache__")


def is_junk_file(name: str) -> bool:
    """判断是否为系统元数据垃圾文件。"""
    return name.startswith("._") or name == ".DS_Store"


def human_readable_size(size_bytes: int) -> str:
    """将字节数转换为人类可读的格式。"""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def collect_source_items(results_dir: str) -> list:
    """收集需要备份的文件和目录列表（仅返回存在的项）。"""
    candidates = [
        "metrics.json",
        "predictions.json",
        "metrics",
        "predictions",
        "cache",
    ]
    items = []
    for name in candidates:
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            items.append(name)
    return items


def count_files_and_size(directory: str) -> tuple:
    """递归统计目录下的文件数量和总大小。"""
    total_files = 0
    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(directory):
        for f in filenames:
            if is_junk_file(f):
                continue
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_files += 1
                total_size += os.path.getsize(fp)
    return total_files, total_size


def main():
    parser = argparse.ArgumentParser(description="备份实验结果文件")
    parser.add_argument(
        "--results-dir",
        default="results/exp01",
        help="结果目录路径（默认: results/exp01）",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="可选标签，追加到备份目录名中",
    )
    args = parser.parse_args()

    results_dir = args.results_dir

    if not os.path.isdir(results_dir):
        print(f"错误：结果目录不存在: {results_dir}")
        return


    source_items = collect_source_items(results_dir)

    if not source_items:
        print(f"提示：{results_dir} 中没有找到需要备份的文件或目录，跳过备份。")
        print(f"  检查的目标: metrics.json, predictions.json, metrics/, predictions/, cache/")
        return


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backup_{timestamp}"
    if args.label:
        backup_name = f"{backup_name}_{args.label}"

    backup_base = os.path.join(results_dir, "backup")
    backup_dir = os.path.join(backup_base, backup_name)


    os.makedirs(backup_dir, exist_ok=True)


    copied_items = []
    for item in source_items:
        src_path = os.path.join(results_dir, item)
        dst_path = os.path.join(backup_dir, item)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, ignore=IGNORE_PATTERNS)
            copied_items.append({"name": item, "type": "directory"})
        else:
            shutil.copy2(src_path, dst_path)
            copied_items.append({"name": item, "type": "file"})


    file_count, total_size = count_files_and_size(backup_dir)


    backup_info = {
        "backup_time": datetime.now().isoformat(),
        "source_dir": os.path.abspath(results_dir),
        "backup_dir": os.path.abspath(backup_dir),
        "items": copied_items,
        "total_files": file_count,
        "total_size_bytes": total_size,
        "total_size_human": human_readable_size(total_size),
    }
    info_path = os.path.join(backup_dir, "backup_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(backup_info, f, ensure_ascii=False, indent=2)


    print("备份完成！")
    print(f"  源目录: {results_dir}/")
    print(f"  备份到: {backup_dir}/")
    print(f"  文件数: {file_count}")
    print(f"  总大小: {human_readable_size(total_size)}")
    if args.label:
        print(f"  标签: {args.label}")


if __name__ == "__main__":
    main()
