"""
中文作文修辞识别评测项目 - 核心模块包

模块构成:
    - data_loader: 数据集加载与 few-shot 采样
    - model_api:   OpenAI 兼容 API 封装与 prompt 构造
    - evaluator:   多级容错解析与 IoU/F1/成分抽取指标
    - main:        CLI 入口与 dry-run 管线
"""
