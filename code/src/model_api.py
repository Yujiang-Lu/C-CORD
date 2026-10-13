"""
大模型 API 封装与 prompt 构造模块。

职责:
    - 维护中文作文修辞识别任务的标签体系（RhetoricType / FormType / ContentType / ComponentType）
    - 构造 OpenAI chat 格式的多轮 prompt（system + 15-shot 多轮对话 + 当前样本）
    - 支持两阶段流程：init prompt 生成 criteria → task prompt 注入 criteria
    - 通过 OpenAI 兼容 API 调用大模型，支持指数退避重试
    - 批量预测并返回 list[dict]
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


RHETORIC_LABELS: tuple[str, ...] = (
    "比喻", "比拟", "夸张", "排比", "反复",
    "设问", "反问", "摹状", "引用", "对偶",
)


FORM_LABELS: tuple[str, ...] = (
    "比喻_明喻", "比喻_暗喻", "比喻_借喻",
    "比拟_名词", "比拟_动词", "比拟_形容词",
    "夸张_直接夸张", "夸张_间接夸张",
    "排比_成分排比", "排比_句子排比",
    "反复_连续反复", "反复_间隔反复",
    "设问_问答连属", "设问_问答不连属",
    "反问_单句反问", "反问_复句反问",
    "摹状_通感", "摹状_直感",
    "引用_明引", "引用_暗引", "引用_转引",
    "对偶_正对", "对偶_反对", "对偶_串对",
)


CONTENT_LABELS: tuple[str, ...] = (
    "比喻_实在物", "比喻_动作", "比喻_抽象概念",
    "比拟_拟人", "比拟_拟物",
    "夸张_扩大夸张", "夸张_缩小夸张", "夸张_超前夸张",
    "排比_并列", "排比_承接", "排比_递进",
    "反复_词语", "反复_句子",
    "设问_问而不答", "设问_一问一答", "设问_数问一答", "设问_一问数答",
    "反问_肯定反问", "反问_否定反问",
    "摹状_摹视", "摹状_摹听", "摹状_摹嗅", "摹状_摹触", "摹状_摹味",
    "引用_话语", "引用_引典",
    "对偶_工整对", "对偶_宽松对",
)


COMPONENT_LABELS: Dict[str, tuple[str, str, str]] = {
    "比喻_明喻": ("明喻喻词", "本体", "喻体"),
    "比喻_暗喻": ("暗喻喻词/无", "本体", "喻体"),
    "比喻_借喻": ("无", "无", "喻体"),
    "比拟_名词": ("无", "本体", "比拟内容"),
    "比拟_动词": ("无", "本体", "比拟内容"),
    "比拟_形容词": ("无", "本体", "比拟内容"),
    "夸张_直接夸张": ("无", "夸张对象", "夸张内容"),
    "夸张_间接夸张": ("无", "夸张对象", "夸张内容"),
    "排比_成分排比": ("提挈语/排比结构", "无", "排比内容"),
    "排比_句子排比": ("提挈语/排比结构", "无", "排比内容"),
    "反复_连续反复": ("提挈语/反复句式", "无", "反复内容"),
    "反复_间隔反复": ("提挈语/反复句式", "无", "反复内容"),
    "设问_问答连属": ("无", "提问", "回答"),
    "设问_问答不连属": ("无", "提问", "回答/无"),
    "反问_单句反问": ("无", "反问对象", "反问内容"),
    "反问_复句反问": ("无", "反问对象", "反问内容"),
    "摹状_通感": ("摹状词", "目标感官", "摹状感官"),
    "摹状_直感": ("摹状词", "目标感官", "摹状感官"),
    "引用_明引": ("引用出处", "引用对象/无", "引用内容"),
    "引用_暗引": ("无", "引用对象/无", "引用内容"),
    "引用_转引": ("引用出处", "引用对象/无", "引用内容"),
    "对偶_正对": ("无", "出句词", "对句词"),
    "对偶_反对": ("无", "出句词", "对句词"),
    "对偶_串对": ("无", "出句词", "对句词"),
}


def _format_label_list(labels: tuple[str, ...]) -> str:
    """将标签元组格式化为 JSON list 字符串，便于注入 prompt。"""
    return json.dumps(list(labels), ensure_ascii=False)


def _format_component_labels() -> str:
    """将 COMPONENT_LABELS 格式化为 JSON dict 字符串，便于注入 prompt。

    """

    serializable = {k: list(v) for k, v in COMPONENT_LABELS.items()}
    return json.dumps(serializable, ensure_ascii=False)


_SYSTEM_TEMPLATE_LINES: tuple[str, ...] = (
    "你是一个对中文作文进行修辞学分析的资深专家，精通于感知并拆解文本修辞，理解每个修辞实体的修辞成分。",
    "### 修辞类型",
    "- 修辞粗粒度类型包含以下几种: {rhetoric_labels}。",
    "- 对应各修辞粗粒度类型，修辞细粒度形式类型包含以下几种：{form_labels}。",
    "- 对应各修辞粗粒度类型，修辞细粒度内容类型包含以下几种：{content_labels}。",
    "### 修辞成分",
    "修辞成分是三个生成式命名实体(关系三元组)：连接词(conjunction)，描写对象(tenor)，描写内容(vehicle)。",
    "对应各细粒度形式类型，修辞成分定义如下：{component_labels}。",
    "### 任务介绍",
    "对于给定的中小学生中文作文数据，对其中出现的每个修辞实体按以下 JSON 格式进行标注：",
    "{{",
    '    "rhetoric": "一个字符串，表示修辞粗粒度类型。",',
    '    "form": "一个字符串，表示修辞细粒度形式类型。",',
    '    "content": "一个字符串，表示修辞细粒度内容类型。",',
    '    "conjunction": [一个列表，表示修辞成分"连接词"。],',
    '    "conjunctionBeginIdx": [一个列表，表示修辞成分连接词在句组文本中的起始字符下标。],',
    '    "conjunctionEndIdx": [一个列表，表示修辞成分连接词在句组文本中的结束字符下标。],',
    '    "tenor": [一个列表，表示修辞成分"描写对象"。],',
    '    "tenorBeginIdx": [一个列表，表示修辞成分描写对象在句组文本中的起始字符下标。],',
    '    "tenorEndIdx": [一个列表，表示修辞成分描写对象在句组文本中的结束字符下标。],',
    '    "vehicle": [一个列表，表示修辞成分"描写内容"。],',
    '    "vehicleBeginIdx": [一个列表，表示修辞成分描写内容在句组文本中的起始字符下标。],',
    '    "vehicleEndIdx": [一个列表，表示修辞成分描写内容在句组文本中的结束字符下标。]',
    "}}",
    "### 注意",
    "- 句组为 sentenceList 中句子的任意连续的顺序组合，可能存在多个修辞句组，一个修辞句组可能有多个修辞实体。",
    "- 句组文本的字符下标从 0 开始。",
    "- 输出必须是严格的 JSON 字典，且键名顺序为 rhetoricItems。",
)


def _build_system_content(criteria: Optional[str] = None) -> str:


    lines: List[str] = []
    for line in _SYSTEM_TEMPLATE_LINES:

        if line == "### 任务介绍" and criteria is not None:
            lines.append("### 修辞标准")
            lines.append(json.dumps(criteria, ensure_ascii=False))
        lines.append(
            line.format(
                rhetoric_labels=_format_label_list(RHETORIC_LABELS),
                form_labels=_format_label_list(FORM_LABELS),
                content_labels=_format_label_list(CONTENT_LABELS),
                component_labels=_format_component_labels(),
            )
        )
    return "\n".join(lines)


def _build_user_question(document: str, sentence_list: List[str]) -> str:

    return "\n".join([
        "### 中小学生中文作文数据",
        "{",
        f'    "document": {json.dumps(document, ensure_ascii=False)},',
        f'    "sentenceList": {json.dumps(sentence_list, ensure_ascii=False)}',
        "}",
        "### 输出要求",
        "严格使用以下 JSON 格式，要求注意 JSON 格式的完整性：",
        "{",
        '    "rhetoricItems": [列表中存放你对给定中小学生中文作文数据的修辞理解标注结果。]',
        "}",
    ])


def _build_assistant_answer(rhetoric_items: List[Dict[str, Any]]) -> str:

    return "\n".join([
        "{",
        f'    "rhetoricItems": {json.dumps(rhetoric_items, ensure_ascii=False)}',
        "}",
    ])


def build_messages(
    few_shots: List[Dict[str, Any]],
    current_item: Dict[str, Any],
    criteria: Optional[str] = None,
) -> List[Dict[str, str]]:
    """构造 OpenAI chat messages 格式。

    结构:
        1. ``system``: 任务描述 + 全部标签枚举 + component 定义 + criteria + JSON 输出要求
        2. ``user``/``assistant`` 交替 n 轮: few-shot 示例（n = ``len(few_shots)``）
        3. 末尾一条 ``user``: 当前待预测样本

    :param few_shots: few-shot 示例 list[dict]，每条需含 ``document`` / ``sentenceList`` / ``rhetoricItems``
    :param current_item: 当前测试样本，需含 ``document`` / ``sentenceList``
    :param criteria: 可选的修辞标注规范字符串（由 generate_criteria 生成）
    :return: list[dict]，可直接传给 ``openai.ChatCompletion.create(messages=...)``
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _build_system_content(criteria=criteria)}
    ]

    for shot in few_shots:
        messages.append({
            "role": "user",
            "content": _build_user_question(shot["document"], shot["sentenceList"]),
        })
        messages.append({
            "role": "assistant",
            "content": _build_assistant_answer(shot.get("rhetoricItems", [])),
        })

    messages.append({
        "role": "user",
        "content": _build_user_question(current_item["document"], current_item["sentenceList"]),
    })
    return messages


def build_init_messages(init_shots: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """构造 init prompt 的 messages，用于让模型总结修辞标注规范。

    :param init_shots: 初始化样本 list[dict]，每条含 ``document`` / ``sentenceList`` / ``rhetoricItems``
    :return: list[dict]，OpenAI chat messages 格式
    """

    system_content = "\n".join([
        "你是一个对中文作文进行修辞学分析的资深专家，精通于感知并拆解文本修辞，理解每个修辞实体的修辞成分。",
        "### 修辞类型",
        f"- 修辞粗粒度类型包含以下几种: {_format_label_list(RHETORIC_LABELS)}。",
        f"- 对应各修辞粗粒度类型，修辞细粒度形式类型包含以下几种：{_format_label_list(FORM_LABELS)}。",
        f"- 对应各修辞粗粒度类型，修辞细粒度内容类型包含以下几种：{_format_label_list(CONTENT_LABELS)}。",
        "### 修辞成分",
        "修辞成分是三个生成式命名实体(关系三元组)：连接词(conjunction)，描写对象(tenor)，描写内容(vehicle)。",
        f"对应各细粒度形式类型，修辞成分定义如下：{_format_component_labels()}。",
    ])


    user_parts: List[str] = [
        "### 任务介绍",
        "你的任务是参考以下样例，生成每种修辞类型的说明和特征、每种修辞类型对应修辞成分的说明和特征，"
        "以确定不同修辞类型之间的边界、不同修辞类型对应修辞成分之间的边界。"
        "你需要在生成内容中给出一致性高的修辞标注规范与数据规范标准。",
        "### 样例",
        "每一个样例由文档(document)、句子列表(sentenceList)与理解结果(rhetoricItems)三部分构成。",
    ]
    for i, shot in enumerate(init_shots):
        user_parts.append(f"- 样例 {i}: {json.dumps(shot, ensure_ascii=False)}")

    user_parts.extend([
        "",
        "### 输出要求",
        "严格使用以下JSON格式，要求注意JSON格式的完整性：",
        "{",
        '    "criteria": "一个字符串，表示你给出的每种修辞类型的说明和特征、每种修辞类型对应修辞成分的说明和特征，'
        '即一致性高的修辞标注规范与数据规范标准。你的输出应尽可能详细。在你的输出中，不应该含有上述样例的任何引用。"',
        "}",
    ])

    user_content = "\n".join(user_parts)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def generate_criteria(
    client: Any,
    model_name: str,
    init_shots: List[Dict[str, Any]],
    temperature: float = 1.0,
    max_tokens: int = 8192,
    max_retries: int = 3,
    retry_backoff: float = 1.5,
    enable_thinking: Optional[bool] = None,
    base_url: str = "",
) -> str:
    """调用 LLM 生成修辞标注规范（criteria）。

    :param client: ``openai.OpenAI`` 实例
    :param model_name: 模型名
    :param init_shots: 初始化样本（~30 条）
    :param temperature: 采样温度
    :param max_tokens: 最大生成 token 数
    :param max_retries: 最大重试次数
    :param retry_backoff: 指数退避底数
    :param enable_thinking: 思考模式开关（``True`` / ``False`` / ``None``），透传给 ``predict_single``
    :param base_url: API 的 base URL，用于自动识别模型厂商，透传给 ``predict_single``
    :return: criteria 字符串
    """
    messages = build_init_messages(init_shots)
    logging.info("正在生成 criteria（init_shots=%d 条）...", len(init_shots))

    raw = predict_single(
        client=client,
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        enable_thinking=enable_thinking,
        base_url=base_url,
    )


    try:
        criteria = json.loads(raw)["criteria"]
        logging.info("criteria 生成成功（JSON 解析），长度=%d 字符", len(criteria))
        return criteria
    except (json.JSONDecodeError, KeyError, TypeError):
        pass


    try:
        left = raw.find("{")
        right = raw.rfind("}")
        if left != -1 and right > left:
            candidate = raw[left:right + 1]
            criteria = json.loads(candidate)["criteria"]
            logging.info("criteria 生成成功（截取 {} 后解析），长度=%d 字符", len(criteria))
            return criteria
    except (json.JSONDecodeError, KeyError, TypeError):
        pass


    logging.warning("criteria JSON 解析失败，使用原始 response 文本作为 criteria（长度=%d）", len(raw))
    return raw


def detect_provider(base_url: str, model_name: str) -> str:
    """根据模型名称（优先）和 base_url（兜底）自动识别模型厂商。

    识别优先级：model_name > base_url
    支持的厂商：dashscope, deepseek, openai, gemini, anthropic

    设计原因：用户使用 ChatAnyWhere 等 API 代理转发所有海外模型请求，
    base_url 无法区分 GPT/Gemini/Claude，模型名称才是最可靠的厂商标识。

    :param base_url: OpenAI 兼容 API 的 base URL（兜底依据）
    :param model_name: 模型名称（优先依据）
    :return: 厂商标识字符串
    """
    model_lower = (model_name or "").lower()
    base_url_lower = (base_url or "").lower()


    if model_lower.startswith("qwen"):
        return "dashscope"
    if model_lower.startswith("deepseek"):
        return "deepseek"
    if model_lower.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if model_lower.startswith("gemini"):
        return "gemini"
    if model_lower.startswith("claude"):
        return "anthropic"
    if model_lower.startswith(("glm", "zhipu")):
        return "glm"
    if model_lower.startswith(("kimi", "moonshot")):
        return "kimi"


    if "dashscope" in base_url_lower or "aliyuncs" in base_url_lower or "maas" in base_url_lower:
        return "dashscope"
    if "deepseek" in base_url_lower:
        return "deepseek"
    if "openai" in base_url_lower:
        return "openai"
    if "google" in base_url_lower or "generativelanguage" in base_url_lower or "vertex" in base_url_lower:
        return "gemini"
    if "anthropic" in base_url_lower:
        return "anthropic"
    if "bigmodel" in base_url_lower or "zhipu" in base_url_lower:
        return "glm"
    if "moonshot" in base_url_lower or "kimi" in base_url_lower:
        return "kimi"


    return "unknown"


def _is_proxy_url(base_url: str) -> bool:
    """判断是否为第三方代理 URL（非厂商官方 endpoint）。

    用途：对第三方代理（如 ChatAnyWhere），某些顶层参数（如 ``reasoning_effort``）
    可能不被支持或透传，需要特殊处理。

    :param base_url: API 的 base URL
    :return: ``True`` 表示是第三方代理，``False`` 表示官方 endpoint 或空 URL
    """
    if not base_url:
        return False
    url_lower = base_url.lower()

    official_patterns = [
        "api.openai.com",
        "dashscope.aliyuncs.com",
        "aliyuncs.com/compatible-mode",
        "maas.aliyuncs.com",
        "api.deepseek.com",
        "generativelanguage.googleapis.com",
        "ai.google.dev",
        "vertexai.googleapis.com",
        "api.anthropic.com",
    ]
    return not any(p in url_lower for p in official_patterns)


def _extract_error_detail(exc: BaseException) -> str:
    """从异常中提取 API 错误详情字符串，用于日志输出。

    兼容 openai SDK（``response.text`` / ``body`` / ``message``）与 httpx 异常。

    :param exc: 捕获的异常对象
    :return: 错误详情字符串（可能为空字符串）
    """
    if hasattr(exc, "response") and hasattr(exc.response, "text"):
        return f" | 响应: {exc.response.text[:500]}"
    if hasattr(exc, "body"):
        body = exc.body
        if isinstance(body, dict):
            err_msg = body.get("error", {})
            if isinstance(err_msg, dict):
                return f" | body: {err_msg.get('message', str(body))[:500]}"
            return f" | body: {str(body)[:500]}"
        return f" | body: {str(body)[:500]}"
    if hasattr(exc, "message") and exc.message:
        return f" | message: {str(exc.message)[:500]}"
    return ""


def _uses_max_completion_tokens(model_name: str) -> bool:
    """判断模型是否使用 max_completion_tokens 替代 max_tokens。

    OpenAI 推理模型（o1/o3/o4）和 GPT-5+ 系列不接受 max_tokens 参数，
    需要改用 max_completion_tokens。

    :param model_name: 模型名称
    :return: ``True`` 表示应使用 ``max_completion_tokens``
    """
    model_lower = model_name.lower()

    if model_lower.startswith(("o1", "o3", "o4")):
        return True

    if model_lower.startswith(("gpt-5", "gpt-6", "gpt-7", "gpt-8", "gpt-9")):
        return True

    return False


def _is_new_claude_model(model_name: str) -> bool:
    """判断是否为新一代 Claude 模型（5.x+，adaptive thinking 系列）。

    新一代模型（如 ``claude-sonnet-5-5``）与旧模型（如 ``claude-haiku-4-5-20251001``）
    的 API 行为差异：

    - 思考控制：新模型用 ``thinking={"type": "adaptive"}`` /
      ``{"type": "between_tools"}`` + ``output_config={"effort": ...}``；
      旧模型用 ``thinking={"type": "enabled", "budget_tokens": N}`` / ``{"type": "disabled"}``。
      ``budget_tokens`` 在新模型上直接返回 400。
    - temperature：新模型不支持该参数，传了会报错。
    - 新模型无法完全关闭思考（最低为 between_tools 最少前置思考）。

    判断方法：用正则从模型名提取主版本号（``claude-\\w+-(\\d+)``），major >= 5 为新模型。

    :param model_name: 模型名称，如 ``"claude-sonnet-5-5"``
    :return: ``True`` 表示新一代 Claude 模型
    """
    model_lower = (model_name or "").lower()
    m = re.search(r"claude-\w+-(\d+)", model_lower)
    if m:
        return int(m.group(1)) >= 5

    return False


def _build_thinking_kwargs(provider: str, model_name: str, enable_thinking: Optional[bool], base_url: str = "") -> dict:
    """根据厂商和 enable_thinking 设置，构造 API 调用所需的额外参数。

    返回的 dict 可直接 ``**unpack`` 到 ``client.chat.completions.create()`` 调用中。

    :param provider: 厂商标识（``detect_provider`` 的返回值）
    :param model_name: 模型名称
    :param enable_thinking: ``True`` 开启 / ``False`` 关闭 / ``None`` 不传
    :param base_url: API 的 base URL，用于判断是否为第三方代理
    :return: 额外参数字典，可能包含 ``extra_body`` 或 ``reasoning_effort``
    """
    if enable_thinking is None:
        return {}

    model_lower = (model_name or "").lower()

    if provider == "dashscope":

        return {"extra_body": {"enable_thinking": enable_thinking}}

    elif provider == "deepseek":

        if enable_thinking:
            return {"extra_body": {"thinking": {"type": "enabled"}}}
        else:

            kwargs = {"extra_body": {"thinking": {"type": "disabled"}}}

            if "reasoner" in model_lower:
                logger.warning(
                    "模型 %s 是专用推理模型，无法关闭思考模式，参数将被忽略", model_name
                )
            return kwargs

    elif provider == "openai":


        non_reasoning_models = ("gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-3.5")
        if any(model_lower.startswith(m) for m in non_reasoning_models):

            if enable_thinking:
                logger.warning(
                    "模型 %s 不支持思考模式，参数将被忽略", model_name
                )
            return {}


        if enable_thinking:
            return {"reasoning_effort": "medium"}
        else:
            return {"reasoning_effort": "none"}

    elif provider == "gemini":


        if "3." in model_lower or "3-" in model_lower:
            if not enable_thinking:
                logger.warning(
                    "模型 %s (Gemini 3.x) 始终开启思考，无法关闭，将使用最低强度 reasoning_effort='low'",
                    model_name,
                )
                return {"reasoning_effort": "low"}
            else:
                return {"reasoning_effort": "high"}

        elif "flash" in model_lower:
            if not enable_thinking:
                return {"extra_body": {"google": {"thinking_config": {"thinking_budget": 0}}}}
            else:
                return {"extra_body": {"google": {"thinking_config": {"thinking_budget": 8192}}}}
        else:

            if not enable_thinking:
                logger.warning(
                    "模型 %s (Gemini 2.5 Pro) 无法完全关闭思考，使用最小 thinking_budget=128",
                    model_name,
                )
                return {"extra_body": {"google": {"thinking_config": {"thinking_budget": 128}}}}
            else:
                return {"extra_body": {"google": {"thinking_config": {"thinking_budget": 8192}}}}

    elif provider == "anthropic":


        if _is_new_claude_model(model_name):


            if enable_thinking:
                return {
                    "extra_body": {
                        "thinking": {"type": "adaptive"},
                        "output_config": {"effort": "high"},
                    }
                }
            else:


                logger.warning(
                    "模型 %s 为新一代 Claude，无法完全关闭思考模式，"
                    "将使用 between_tools 模式（最少前置思考）",
                    model_name,
                )
                return {
                    "extra_body": {
                        "thinking": {"type": "between_tools"},
                        "output_config": {"effort": "low"},
                    }
                }


        is_thinking_model = model_lower.endswith("-thinking") or "-thinking-" in model_lower
        if is_thinking_model and not enable_thinking:
            logger.warning(
                "模型 %s 是专用思考路由模型，无法关闭思考模式，参数将被忽略",
                model_name,
            )
            return {}
        if enable_thinking:


            return {"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 4096}}}
        else:
            return {"extra_body": {"thinking": {"type": "disabled"}}}

    elif provider in ("glm", "kimi"):

        if not enable_thinking:
            logger.warning(
                "模型 %s (%s) 默认开启思考且无法关闭，enable_thinking=false 参数将被忽略",
                model_name, provider.upper(),
            )
        return {}

    else:

        logger.warning(
            "未识别的模型厂商 (provider=%s)，将使用 DashScope 风格的 enable_thinking 参数",
            provider,
        )
        return {"extra_body": {"enable_thinking": enable_thinking}}


def predict_single(
    client: Any,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    max_retries: int = 3,
    retry_backoff: float = 1.5,
    enable_thinking: Optional[bool] = None,
    base_url: str = "",
) -> str:
    """调用 OpenAI 兼容 API 获取单条预测，指数退避重试。

    :param client: ``openai.OpenAI`` 实例（或任何兼容 ``client.chat.completions.create`` 的对象）
    :param model_name: 模型名，如 ``"qwen-plus"`` / ``"deepseek-chat"`` / ``"gpt-4o"``
    :param messages: chat messages list
    :param temperature: 采样温度
    :param max_tokens: 单次最大生成 token 数
    :param max_retries: 最大重试次数（不含首次）
    :param retry_backoff: 指数退避底数，第 i 次重试等待 ``retry_backoff ** i`` 秒
    :param enable_thinking: 思考模式开关。``True`` 强制开启、``False`` 强制关闭；
        ``None`` 时不传递思考参数，使用模型默认行为。
        通过多厂商适配层自动映射为对应 API 参数。
    :param base_url: API 的 base URL，用于自动识别模型厂商
    :return: 模型原始返回字符串（``choices[0].message.content``）
    :raises RuntimeError: 重试 ``max_retries`` 次后仍失败
    """

    provider = detect_provider(base_url, model_name)


    kwargs = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        stream=False,
    )


    if provider == "openai" and _uses_max_completion_tokens(model_name):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    thinking_kwargs = _build_thinking_kwargs(provider, model_name, enable_thinking, base_url)
    kwargs.update(thinking_kwargs)


    if provider == "anthropic" and _is_new_claude_model(model_name):
        kwargs.pop("temperature", None)
        logger.info("新 Claude 模型 %s 不支持 temperature，已跳过", model_name)


    if (
        provider == "anthropic"
        and not _is_new_claude_model(model_name)
        and enable_thinking
        and kwargs.get("temperature") != 1
    ):
        logger.warning(
            "Anthropic 思考模式要求 temperature=1，当前 config 设置为 %.1f，"
            "请在 config.yaml 中手动修改为 1.0，否则 API 将返回 400 错误",
            kwargs.get("temperature", 0.7),
        )


    last_exc: Optional[BaseException] = None
    _reasoning_effort_removed = False
    _max_tokens_fixed = False

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError("API 返回 content 为 None")
            return content
        except Exception as exc:  # noqa: BLE001
            last_exc = exc


            is_bad_request = (
                type(exc).__name__ in ("BadRequestError", "APIStatusError")
                or (hasattr(exc, "status_code") and getattr(exc, "status_code", None) == 400)
            )
            if (
                is_bad_request
                and "reasoning_effort" in kwargs
                and not _reasoning_effort_removed
            ):
                _reasoning_effort_removed = True
                logger.warning(
                    "API 返回 400，可能是代理不支持 reasoning_effort 参数，"
                    "已移除该参数重试。错误详情: %s",
                    _extract_error_detail(exc),
                )
                kwargs.pop("reasoning_effort", None)
                continue


            if is_bad_request and "max_tokens" in kwargs and not _max_tokens_fixed:
                _max_tokens_fixed = True
                logger.warning(
                    "API 返回 400，模型可能不支持 max_tokens 参数，"
                    "已自动换用 max_completion_tokens 重试。"
                    "错误详情: %s",
                    _extract_error_detail(exc),
                )
                max_tokens_val = kwargs.pop("max_tokens")
                kwargs["max_completion_tokens"] = max_tokens_val
                continue


            err_detail = _extract_error_detail(exc)
            if attempt >= max_retries:
                break
            wait = retry_backoff ** attempt
            logger.warning(
                "predict_single 第 %d 次调用失败 (%s)%s，%.2fs 后重试",
                attempt + 1, type(exc).__name__, err_detail, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"predict_single 在 {max_retries + 1} 次尝试后仍失败: {last_exc!r}"
        f"{_extract_error_detail(last_exc)}"
    )


def predict_batch(
    data: List[Dict[str, Any]],
    few_shots: List[Dict[str, Any]],
    config: Dict[str, Any],
    max_samples: Optional[int] = None,
    criteria: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """批量预测，返回带 ``rhetoricItems_pred`` 字段的 list[dict]。

    :param data: 测试集 list[dict]
    :param few_shots: few-shot 示例 list[dict]
    :param config: 配置 dict，需包含 ``model_name`` / ``api_key`` / ``base_url``，
        可选 ``temperature`` / ``max_tokens`` / ``max_retries``
    :param max_samples: 若指定，仅对前 N 条进行预测（用于冒烟测试）
    :param criteria: 可选的修辞标注规范（由 generate_criteria 生成）
    :return: list[dict]，每条在原数据基础上增加 ``rhetoricItems_pred``（模型原始返回字符串）
    """

    import openai


    api_key = os.environ.get("RHETORIC_API_KEY") or config.get("api_key")
    base_url = os.environ.get("RHETORIC_BASE_URL") or config.get("base_url")
    model_name = config["model_name"]
    if not api_key or api_key == "YOUR_API_KEY":
        raise ValueError(
            "API Key 未配置：请设置环境变量 RHETORIC_API_KEY 或在 config['api_key'] 中填入；"
            "如需冒烟测试请使用 --dry-run"
        )

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    temperature = float(config.get("temperature", 0.7))
    max_tokens = int(config.get("max_tokens", 8192))
    max_retries = int(config.get("max_retries", 3))
    enable_thinking = config.get("enable_thinking")

    subset = data if max_samples is None else data[:max_samples]
    results: List[Dict[str, Any]] = []
    for item in tqdm(subset, desc=f"predict[{model_name}]", ncols=80):
        messages = build_messages(few_shots=few_shots, current_item=item, criteria=criteria)
        raw = predict_single(
            client=client,
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            enable_thinking=enable_thinking,
            base_url=base_url or "",
        )
        new_item = deepcopy(item)
        new_item["rhetoricItems_pred"] = raw
        results.append(new_item)

    logging.info("predict_batch 完成: %d 条 (model=%s)", len(results), model_name)
    return results


__all__ = [
    "RHETORIC_LABELS",
    "FORM_LABELS",
    "CONTENT_LABELS",
    "COMPONENT_LABELS",
    "detect_provider",
    "_is_proxy_url",
    "_extract_error_detail",
    "_uses_max_completion_tokens",
    "_build_thinking_kwargs",
    "build_init_messages",
    "generate_criteria",
    "build_messages",
    "predict_single",
    "predict_batch",
]
