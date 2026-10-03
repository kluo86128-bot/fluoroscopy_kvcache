from copy import deepcopy
import math
from pathlib import Path

from .io import read_json
from .question_semantics import SEMANTIC_GROUPS, SEMANTIC_WEIGHTS

METHODS = ("baseline", "question_weighted_kv", "question_attention_reconstruction",
           "oracle_private_prefix_distillation")
RETIRED_METHODS = ("question_output_consistency",)
DIRECT_OUTPUT = "Output only the requested answer in English. No labels, numbering, explanation, or extra text."
DEFAULTS = {
    "experiment_version": 2,
    "model_path": "/home/kevin/models/Qwen3-4B", "device": "cuda:0", "cuda_devices": None,
    "dtype": "bfloat16", "attention_backend": "sdpa", "local_files_only": True,
    "datasets": ["../test_samples/tasks.jsonl"], "include_tasks": [], "exclude_tasks": [], "limit": None,
    "methods": ["baseline"], "method_options": {}, "output_dir": "../results",
    "background": True, "continue_on_error": True, "auto_test": True,
    "output_instruction": DIRECT_OUTPUT, "question_format": "raw", "enable_thinking": False,
    "answer_boundary": "Answer: ", "stop_strings": ["\n"],
    "seed": 42, "rounds": 20, "steps_per_round": 200, "base_loss": "token",
    "lr": 0.01, "init_std": 0.02, "grad_clip": 1.0,
    "log_every": 25, "eval_every": 25, "plot_every": 200, "checkpoint_every": 1,
    "resume_every": 25, "save_topk": 20, "checkpoint_metric": "base",
    "reference_refresh_steps": 200, "lambda_weighted": 1.0, "weight_floor": 0.1,
    "weighted_kv_version": 2, "semantic_query_mode": "structured",
    "semantic_query_weights": SEMANTIC_WEIGHTS.copy(),
    "lambda_output": 1.0, "lambda_lse": 1.0, "lambda_attention": 1.0,
    "attention_loss_version": 2, "lambda_kl": 1.0,
    "temperature": 1.0, "rollout_tokens": 8, "max_new_tokens": 16,
    "score_window_steps": 400, "score_tail_fraction": 0.25, "ema_alpha": 0.5,
    "ema_span_steps": 400, "score_probability": "accepted_forms", "answer_leading_spaces": [0, 1],
}
METHOD_KEYS = {"lambda_weighted", "weight_floor", "lambda_output", "lambda_lse", "lambda_attention", "lambda_kl",
               "temperature", "rollout_tokens", "reference_refresh_steps", "semantic_query_mode", "semantic_query_weights"}


def validate(config, *, allow_retired=False):
    unknown = set(config) - set(DEFAULTS) - {"config_path"}
    if unknown:
        raise ValueError(f"未知配置项: {sorted(unknown)}")
    if type(config["experiment_version"]) is not int or config["experiment_version"] not in (1, 2):
        raise ValueError("experiment_version 必须为 1 或 2")
    if type(config["attention_loss_version"]) is not int or config["attention_loss_version"] not in (1, 2):
        raise ValueError("attention_loss_version 必须为 1 或 2")
    if type(config["weighted_kv_version"]) is not int or config["weighted_kv_version"] not in (1, 2):
        raise ValueError("weighted_kv_version 必须为 1 或 2")
    shares = config["semantic_query_weights"]
    if not isinstance(shares, dict) or set(shares) != set(SEMANTIC_GROUPS):
        raise ValueError("semantic_query_weights 必须包含 object_state/target_field/required_value/question 四组")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in shares.values()) or not math.isclose(sum(shares.values()), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("semantic_query_weights 必须为有限非负数，且总和为 1")
    spaces = config["answer_leading_spaces"]
    if not isinstance(spaces, list) or not spaces or any(type(n) is not int or n not in (0, 1) for n in spaces) or len(set(spaces)) != len(spaces):
        raise ValueError("answer_leading_spaces 必须为 0/1 的非空、不重复列表")
    for key in ("rounds", "steps_per_round", "log_every", "eval_every", "plot_every", "checkpoint_every",
                "resume_every", "save_topk", "reference_refresh_steps", "rollout_tokens", "max_new_tokens",
                "score_window_steps", "ema_span_steps"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} 必须为正整数")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed 必须为非负整数，不支持多个种子")
    for key in ("lr", "init_std", "grad_clip", "temperature"):
        if isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} 必须为有限正数")
    for key in ("lambda_weighted", "lambda_output", "lambda_lse", "lambda_attention", "lambda_kl"):
        if isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} 必须为有限非负数")
    for key in ("weight_floor", "score_tail_fraction", "ema_alpha"):
        if not isinstance(config[key], (int, float)) or isinstance(config[key], bool) or not math.isfinite(config[key]) or not 0 < config[key] <= 1:
            raise ValueError(f"{key} 必须在 (0,1] 内")
    for key, allowed in {"base_loss": ("mean", "token", "none"), "checkpoint_metric": ("base", "total"),
                         "dtype": ("float32", "float16", "bfloat16"), "attention_backend": ("eager", "sdpa"),
                         "semantic_query_mode": ("structured", "uniform"),
                         "question_format": ("raw", "chat"), "score_probability": ("sequence", "accepted_forms")}.items():
        if config[key] not in allowed:
            raise ValueError(f"{key} 只能选择 {allowed}")
    methods = config["methods"]
    allowed_methods = METHODS + RETIRED_METHODS if allow_retired else METHODS
    if isinstance(methods, list) and not allow_retired and any(m in RETIRED_METHODS for m in methods):
        raise ValueError("question_output_consistency 已退出实验组；仅允许历史结果测试与诊断，不允许新训练或续训")
    if not isinstance(methods, list) or not methods or any(m not in allowed_methods for m in methods) or len(set(methods)) != len(methods):
        raise ValueError(f"methods 必须为非空、不重复的方法列表: {allowed_methods}")
    if config["base_loss"] == "none":
        auxiliary_only = {"question_weighted_kv", "question_attention_reconstruction", "oracle_private_prefix_distillation"}
        if any(method not in auxiliary_only for method in methods):
            raise ValueError("base_loss=none 仅允许 question_weighted_kv、question_attention_reconstruction 和 oracle_private_prefix_distillation；Baseline 必须使用 mean 或 token 基础损失")
        if config["checkpoint_metric"] != "total":
            raise ValueError("base_loss=none 时必须按 total 保存 TopK 前缀")
    for key in ("datasets", "include_tasks", "exclude_tasks"):
        if not isinstance(config[key], list) or any(not isinstance(x, str) for x in config[key]):
            raise ValueError(f"{key} 必须是字符串列表")
    if not config["datasets"]:
        raise ValueError("datasets 不能为空")
    if config["limit"] is not None and (isinstance(config["limit"], bool) or not isinstance(config["limit"], int) or config["limit"] < 1):
        raise ValueError("limit 必须为正整数或 null")
    for key in ("background", "continue_on_error", "auto_test", "local_files_only", "enable_thinking"):
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} 必须为布尔值")
    for key in ("model_path", "device", "output_dir", "output_instruction"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} 必须为非空字符串")
    if config["cuda_devices"] is not None and not isinstance(config["cuda_devices"], str):
        raise ValueError("cuda_devices 必须为字符串，例如 '2'；暴露一张卡时 device 使用 cuda:0")
    boundary = config["answer_boundary"]
    if not isinstance(boundary, str) or not boundary.strip() or not boundary.endswith((" ", "\n", "\t")):
        raise ValueError("answer_boundary 必须包含标记并以空白结束，例如 'Answer: '")
    if not isinstance(config["stop_strings"], list) or any(not isinstance(s, str) or not s for s in config["stop_strings"]):
        raise ValueError("stop_strings 必须为非空字符串组成的列表，可用 [] 禁用")
    options = config["method_options"]
    if not isinstance(options, dict) or any(m not in allowed_methods for m in options):
        raise ValueError("method_options 的键必须为方法名")
    for method, override in options.items():
        if not isinstance(override, dict) or set(override) - METHOD_KEYS:
            raise ValueError(f"{method}: method_options 只允许辅助损失和参考参数")
        trial = {**config, **override, "method_options": {}}
        validate(trial, allow_retired=allow_retired)
    return config


def load_config(path, overrides=None):
    path = Path(path).resolve()
    config = deepcopy(DEFAULTS)
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("配置文件必须为 JSON 对象")
    config.update(raw)
    config.update(overrides or {})
    config["config_path"] = str(path)
    for key in ("output_dir",):
        value = Path(config[key]).expanduser()
        config[key] = str((value if value.is_absolute() else path.parent / value).resolve())
    config["datasets"] = [str((Path(p).expanduser() if Path(p).expanduser().is_absolute() else path.parent / p).resolve()) for p in config["datasets"]]
    return validate(config)


def method_config(config, method):
    return {**config, **config["method_options"].get(method, {})}


def saved_config(raw):
    """Missing loss version marks historical O/LSE-only runs, never version 2."""
    return {**deepcopy(DEFAULTS), **raw,
            "experiment_version": raw.get("experiment_version", 1),
            "attention_loss_version": raw.get("attention_loss_version", 1),
            "lambda_attention": raw.get("lambda_attention", 0.0),
            "weighted_kv_version": raw.get("weighted_kv_version", 1),
            "semantic_query_mode": raw.get("semantic_query_mode", "uniform")}


def require_current_attention_loss(config):
    if "question_attention_reconstruction" in config["methods"] and config.get("attention_loss_version", 1) != DEFAULTS["attention_loss_version"]:
        raise ValueError("attention_restruct 损失已升级为逐位置注意力 KL + output + logsumexp；旧 attention 运行不能续训，请新建实验（历史 test/diagnose 仍可用）")


def require_current_weighted_kv(config):
    if "question_weighted_kv" in config["methods"] and config.get("weighted_kv_version", 1) != DEFAULTS["weighted_kv_version"]:
        raise ValueError("weight_kv 已升级为四组语义查询加权；旧 weight_kv 运行不能续训，请新建实验（历史 test/diagnose 仍可用）")
