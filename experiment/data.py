"""Read existing JSON/JSONL manifests or inline samples; no dataset migration."""
from dataclasses import dataclass
from pathlib import Path
import json
import re

from .io import read_json
from .answer_forms import FORMATS


@dataclass
class Sample:
    task_id: str
    private_text: str       # Experiment setup/oracle only, never passed to other objectives.
    public_text: str
    question: str
    answer: str            # Diagnostic evaluator only.
    aliases: list
    held_out_question: str | None
    held_out_answer: str | None
    held_out_aliases: list
    output_instruction: str
    manifest: str
    answer_format: str = "free_text"


def catalog(config):
    result, seen = [], set()
    for value in config["datasets"]:
        path = Path(value)
        if path.is_dir():
            path /= "tasks.jsonl"
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        else:
            rows = read_json(path)
            if isinstance(rows, dict):
                rows = rows.get("tasks", rows.get("samples"))
        if not isinstance(rows, list):
            raise ValueError(f"{path}: 需要样本列表")
        for row in rows:
            name = row.get("task_id", row.get("id"))
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in (".", ".."):
                raise ValueError(f"无效样本 ID: {name!r}")
            if name in seen:
                raise ValueError(f"重复样本: {name}")
            seen.add(name)
            result.append({**row, "task_id": name, "_manifest": str(path.resolve())})
    unknown = (set(config["include_tasks"]) | set(config["exclude_tasks"])) - seen
    if unknown:
        raise ValueError(f"配置中的样本不存在: {sorted(unknown)}")
    result = [r for r in result if (not config["include_tasks"] or r["task_id"] in config["include_tasks"])
              and r["task_id"] not in config["exclude_tasks"]]
    if config["limit"]:
        result = result[:config["limit"]]
    if not result:
        raise ValueError("没有选中样本")
    return result


def materialize(row, instruction):
    base = Path(row["_manifest"]).parent

    def field(mapping, *names, required=True):
        for name in names:
            if name + "_file" in mapping:
                value = (base / mapping[name + "_file"]).read_text(encoding="utf-8-sig")
            else:
                value = mapping.get(name)
            if isinstance(value, str) and value.strip():
                return value
        if required:
            raise ValueError(f"{row['task_id']}: 缺少文本字段 {names}")
        return None

    teacher = row.get("teacher_only", row)
    receiver = row.get("receiver_inputs", row)
    evaluation = read_json(base / row["evaluation_only_file"]) if row.get("evaluation_only_file") else row.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation 必须为对象")
    answer = evaluation.get("canonical_answer", row.get("answer"))
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{row['task_id']}: 需要 canonical_answer 用于可视化和测试")
    aliases = evaluation.get("accepted_aliases", row.get("accepted_aliases", []))
    if not isinstance(aliases, list) or any(not isinstance(x, str) or not x.strip() for x in aliases):
        raise ValueError("accepted_aliases 必须为非空字符串组成的列表")
    held = field(row, "held_out_question", required=False)
    held_evaluation = row.get("held_out_evaluation", {})
    if row.get("held_out_evaluation_only_file"):
        held_evaluation = read_json(base / row["held_out_evaluation_only_file"])
    held_answer = held_evaluation.get("canonical_answer", answer) if held else None
    held_aliases = held_evaluation.get("accepted_aliases", aliases) if held else []
    sample_instruction = receiver.get("output_instruction", row.get("output_instruction", instruction))
    if not isinstance(sample_instruction, str) or not sample_instruction.strip():
        raise ValueError("output_instruction 不能为空")
    answer_format = row.get("answer_format", "free_text")
    if answer_format not in FORMATS:
        raise ValueError(f"未知答案格式: {answer_format}")
    return Sample(row["task_id"], field(teacher, "prefix_a", "private_prefix"),
                  field(receiver, "public_chunk", "public_text"), field(receiver, "question"),
                  answer, aliases, held, held_answer, held_aliases, sample_instruction, row["_manifest"], answer_format)


def question_text(question, instruction, answer_boundary="Answer: "):
    # A fixed prefill belongs to the input, including its trailing whitespace.
    text = re.sub(r"(?:^|\n)\s*(?:Answer|答案)\s*[:：]\s*$", "", question, flags=re.I).strip()
    return f"{instruction.strip()}\n\n{text}\n{answer_boundary}"
