"""Final test tables use answer matches, independently of training objectives."""
from collections import defaultdict
from pathlib import Path

from .io import write_csv


def build_tables(rows, methods, task_ids, question_kinds, planned):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["question_kind"], row["method"], row["task_id"]].append(row)
    method_rows, sample_rows = [], []
    for kind in question_kinds:
        for method in methods:
            tested, successful = 0, 0
            for task_id in task_ids:
                tests = grouped[kind, method, task_id]
                correct = sum(bool(row["answer_match"]) for row in tests)
                count = len(tests)
                tested += int(count > 0)
                successful += int(correct > 0)
                sample_rows.append({"question_kind": kind, "task_id": task_id, "method": method,
                                    "correct_prefixes": correct, "tested_prefixes": count,
                                    "prefix_success_rate": correct / count if count else None,
                                    "sample_success": bool(correct) if count else None})
            method_rows.append({"question_kind": kind, "method": method,
                                "successful_samples": successful, "tested_samples": tested,
                                "planned_samples": planned,
                                "sample_success_rate": successful / tested if tested else None})
    return method_rows, sample_rows


def render_tables(method_rows, sample_rows, methods, task_ids, question_kinds):
    lines = ["统计口径：answer_match；同一样本至少一个测试前缀正确即为成功；未测试不记为0%。"]
    cells = {(row["question_kind"], row["task_id"], row["method"]): row for row in sample_rows}
    for kind in question_kinds:
        lines.extend(["", f"[统计表1/{kind}] 各方法成功样本数",
                      "| 方法 | 成功样本数 | 已测试样本数 | 计划样本数 | 样本成功率 |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for row in method_rows:
            if row["question_kind"] == kind:
                rate = "未测试" if row["sample_success_rate"] is None else f"{row['sample_success_rate']:.2%}"
                lines.append(f"| {row['method']} | {row['successful_samples']} | "
                             f"{row['tested_samples']} | {row['planned_samples']} | {rate} |")
        lines.extend(["", f"[统计表2/{kind}] 每个样本的测试前缀成功率（正确数/实际测试数）",
                      "| 样本 | " + " | ".join(methods) + " |",
                      "| --- | " + " | ".join("---:" for _ in methods) + " |"])
        for task_id in task_ids:
            values = []
            for method in methods:
                row = cells[kind, task_id, method]
                values.append("未测试" if row["prefix_success_rate"] is None else
                              f"{row['prefix_success_rate']:.2%} "
                              f"({row['correct_prefixes']}/{row['tested_prefixes']})")
            lines.append(f"| {task_id} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_test_tables(root, config, planned, rows, task_ids, *, announce):
    methods = config["methods"]
    if config.get("test_question_mode", "both") == "held_out":
        kinds = ["held_out_question"]
    else:
        kinds = [kind for kind in ("training_question", "held_out_question")
                 if any(row["question_kind"] == kind for row in rows)] or ["training_question"]
    method_rows, sample_rows = build_tables(rows, methods, task_ids, kinds, planned)
    root = Path(root)
    write_csv(root / "test_method_success.csv", method_rows)
    write_csv(root / "test_sample_prefix_rates.csv", sample_rows)
    if announce:
        print(render_tables(method_rows, sample_rows, methods, task_ids, kinds), flush=True)

