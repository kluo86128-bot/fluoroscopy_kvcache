"""Reevaluate a completed run's final prefix pools with current question_test files.

python -u reevaluate_saved_run.py --run-dir /path/to/timestamp_run --cuda-devices 3
Add --inspect for a model-free check. Source runs are never rewritten.
"""
import argparse
import contextlib
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from experiment.cli import Tee
from experiment.config import saved_config
from experiment.data import catalog, materialize, question_text
from experiment.io import read_json, write_json, write_csv, write_error_status, is_storage_error
from experiment.test_tables import build_tables, render_tables
from test_saved_prefixes import load_prefix

ROOT = Path(__file__).resolve().parent
QUESTION_KIND = "background_single_field_retest"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_plan(source, dataset, *, methods=None, tasks=None, overrides=None):
    source = Path(source).expanduser().resolve()
    dataset = Path(dataset).expanduser().resolve()
    config = saved_config(read_json(source / "resolved_config.json"))
    final = read_json(source / "result.json")
    if final.get("status") not in ("completed", "completed_with_errors", "partial"):
        raise ValueError("来源训练尚未完成，不能重测最终保存池")
    status_path = source / "status.json"
    if status_path.is_file() and read_json(status_path).get("status") in ("starting", "running"):
        raise ValueError("来源运行仍在执行，请等待其完成")
    config.update({key: value for key, value in (overrides or {}).items() if value is not None})
    selected = read_json(source / "selected_tasks.json")
    if not isinstance(selected, list) or len(set(selected)) != len(selected) or not selected:
        raise ValueError("原运行的 selected_tasks.json 无效")
    if tasks and set(tasks) - set(selected):
        raise ValueError("指定样本不属于原运行")
    task_ids = [task for task in selected if not tasks or task in tasks]
    if methods and set(methods) - set(config["methods"]):
        raise ValueError("指定方法不属于原运行")
    chosen_methods = [method for method in config["methods"] if not methods or method in methods]
    if not chosen_methods:
        raise ValueError("没有选中的方法")
    rows = catalog({**config, "datasets": [str(dataset)], "include_tasks": task_ids,
                    "exclude_tasks": [], "limit": None})
    by_id = {row["task_id"]: row for row in rows}
    problems, samples = [], []
    for task_id in task_ids:
        sample = materialize(by_id[task_id], config["output_instruction"])
        if not sample.held_out_question:
            raise ValueError(f"{task_id}: 缺少question_test问题")
        directory = source / "samples" / task_id
        info_path = directory / "sample_info.json"
        if not info_path.is_file():
            problems.append({"task_id": task_id, "error": "缺少sample_info.json，无法核验来源上下文"})
            continue
        info = read_json(info_path)
        required = {"question", "output_instruction", "context_signature"}
        if required - info.keys():
            problems.append({"task_id": task_id, "error": "sample_info缺少来源上下文校验字段"})
            continue
        context = sample.private_text + "\0" + sample.public_text + "\0" + info["question"] + "\0" + info["output_instruction"]
        if digest(context) != info["context_signature"] or info.get("task_id", task_id) != task_id:
            problems.append({"task_id": task_id, "error": "当前私有/公共上下文与原运行不同"})
            continue
        # Original output instruction is retained even if the live manifest changed.
        sample.output_instruction = info["output_instruction"]
        jobs = []
        for method in chosen_methods:
            if Path(method).name != method or method in (".", ".."):
                raise ValueError("来源方法名不能包含目录路径")
            method_result = directory / method / "result.json"
            if not method_result.is_file() or read_json(method_result).get("status") != "completed":
                problems.append({"task_id": task_id, "method": method, "error": "该方法未完成训练"})
                continue
            manifest_path = directory / method / "prefixes/manifest.json"
            if not manifest_path.is_file():
                problems.append({"task_id": task_id, "method": method, "error": "缺少最终前缀manifest"})
                continue
            items = read_json(manifest_path)["prefixes"]
            if not items:
                problems.append({"task_id": task_id, "method": method, "error": "最终前缀池为空"})
                continue
            paths, prepared = set(), []
            for item in sorted(items, key=lambda item: item["rank"]):
                filename = item["path"]
                if Path(filename).name != filename or filename in (".", "..") or filename in paths:
                    raise ValueError(f"{manifest_path}: 无效或重复的前缀文件名")
                paths.add(filename)
                path = manifest_path.parent / filename
                prepared.append({**item, "file": path})
                if not path.is_file():
                    problems.append({"task_id": task_id, "method": method, "prefix_path": str(path), "error": "保存前缀文件缺失"})
            jobs.append({"method": method, "items": prepared})
        samples.append({"sample": sample, "methods": jobs,
                        "source_training_question_sha256": digest(info["question"]),
                        "test_question_version": by_id[task_id].get("test_question_version")})
    return {"source": source, "dataset": dataset, "config": config, "methods": chosen_methods,
            "task_ids": task_ids, "samples": samples, "problems": problems}


def plan_summary(plan):
    return {"source_run": str(plan["source"]), "evaluation_scope": QUESTION_KIND,
            "planned_samples": len(plan["task_ids"]), "methods": plan["methods"],
            "manifest_prefixes": sum(len(job["items"]) for sample in plan["samples"] for job in sample["methods"]),
            "problem_count": len(plan["problems"]), "problems": plan["problems"],
            "device": plan["config"]["device"], "cuda_devices": plan["config"]["cuda_devices"]}


def save_report(root, plan, rows, failures, status, *, announce=False):
    valid = [row for row in rows if row["status"] == "tested"]
    method_rows, sample_rows = build_tables(valid, plan["methods"], plan["task_ids"],
                                           [QUESTION_KIND], len(plan["task_ids"]))
    write_csv(root / "test_method_success.csv", method_rows)
    write_csv(root / "test_sample_prefix_rates.csv", sample_rows)
    write_csv(root / "test_results.csv", [{key: value for key, value in row.items() if key != "generated_token_ids"} for row in rows])
    write_json(root / "test_results.json", rows)
    report = {**plan_summary(plan), "status": status, "tested_prefixes": len(valid),
              "invalid_prefixes": sum(row["status"] == "invalid_prefix" for row in rows),
              "unprocessed_prefixes": plan_summary(plan)["manifest_prefixes"] - len(rows),
              "methods_summary": method_rows, "failures": failures}
    write_json(root / "summary.json", report)
    write_json(root / "status.json", {"status": status, "tested_prefixes": len(valid), "failure_count": len(failures)})
    if announce:
        print(render_tables(method_rows, sample_rows, plan["methods"], plan["task_ids"], [QUESTION_KIND]), flush=True)
        print(f"重评估结果目录: {root}", flush=True)
    return report


def run_reevaluation(plan, output_dir, *, backend=None):
    root = Path(output_dir).expanduser().resolve()
    if root.is_relative_to(plan["source"]):
        raise ValueError("重评估目录必须位于原运行目录之外")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "reevaluation_config.json", {"plan": plan_summary(plan), "inference_config": plan["config"],
                                                  "scope": "current_test_question; final_manifest_only; no_optimization"})
    questions = [{"task_id": item["sample"].task_id, "question": item["sample"].held_out_question,
                  "question_sha256": digest(item["sample"].held_out_question),
                  "source_training_question_sha256": item["source_training_question_sha256"],
                  "test_question_version": item["test_question_version"],
                  "output_instruction": item["sample"].output_instruction}
                 for item in plan["samples"]]
    write_json(root / "questions_used.json", questions)
    rows, failures = [], list(plan["problems"])
    try:
        with (root / "reevaluation.log").open("w", encoding="utf-8") as log, \
                contextlib.redirect_stdout(Tee(sys.stdout, log)):
            if any(item["methods"] for item in plan["samples"]):
                import torch
                from experiment.backend import load_backend
                from experiment.metrics import test_prefix
                backend = backend or load_backend(plan["config"])
                config = plan["config"]
                with (root / "test_results.jsonl").open("w", encoding="utf-8") as stream, torch.no_grad():
                    for item in plan["samples"]:
                        if not item["methods"]:
                            continue
                        sample = item["sample"]
                        public_ids = backend.encode(sample.public_text)
                        observed, length, _ = backend.setup_teacher(backend.encode(sample.private_text), public_ids, include_private=False)
                        ids = backend.encode_question(question_text(sample.held_out_question, sample.output_instruction,
                                                                     config["answer_boundary"]), config)
                        shape = (1, length, backend.embedding.weight.shape[1])
                        for job in item["methods"]:
                            method_rows = []
                            for prefix_item in job["items"]:
                                path = prefix_item["file"]
                                row = {"source_run": str(plan["source"]), "task_id": sample.task_id,
                                       "method": job["method"], "question_kind": QUESTION_KIND,
                                       "rank": prefix_item["rank"], "step": prefix_item["step"],
                                       "checkpoint_loss": prefix_item["loss"], "prefix_path": str(path),
                                       "correct_answer": sample.held_out_answer}
                                try:
                                    prefix, _ = load_prefix(path, shape)
                                except Exception as error:
                                    if is_storage_error(error):
                                        raise
                                    row.update(status="invalid_prefix", error=f"{type(error).__name__}: {error}")
                                    if not any(failure.get("prefix_path") == str(path) for failure in failures):
                                        failures.append({"task_id": sample.task_id, "method": job["method"],
                                                         "prefix_path": str(path), "error": row["error"]})
                                else:
                                    try:
                                        row.update(test_prefix(backend, observed, public_ids, ids, prefix,
                                                               sample.held_out_answer, sample.held_out_aliases,
                                                               config["max_new_tokens"], config["stop_strings"], sample.answer_format))
                                        row["status"] = "tested"
                                    except Exception as error:
                                        failures.append({"task_id": sample.task_id, "method": job["method"],
                                                         "prefix_path": str(path), "error": f"{type(error).__name__}: {error}"})
                                        raise
                                rows.append(row)
                                method_rows.append(row)
                                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                                stream.flush()
                                detail = f"answer={row['answer_match']} output={row['output']!r}" if row["status"] == "tested" else row["error"]
                                print(f"[重评估 {sample.task_id}/{job['method']} rank={row['rank']}/{len(job['items'])}] {detail}", flush=True)
                            destination = root / "samples" / sample.task_id / job["method"]
                            write_json(destination / "test_results.json", method_rows)
                            write_csv(destination / "test_results.csv", [{key: value for key, value in row.items()
                                                                          if key != "generated_token_ids"} for row in method_rows])
                            save_report(root, plan, rows, failures, "running")
            save_report(root, plan, rows, failures, "completed_with_errors" if failures else "completed", announce=True)
        return 1 if failures else 0
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        try:
            save_report(root, plan, rows, failures, status)
        except Exception:
            pass
        write_error_status(root / "status.json", {"status": status, "error": f"{type(error).__name__}: {error}"})
        raise


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="用当前question_test批量重评估已完成运行的最终保存池，不训练")
    parser.add_argument("--run-dir", type=Path, required=True, help="具体时间戳运行目录，含resolved_config.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "test_samples/tasks.jsonl")
    parser.add_argument("--cuda-devices", help="物理GPU，例如3")
    parser.add_argument("--device", choices=("cpu", "cuda:0"))
    parser.add_argument("--model-path")
    parser.add_argument("--methods", nargs="+", help="默认原运行的全部方法")
    parser.add_argument("--include-tasks", nargs="+", help="默认原运行的全部样本")
    parser.add_argument("--output-dir", type=Path, help="尚不存在，且位于原运行之外")
    parser.add_argument("--inspect", action="store_true", help="仅检查来源、问题与manifest，不加载模型")
    args = parser.parse_args(argv)
    plan = build_plan(args.run_dir, args.dataset, methods=args.methods, tasks=args.include_tasks,
                      overrides={key: getattr(args, key) for key in ("cuda_devices", "device", "model_path")})
    print(json.dumps(plan_summary(plan), ensure_ascii=False, indent=2), flush=True)
    output = (args.output_dir or ROOT / "outputs/run_reevaluation" /
              (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])).expanduser().resolve()
    if output.exists() or output.is_relative_to(plan["source"]):
        raise ValueError("输出目录必须尚不存在且位于原运行之外")
    if args.inspect:
        return 1 if plan["problems"] else 0
    if plan["config"]["cuda_devices"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = plan["config"]["cuda_devices"]
    return run_reevaluation(plan, output)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"重评估失败: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
