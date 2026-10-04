"""Paired question-only tests of existing prefixes; never train or edit source runs.

Examples (run from the project directory):
  python retest_questions.py --inspect
  python retest_questions.py --run-dirs results/three_loss_token --cuda-devices 1
  python retest_questions.py --cuda-devices 0

By default discover the latest three_loss mean/token/none runs and test all their
enabled methods. A group directory or an exact timestamp directory is accepted.
Results go to output/question_retest/<timestamp>. Only saved TopK snapshots are
observed; this script cannot reconstruct unsaved training steps.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from experiment.config import METHODS, saved_config
from experiment.data import catalog, materialize, question_text
from experiment.io import read_json, write_csv, write_json, write_error_status

ROOT = Path(__file__).resolve().parent
PREVIOUS = "unique_field_english_v2"
CURRENT = "structured_field_semantic_v3"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def context_signature(sample, question, instruction):
    return digest(sample.private_text + "\0" + sample.public_text + "\0" + question + "\0" + instruction)


def read_source_config(source):
    path = source / "resolved_config.json"
    if path.is_file():
        raw = read_json(path)
    else:
        # Exported log/image collections can be inspected, but cannot be tested.
        text = (source / "train.log").read_text(encoding="utf-8-sig")
        marker = "完整配置:"
        if marker not in text:
            raise ValueError(f"缺少 resolved_config.json 或日志完整配置: {source}")
        raw = json.JSONDecoder().raw_decode(text.split(marker, 1)[1].lstrip())[0]
    config = saved_config(raw)
    if config["experiment_version"] != 2:
        raise ValueError("此脚本只比较 experiment_version=2，不能混用旧概率定义")
    if config["score_probability"] != "accepted_forms" or config["answer_leading_spaces"] != [0, 1]:
        raise ValueError("来源须使用 accepted_forms 与零/一个额外前导空格")
    if config["answer_boundary"] != "Answer: ":
        raise ValueError("来源回答边界须为带尾部空格的 'Answer: '")
    return config


def resolve_source(path):
    path = Path(path).expanduser().resolve()
    if (path / "resolved_config.json").is_file() or (path / "train.log").is_file():
        return path
    candidates = {p.parent for name in ("resolved_config.json", "train.log") for p in path.rglob(name)}
    if not candidates:
        raise FileNotFoundError(f"没有运行目录: {path}")
    return max(candidates, key=lambda p: (p.name, str(p)))


def discover_sources():
    candidates = {}
    for log in (ROOT / "results").rglob("train.log"):
        if any(part.startswith("three_loss_") for part in log.relative_to(ROOT / "results").parts):
            config = read_source_config(log.parent)
            mode = config["base_loss"]
            previous = candidates.get(mode)
            if previous is None or log.parent.name > previous.name:
                candidates[mode] = log.parent
    if not candidates:
        raise FileNotFoundError("未发现 three_loss 运行，请用 --run-dirs 指定完整结果目录")
    return [candidates[mode] for mode in ("mean", "token", "none") if mode in candidates]


def load_questions(bundle, task, current_questions, metadata):
    entry = metadata["samples"].get(task)
    if entry is None:
        raise ValueError(f"隔离问题集中没有样本 {task}")
    result = {}
    for kind, current in current_questions.items():
        item = entry[kind]
        path = (bundle / item["file"]).resolve()
        if bundle not in path.parents:
            raise ValueError("隔离问题文件路径越界")
        previous = path.read_text(encoding="utf-8-sig")
        if digest(previous) != item["sha256"]:
            raise ValueError(f"上一版问题与 Git 快照不同: {path}")
        if digest(current) != item["expected_current_sha256"]:
            raise ValueError(f"{task}/{kind}: 当前问题已变化，无法与本次结构化版本对照")
        result[kind] = {PREVIOUS: previous, CURRENT: current}
    return result


def build_plan(sources, dataset, bundle, methods=None, tasks=None, max_prefixes=None):
    bundle = Path(bundle).resolve()
    metadata = read_json(bundle / "manifest.json")
    if metadata["question_version"] != PREVIOUS or metadata["current_question_version"] != CURRENT:
        raise ValueError("隔离 question 版本不匹配")
    plans, problems = [], []
    for source in sources:
        source = Path(source).resolve()
        config = read_source_config(source)
        config["datasets"] = [str(Path(dataset).resolve())]
        selected_methods = [m for m in config["methods"] if m in METHODS and (not methods or m in methods)]
        if not selected_methods:
            raise ValueError(f"{source}: 没有选中的现行方法")
        samples = [materialize(row, config["output_instruction"]) for row in catalog(config)]
        if tasks:
            unknown = set(tasks) - {s.task_id for s in samples}
            if unknown:
                raise ValueError(f"{source}: 没有选中的样本 {sorted(unknown)}")
            samples = [s for s in samples if s.task_id in tasks]
        sample_jobs = []
        for sample in samples:
            current = {"training_question": sample.question}
            if sample.held_out_question:
                current["held_out_question"] = sample.held_out_question
            questions = load_questions(bundle, sample.task_id, current, metadata)
            directory = source / "samples" / sample.task_id
            info_path = directory / "sample_info.json"
            if not info_path.is_file():
                problems.append(f"缺少上下文校验文件: {info_path}")
            else:
                info = read_json(info_path)
                if info["question"] != sample.question or info["output_instruction"] != sample.output_instruction:
                    raise ValueError(f"{sample.task_id}: 来源问题/输出约束与结构化快照不同")
                if info["context_signature"] != context_signature(sample, sample.question, sample.output_instruction):
                    raise ValueError(f"{sample.task_id}: 私有/公共上下文与原运行不同")
            method_jobs = []
            for method in selected_methods:
                manifest_path = directory / method / "prefixes" / "manifest.json"
                if not manifest_path.is_file():
                    problems.append(f"缺少保存前缀清单: {manifest_path}")
                    continue
                items = sorted(read_json(manifest_path)["prefixes"], key=lambda item: item["rank"])
                if not items:
                    problems.append(f"保存前缀清单为空: {manifest_path}")
                    continue
                if max_prefixes:
                    items = items[:max_prefixes]
                for item in items:
                    if Path(item["path"]).name != item["path"] or item["path"] in (".", ".."):
                        raise ValueError(f"无效前缀文件名: {manifest_path}")
                    item["file"] = manifest_path.parent / item["path"]
                    if not item["file"].is_file():
                        problems.append(f"缺少前缀张量: {item['file']}")
                method_jobs.append({"method": method, "items": items})
            sample_jobs.append({"sample": sample, "questions": questions, "methods": method_jobs})
        plans.append({"source": source, "config": config, "selected_methods": selected_methods, "samples": sample_jobs})
    return plans, metadata, problems


def plan_summary(plans, problems):
    return {"scope": "saved_snapshots_only", "question_versions": [PREVIOUS, CURRENT],
            "ready": not problems, "problem_count": len(problems), "problems": problems[:12],
            "runs": [{"source": str(p["source"]), "base_loss": p["config"]["base_loss"],
                      "methods": p["selected_methods"],
                      "sample_count": len(p["samples"]),
                      "prefix_count": sum(len(m["items"]) for s in p["samples"] for m in s["methods"]),
                      "planned_outputs": sum(len(m["items"]) * len(s["questions"]) * 2
                                             for s in p["samples"] for m in s["methods"])} for p in plans]}


def evaluate_question(backend, private, observed, ids, answer, aliases, answer_format, config, monitor):
    from experiment.backend import join_cache
    from experiment.metrics import answer_match
    probability = monitor(private)
    generated = backend.rollout(join_cache(private, observed), ids, config["max_new_tokens"],
                                stop_strings=config["stop_strings"])
    values = generated[0].tolist()
    text = backend.tokenizer.decode(values, skip_special_tokens=True)
    stops = [text.find(stop) for stop in config["stop_strings"] if stop in text]
    if stops:
        text = text[:min(stops)]
    answer_ids = backend.encode(answer)[0].tolist()
    return {**probability, "output": text, "generated_token_ids": values,
            "first_token_match": bool(values and values[0] == answer_ids[0]),
            **answer_match(text, answer, aliases, answer_format)}


def paired_row(previous, current):
    keys = ("source_run", "base_loss", "task_id", "method", "question_kind", "rank", "step", "checkpoint_loss", "prefix_sha256")
    result = {key: previous[key] for key in keys}
    for label, row in (("previous", previous), ("current", current)):
        for key in ("output", "answer_match", "format_compliant", "target_mentioned", "answer_probability", "canonical_answer_probability"):
            result[label + "_" + key] = row[key]
    result["answer_probability_delta_previous_minus_current"] = previous["answer_probability"] - current["answer_probability"]
    result["automatic_match_transition"] = f"{int(current['answer_match'])}->{int(previous['answer_match'])}"
    return result


def summarize_pairs(pairs):
    groups = defaultdict(list)
    for row in pairs:
        groups[(row["source_run"], row["base_loss"], row["method"], row["question_kind"])].append(row)
    rows = []
    for key, items in sorted(groups.items()):
        old = sum(r["previous_answer_match"] for r in items)
        new = sum(r["current_answer_match"] for r in items)
        rows.append(dict(zip(("source_run", "base_loss", "method", "question_kind"), key),
                         samples=len({r["task_id"] for r in items}), paired_prefixes=len(items),
                         previous_automatic_match_count=old, current_automatic_match_count=new,
                         previous_automatic_match_rate=old / len(items), current_automatic_match_rate=new / len(items),
                         gained_automatic_matches=sum(r["automatic_match_transition"] == "0->1" for r in items),
                         lost_automatic_matches=sum(r["automatic_match_transition"] == "1->0" for r in items),
                         mean_probability_delta_previous_minus_current=sum(r["answer_probability_delta_previous_minus_current"] for r in items) / len(items),
                         metric_scope="accepted_forms_only; format_separate; semantic_review_required"))
    return rows


def run_retest(plans, root, metadata, *, backend=None):
    import torch
    from experiment.backend import load_backend
    from experiment.metrics import AnswerMonitor
    root = Path(root).resolve()
    if any(root == p["source"] or p["source"] in root.parents for p in plans):
        raise ValueError("输出目录必须与原运行目录隔离")
    bundle = (ROOT / "question_sets").resolve()
    if root == bundle or bundle in root.parents:
        raise ValueError("输出不能写入隔离问题集")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "status.json", {"status": "running", "scope": "saved_snapshots_only"})
    write_json(root / "retest_config.json", {"question_metadata": metadata, "plan": plan_summary(plans, []),
               "sources": [{"run": str(p["source"]), "evaluation_config": p["config"]} for p in plans],
               "scope": "same_prefix_paired_questions; no_optimization; saved_snapshots_only"})
    pairs, questions_used, completed = [], [], 0
    try:
        backend = backend or load_backend(plans[0]["config"])
        with (root / "test_results.jsonl").open("w", encoding="utf-8") as raw, \
             (root / "test_results.csv").open("w", newline="", encoding="utf-8-sig") as csvfile:
            import csv
            writer = None
            for plan in plans:
                config = plan["config"]
                for job in plan["samples"]:
                    sample = job["sample"]
                    # True private text is used only to reconstruct the original observation.
                    observed, length, _ = backend.setup_teacher(backend.encode(sample.private_text), backend.encode(sample.public_text), include_private=False)
                    public_ids = backend.encode(sample.public_text)
                    encoded = {}
                    for kind, versions in job["questions"].items():
                        answer = sample.answer if kind == "training_question" else sample.held_out_answer
                        aliases = sample.aliases if kind == "training_question" else sample.held_out_aliases
                        for version, question in versions.items():
                            prompt = question_text(question, sample.output_instruction, config["answer_boundary"])
                            ids = backend.encode_question(prompt, config)
                            monitor = AnswerMonitor(backend, observed, ids, answer, aliases,
                                                    probability_mode=config["score_probability"], leading_spaces=config["answer_leading_spaces"])
                            encoded[(kind, version)] = (ids, answer, aliases, monitor)
                            questions_used.append({"source_run": str(plan["source"]), "task_id": sample.task_id,
                                                   "question_kind": kind, "question_version": version, "question": question,
                                                   "output_instruction": sample.output_instruction, "answer_boundary": config["answer_boundary"],
                                                   "question_tokens": ids.shape[1], "question_sha256": digest(question)})
                    for method in job["methods"]:
                        for item in method["items"]:
                            prefix = torch.load(item["file"], map_location="cpu", weights_only=True)["soft_prefix"]
                            if prefix.shape != (1, length, backend.embedding.weight.shape[1]):
                                raise ValueError(f"前缀形状与模型或样本不一致: {item['file']}")
                            prefix_hash = hashlib.sha256(prefix.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                            with torch.no_grad():
                                private, _ = backend.student(prefix.to(backend.device), public_ids)
                                for kind in job["questions"]:
                                    evaluated = {}
                                    for version in (PREVIOUS, CURRENT):
                                        ids, answer, aliases, monitor = encoded[(kind, version)]
                                        row = {"source_run": str(plan["source"]), "base_loss": config["base_loss"], "task_id": sample.task_id,
                                               "method": method["method"], "question_kind": kind, "question_version": version,
                                               "rank": item["rank"], "step": item["step"], "checkpoint_loss": item["loss"],
                                               "prefix_path": str(item["file"]), "prefix_sha256": prefix_hash, "correct_answer": answer,
                                               "scope": "saved_snapshots_only",
                                               **evaluate_question(backend, private, observed, ids, answer, aliases, sample.answer_format, config, monitor)}
                                        raw.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                                        if writer is None:
                                            writer = csv.DictWriter(csvfile, fieldnames=list(row))
                                            writer.writeheader()
                                        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
                                        evaluated[version] = row
                                    pairs.append(paired_row(evaluated[PREVIOUS], evaluated[CURRENT]))
                            completed += 1
                            if completed % 25 == 0 or item == method["items"][-1]:
                                raw.flush()
                                csvfile.flush()
                                print(f"[重测 {config['base_loss']}/{sample.task_id}/{method['method']}] rank={item['rank']} step={item['step']} 累计前缀={completed}", flush=True)
                                write_json(root / "status.json", {"status": "running", "completed_prefixes": completed, "paired_observations": len(pairs)})
        write_csv(root / "paired_comparison.csv", pairs)
        write_csv(root / "paired_summary.csv", summarize_pairs(pairs))
        write_json(root / "questions_used.json", questions_used)
        write_json(root / "status.json", {"status": "completed", "completed_prefixes": completed,
                   "paired_observations": len(pairs), "outputs": len(pairs) * 2, "scope": "saved_snapshots_only"})
        print(f"重测完成：{root}", flush=True)
        return 0
    except (KeyboardInterrupt, Exception) as error:
        write_csv(root / "paired_comparison.csv", pairs)
        write_json(root / "questions_used.json", questions_used)
        write_error_status(root / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           "completed_prefixes": completed, "paired_observations": len(pairs), "error": f"{type(error).__name__}: {error}"})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="同一保存前缀：上一版短 question 与当前结构化 question 的成对重测")
    parser.add_argument("--run-dirs", nargs="+", type=Path, help="完整运行目录或组目录；默认自动发现最新 three_loss 三组")
    parser.add_argument("--dataset", type=Path, default=ROOT / "test_samples" / "tasks.jsonl")
    parser.add_argument("--question-dir", type=Path, default=ROOT / "question_sets" / PREVIOUS)
    parser.add_argument("--output-dir", type=Path, help="独立的新目录，默认 output/question_retest/时间戳")
    parser.add_argument("--methods", nargs="+", choices=METHODS, help="默认来源运行的全部现行方法")
    parser.add_argument("--include-tasks", nargs="+")
    parser.add_argument("--max-prefixes", type=int, help="每个样本/方法按原损失排名取前 N 个，仅用于快速检查")
    parser.add_argument("--cuda-devices", help="物理 GPU，例如 1；全部来源依次使用同一张卡")
    parser.add_argument("--device", choices=("cpu", "cuda:0"))
    parser.add_argument("--model-path")
    parser.add_argument("--inspect", action="store_true", help="只检查问题、配置与前缀文件，不加载模型")
    args = parser.parse_args(argv)
    if args.max_prefixes is not None and args.max_prefixes < 1:
        parser.error("--max-prefixes 必须为正整数")
    sources = [resolve_source(p) for p in args.run_dirs] if args.run_dirs else discover_sources()
    if len(set(sources)) != len(sources):
        parser.error("重复的来源运行目录")
    plans, metadata, problems = build_plan(sources, args.dataset, args.question_dir, args.methods, args.include_tasks, args.max_prefixes)
    for plan in plans:
        for name in ("model_path", "device", "cuda_devices"):
            value = getattr(args, name)
            if value is not None:
                plan["config"][name] = value
    first = plans[0]["config"]
    # One frozen model is shared across sequential evaluations. Source losses are never used.
    for plan in plans[1:]:
        for name in ("model_path", "device", "cuda_devices", "dtype", "attention_backend"):
            if plan["config"][name] != first[name] and name != "cuda_devices":
                raise ValueError(f"来源模型设置不同: {name}，请分开运行")
        plan["config"]["cuda_devices"] = first["cuda_devices"]
    print(json.dumps(plan_summary(plans, problems), ensure_ascii=False, indent=2), flush=True)
    if problems:
        print("无法执行重测：需要服务器完整运行目录中的 sample_info.json、manifest.json 和 .pt；日志/CSV/图片不能还原前缀。", file=sys.stderr)
        return 2
    if args.inspect:
        return 0
    if first["cuda_devices"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = first["cuda_devices"]
    root = args.output_dir or ROOT / "output" / "question_retest" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    return run_retest(plans, root, metadata)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"重测失败: {error}", file=sys.stderr)
        raise SystemExit(1)
