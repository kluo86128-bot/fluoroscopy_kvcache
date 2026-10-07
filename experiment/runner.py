"""Dataset setup, independent method runs, and final-test orchestration."""
import hashlib
import json
import os
from pathlib import Path
import traceback

import torch

from .backend import load_backend
from .config import WEIGHTED_METHODS, method_config, require_current_attention_loss, require_current_weighted_kv
from .data import catalog, materialize, question_text
from .io import read_json, save_torch, write_csv, write_json, is_storage_error
from .metrics import AnswerMonitor, summary, test_prefix
from .plots import render_comparison
from .trainer import train
from .question_semantics import semantic_spans
from .test_tables import write_test_tables


def prepare(backend, sample, config):
    private_ids, public_ids = backend.encode(sample.private_text), backend.encode(sample.public_text)
    question_ids = backend.encode_question(question_text(sample.question, sample.output_instruction, config["answer_boundary"]), config)
    oracle_needed = "oracle_private_prefix_distillation" in config["methods"]
    observed, length, private_cache = backend.setup_teacher(private_ids, public_ids, include_private=oracle_needed)
    signature = hashlib.sha256((sample.private_text + "\0" + sample.public_text + "\0" +
                                sample.question + "\0" + sample.output_instruction).encode()).hexdigest()
    return observed, length, private_cache, public_ids, question_ids, signature


def test_method(backend, sample, config, directory, observed, public_ids, question_ids, *, output_directory=None, prefix_items=None):
    held_only = config.get("test_question_mode", "both") == "held_out"
    if held_only and not sample.held_out_question:
        raise ValueError(f"{sample.task_id}: held_out 测试需要独立的 question_test 问题")
    directory = Path(directory)
    manifest = read_json(directory / "prefixes" / "manifest.json")
    questions = [] if held_only else [("training_question", question_ids, sample.answer, sample.aliases)]
    if sample.held_out_question:
        held_ids = backend.encode_question(question_text(sample.held_out_question, sample.output_instruction, config["answer_boundary"]), config)
        questions.append(("held_out_question", held_ids, sample.held_out_answer, sample.held_out_aliases))
    rows = []
    items = manifest["prefixes"] if prefix_items is None else prefix_items
    for item in items:
        path = directory / "prefixes" / item["path"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for kind, ids, answer, aliases in questions:
            metrics = test_prefix(backend, observed, public_ids, ids, payload["soft_prefix"],
                                  answer, aliases, config["max_new_tokens"], config["stop_strings"], sample.answer_format)
            row = {"task_id": sample.task_id, "method": directory.name, "question_kind": kind,
                   "rank": item["rank"], "step": item["step"], "checkpoint_loss": item["loss"],
                   "prefix_path": str(path), "correct_answer": answer, **metrics}
            rows.append(row)
            print(f"[测试 {sample.task_id}/{directory.name} {kind} rank={item['rank']}/{len(manifest['prefixes'])}] "
                  f"first={row['first_token_match']} full={row['full_match']} answer={row['answer_match']} "
                  f"output={row['output'][:160]!r}", flush=True)
    destination = Path(output_directory) if output_directory else directory
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "test_results.json", rows)
    write_csv(destination / "test_results.csv", [{k: v for k, v in r.items() if k != "generated_token_ids"} for r in rows])
    return rows


def summarize_root(root, config, planned, announce=True, *, task_ids=None):
    root = Path(root)
    all_rows = []
    for path in root.glob("samples/*/*/test_results.json"):
        if path.parent.name in config["methods"]:
            all_rows.extend(read_json(path))
    summaries = {}
    for method in config["methods"]:
        summaries[method] = {}
        for kind in ("training_question", "held_out_question"):
            rows = [r for r in all_rows if r["method"] == method and r["question_kind"] == kind]
            if rows:
                summaries[method][kind] = summary(rows, planned)
    write_json(root / "test_summary.json", summaries)
    write_csv(root / "test_results.csv", [{k: v for k, v in r.items() if k != "generated_token_ids"} for r in all_rows])
    if task_ids is None:
        selection = root / "selected_tasks.json"
        task_ids = read_json(selection) if selection.exists() else sorted({r["task_id"] for r in all_rows})
    for method, questions in summaries.items():
        for kind, values in questions.items():
            rates = values["answer_match"]
            if announce:
                def rate(value):
                    return "N/A" if value is None else f"{value:.2%}"
                print(f"[汇总 {method}/{kind}] 完成样本={values['completed_samples']}/{planned} 前缀测试={values['prefix_tests']} "
                      f"Top1={rate(rates['top1_rate'])} 前缀正确率={rate(rates['prefix_rate'])} "
                      f"TopK覆盖={rate(rates['topk_coverage'])}", flush=True)
    write_test_tables(root, config, planned, all_rows, task_ids, announce=announce)
    return summaries


def run(config, root, *, resume=False, test_only=False, backend=None):
    if not test_only:
        require_current_attention_loss(config)
        require_current_weighted_kv(config)
    root = Path(root)
    selected = catalog(config)
    # Validate data before loading a multi-GB model. No labels are passed to objectives.
    samples = [materialize(row, config["output_instruction"]) for row in selected]
    if config["auto_test"] and config.get("test_question_mode", "both") == "held_out":
        for sample in samples:
            if not sample.held_out_question:
                raise ValueError(f"{sample.task_id}: held_out 测试需要独立的 question_test 问题")
    for method in config["methods"]:
        if not test_only and method in WEIGHTED_METHODS and method_config(config, method)["semantic_query_mode"] == "structured":
            for sample in samples:
                semantic_spans(sample.question)
    if resume or test_only:
        original = read_json(root / "selected_tasks.json")
        if original != [s.task_id for s in samples]:
            raise ValueError("恢复/测试时选中的样本与原实验不一致")
    else:
        write_json(root / "selected_tasks.json", [s.task_id for s in samples])
    print(f"样本数={len(samples)} 方法数={len(config['methods'])} 方法={config['methods']} "
          f"seed={config['seed']} 原损失={config['base_loss']} rounds={config['rounds']} "
          f"steps_per_round={config['steps_per_round']} TopK={config['save_topk']} "
          f"保存指标={config['checkpoint_metric']} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '未限制')} 输出={root}", flush=True)
    group_role = {"token": "token 主实验", "mean": "mean 对照", "none": "辅助损失单独组"}[config["base_loss"]]
    print(f"实验组定位={group_role} loss_profile={config.get('loss_profile', 'current')} "
          f"attention_loss_version={config.get('attention_loss_version', 1)} "
          f"weighted_kv_version={config.get('weighted_kv_version', 1)} "
          f"weighted_kv_context_version={config.get('weighted_kv_context_version', 2)} "
          f"weighted_kv_normalization_version={config.get('weighted_kv_normalization_version', 2)}", flush=True)
    print("完整配置:\n" + json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    backend = backend or load_backend(config)
    failures, completed, skipped = [], [], []
    for sample_index, sample in enumerate(samples, 1):
        print(f"[样本 {sample_index}/{len(samples)} {sample.task_id}] question={sample.question!r} "
              f"正确答案(仅诊断)={sample.answer!r} 输出约束={sample.output_instruction!r}", flush=True)
        try:
            if test_only:
                ready = []
                for method in config["methods"]:
                    directory = root / "samples" / sample.task_id / method
                    result_path = directory / "result.json"
                    if result_path.exists() and read_json(result_path).get("status") == "completed" and (directory / "prefixes" / "manifest.json").exists():
                        ready.append(method)
                    else:
                        skipped.append({"task_id": sample.task_id, "method": method, "reason": "training_not_completed"})
                if not ready:
                    print(f"{sample.task_id}: 没有已完成训练，跳过测试", flush=True)
                    continue
            observed, length, oracle, public_ids, question_ids, signature = prepare(backend, sample, config)
            sample_root = root / "samples" / sample.task_id
            sample_root.mkdir(parents=True, exist_ok=True)
            info_path = sample_root / "sample_info.json"
            if (resume or test_only) and info_path.exists() and read_json(info_path)["context_signature"] != signature:
                raise ValueError("样本的私有/公共上下文、训练问题或输出约束已变化，不能使用原运行续跑或重测")
            initial_path = sample_root / "initial.pt"
            if initial_path.exists():
                initial = torch.load(initial_path, map_location="cpu", weights_only=True)["soft_prefix"]
            else:
                if test_only:
                    raise ValueError("已完成实验缺少初始状态文件")
                generator = torch.Generator(device="cpu").manual_seed(config["seed"])
                initial = torch.randn((1, length, backend.embedding.weight.shape[1]), generator=generator) * config["init_std"]
                save_torch(initial_path, {"soft_prefix": initial, "seed": config["seed"]})
            if initial.shape != (1, length, backend.embedding.weight.shape[1]):
                raise ValueError("初始前缀形状与当前样本不一致")
            digest = hashlib.sha256(initial.contiguous().numpy().tobytes()).hexdigest()
            print(f"前缀长度={length} 公共token={public_ids.shape[1]} 问题token={question_ids.shape[1]} "
                  f"初始前缀SHA256={digest}", flush=True)
            print(f"单组TopK前缀数据约={initial.numel() * initial.element_size() * config['save_topk'] / 2**20:.1f}MiB "
                  "（不含模型、优化器与暂存的续跑保护文件）", flush=True)
            write_json(sample_root / "sample_info.json", {"task_id": sample.task_id, "manifest": sample.manifest,
                       "question": sample.question, "answer_diagnostic_only": sample.answer,
                       "output_instruction": sample.output_instruction, "prefix_length": length,
                       "initial_sha256": digest, "context_signature": signature})
            series = {}
            for method_index, method in enumerate(config["methods"], 1):
                if test_only and method not in ready:
                    continue
                directory = sample_root / method
                specific = method_config(config, method)
                label = f"[样本 {sample_index}/{len(samples)} {sample.task_id} | 方法 {method_index}/{len(config['methods'])} {method}]"
                try:
                    if not test_only:
                        question_groups = None
                        if method in WEIGHTED_METHODS and specific["semantic_query_mode"] == "structured":
                            text = question_text(sample.question, sample.output_instruction, specific["answer_boundary"])
                            question_groups = backend.semantic_query_groups(text, specific, question_ids)
                        monitor = AnswerMonitor(backend, observed, question_ids, sample.answer, sample.aliases,
                                                probability_mode=config["score_probability"], leading_spaces=config["answer_leading_spaces"])
                        def update_comparison(rows, name=method):
                            series[name] = rows
                            render_comparison(sample_root / "comparison", series)
                        series[method] = train(backend, observed, public_ids, question_ids, initial, method, specific,
                                               directory, monitor, oracle_cache=oracle if method == "oracle_private_prefix_distillation" else None,
                                               resume=resume, label=label, context_signature=signature,
                                               on_progress=update_comparison, question_groups=question_groups)
                        render_comparison(sample_root / "comparison", series)
                    elif not (directory / "prefixes" / "manifest.json").exists():
                        print(f"{label} 没有已导出前缀，跳过测试", flush=True)
                        continue
                    if config["auto_test"] or test_only:
                        test_method(backend, sample, specific, directory, observed, public_ids, question_ids)
                    completed.append({"task_id": sample.task_id, "method": method})
                    write_json(root / "status.json", {"status": "running", "completed": completed, "failures": failures})
                    summarize_root(root, config, len(samples), announce=False,
                                   task_ids=[sample.task_id for sample in samples])
                except Exception as error:
                    failures.append({"task_id": sample.task_id, "method": method, "error": f"{type(error).__name__}: {error}"})
                    traceback.print_exc()
                    if is_storage_error(error) or not config["continue_on_error"]:
                        raise
                    if backend.device.type == "cuda":
                        torch.cuda.empty_cache()
            if series:
                render_comparison(sample_root / "comparison", series)
        except Exception as error:
            failures.append({"task_id": sample.task_id, "error": f"{type(error).__name__}: {error}"})
            traceback.print_exc()
            if is_storage_error(error) or not config["continue_on_error"]:
                raise
    result = {"status": "completed_with_errors" if failures else "partial" if skipped else "completed", "planned_samples": len(samples),
              "planned_method_runs": len(samples) * len(config["methods"]), "completed": completed, "failures": failures, "skipped": skipped}
    write_json(root / "result.json", result)
    write_json(root / "status.json", result)
    summarize_root(root, config, len(samples), task_ids=[sample.task_id for sample in samples])
    return 1 if failures else 0
