"""Re-evaluate saved prefixes under the current boundary without resuming training."""
import json
from pathlib import Path

import torch

from .backend import load_backend
from .data import catalog, materialize
from .io import is_storage_error, read_json, write_csv, write_json
from .metrics import AnswerMonitor, SupportScore
from .plots import render_comparison
from .runner import prepare, summarize_root, test_method


def run_diagnostics(config, root, source, *, backend=None):
    root, source = Path(root), Path(source)
    if not any(source.glob("samples/*/*/prefixes/manifest.json")):
        raise FileNotFoundError("重评需要原运行目录中的前缀 .pt、manifest.json 和 initial.pt；图片/日志不能恢复概率")
    samples = [materialize(row, config["output_instruction"]) for row in catalog(config)]
    original_ids = read_json(source / "selected_tasks.json")
    if original_ids != [sample.task_id for sample in samples]:
        raise ValueError("重评选中的样本与原运行不一致")
    print(f"重评来源={source} 输出={root}；仅重算已保存快照，未保存步骤不可恢复", flush=True)
    print("重评配置:\n" + json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    backend = backend or load_backend(config)
    failures, skipped, completed = [], [], []
    for sample in samples:
        source_sample = source / "samples" / sample.task_id
        methods = [method for method in config["methods"]
                   if (source_sample / method / "prefixes" / "manifest.json").exists()
                   and read_json(source_sample / method / "prefixes" / "manifest.json").get("prefixes")]
        if not methods:
            skipped.append(sample.task_id)
            continue
        try:
            observed, length, _oracle, public_ids, question_ids, signature = prepare(backend, sample, config)
            info = read_json(source_sample / "sample_info.json")
            if info["context_signature"] != signature:
                raise ValueError("原运行的样本文本与当前数据集不同，不能混用前缀")
            initial = torch.load(source_sample / "initial.pt", map_location="cpu", weights_only=True)["soft_prefix"]
            if initial.shape != (1, length, backend.embedding.weight.shape[1]):
                raise ValueError("初始前缀形状与当前模型或样本不一致")
            monitor = AnswerMonitor(backend, observed, question_ids, sample.answer, sample.aliases,
                                    probability_mode=config["score_probability"], leading_spaces=config["answer_leading_spaces"])
            with torch.no_grad():
                initial_private, _public = backend.student(initial.to(backend.device), public_ids)
                initial_metrics = monitor(initial_private)
            series = {}
            for method in methods:
                directory = source_sample / method
                destination = root / "samples" / sample.task_id / method
                destination.mkdir(parents=True, exist_ok=True)
                score = SupportScore(config, initial_metrics["answer_probability"])
                rows = [{"step": 0, **initial_metrics, **score.add(0, initial_metrics["answer_probability"]),
                         "diagnostic_scope": "saved_snapshots_only"}]
                manifest = read_json(directory / "prefixes" / "manifest.json")
                items = sorted(manifest["prefixes"], key=lambda item: item["step"])
                available = []
                for index, item in enumerate(items, 1):
                    if Path(item["path"]).name != item["path"]:
                        raise ValueError("前缀文件名无效")
                    path = directory / "prefixes" / item["path"]
                    if not path.is_file():
                        failures.append({"task_id": sample.task_id, "method": method, "step": item["step"], "error": f"缺少前缀文件: {path}"})
                        print(f"[重评跳过] 缺少 {path}", flush=True)
                        continue
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    with torch.no_grad():
                        private, _public = backend.student(payload["soft_prefix"].to(backend.device), public_ids)
                        metrics = monitor(private)
                    row = {"step": item["step"], "checkpoint_loss": item["loss"], **metrics,
                           **score.add(item["step"], metrics["answer_probability"]), "diagnostic_scope": "saved_snapshots_only"}
                    rows.append(row)
                    available.append(item)
                    if index % config["log_every"] == 0 or index == len(items):
                        print(f"[重评 {sample.task_id}/{method}] {index}/{len(items)} step={item['step']} "
                              f"sequence_probability={metrics['answer_probability']:.6%}", flush=True)
                write_csv(destination / "history.csv", rows)
                (destination / "history.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
                write_json(destination / "diagnostics.json", {"source": str(directory), "scope": "saved_snapshots_only",
                           "score_scope": "available_observations_only", "snapshot_count": len(available),
                           "answer_boundary": config["answer_boundary"], "score_probability": config["score_probability"],
                           "probability_definition": initial_metrics["probability_definition"],
                           "answer_leading_spaces": config["answer_leading_spaces"]})
                render_comparison(destination / "figures", {method: rows}, snapshot_only=True)
                series[method] = rows
                render_comparison(root / "samples" / sample.task_id / "comparison", series, snapshot_only=True)
                if config["auto_test"]:
                    test_method(backend, sample, config, directory, observed, public_ids, question_ids,
                                output_directory=destination, prefix_items=available)
                completed.append({"task_id": sample.task_id, "method": method})
        except Exception as error:
            failures.append({"task_id": sample.task_id, "error": f"{type(error).__name__}: {error}"})
            if is_storage_error(error) or not config["continue_on_error"]:
                raise
            print(f"[重评失败 {sample.task_id}] {error}", flush=True)
    result = {"status": "completed_with_errors" if failures else "partial" if skipped else "completed",
              "source": str(source), "scope": "saved_snapshots_only", "completed": completed,
              "failures": failures, "skipped": skipped}
    write_json(root / "result.json", result)
    write_json(root / "status.json", result)
    if config["auto_test"]:
        summarize_root(root, config, len(samples))
    return 1 if failures else 0
