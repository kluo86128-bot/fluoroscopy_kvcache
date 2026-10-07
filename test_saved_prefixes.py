"""Test every exported prefix with one sample; no original run is required.

python -u test_saved_prefixes.py --sample-dir test_samples/sgd_test_021 \
    --prefix-dir /path/to/prefixes --cuda-devices 3
Use --inspect to check inputs without loading a model.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from experiment.answer_forms import FORMATS
from experiment.config import DEFAULTS
from experiment.data import Sample, question_text
from experiment.io import read_json, write_csv, write_json, write_error_status, is_storage_error

ROOT = Path(__file__).resolve().parent
EVALUATION_KEYS = ("model_path", "device", "cuda_devices", "dtype", "attention_backend", "local_files_only",
                   "output_instruction", "question_format", "enable_thinking", "answer_boundary",
                   "stop_strings", "max_new_tokens")


def discover_prefixes(directory, recursive=True):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"前缀目录不存在: {root}")
    files, excluded = [], []
    for path in sorted(root.rglob("*") if recursive else root.iterdir()):
        if path.is_file() and path.suffix.lower() in (".pt", ".pth"):
            # These contain initialization or optimizer state, not exported snapshots.
            (excluded if path.name in ("initial.pt", "latest.pt") else files).append(path)
    if not files:
        raise ValueError(f"没有 .pt/.pth 导出前缀: {root}")
    return files, excluded


def evaluation_config(prefix_dir, config_file=None, overrides=None):
    source = Path(config_file).expanduser().resolve() if config_file else None
    if source is None:
        root = Path(prefix_dir).expanduser().resolve()
        for parent in (root, *root.parents):
            for name in ("resolved_config.json", "config.json"):
                candidate = parent / name
                if candidate.is_file():
                    source = candidate
                    break
            if source:
                break
    config = {key: deepcopy(DEFAULTS[key]) for key in EVALUATION_KEYS}
    if source:
        raw = read_json(source)
        raw = raw.get("config", raw)
        config.update({key: raw[key] for key in EVALUATION_KEYS if key in raw})
    config.update({key: value for key, value in (overrides or {}).items() if value is not None})
    if config["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens 必须为正整数")
    if not isinstance(config["answer_boundary"], str) or not config["answer_boundary"].endswith((" ", "\n", "\t")):
        raise ValueError("answer_boundary 必须以空白结束")
    if not isinstance(config["stop_strings"], list) or any(not isinstance(s, str) or not s for s in config["stop_strings"]):
        raise ValueError("stop_strings 必须是非空字符串列表")
    return config, source


def load_sample(directory, config, question_file=None, answer_format=None):
    root = Path(directory).expanduser().resolve()
    evaluation = read_json(root / "evaluation.json")
    task_id = evaluation.get("task_id", root.name)
    row = {}
    manifest = root.parent / "tasks.jsonl"
    if manifest.is_file():
        candidates = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        row = next((item for item in candidates if item["task_id"] == task_id), {})
    path = Path(question_file).expanduser().resolve() if question_file else root / "question_test.txt"
    texts = [(root / filename).read_text(encoding="utf-8-sig") for filename in ("prefix_a.txt", "public_chunk.txt")]
    question = path.read_text(encoding="utf-8-sig")
    answer = evaluation["canonical_answer"]
    aliases = evaluation.get("accepted_aliases", [])
    if any(not isinstance(text, str) or not text.strip() for text in (*texts, question, answer)):
        raise ValueError("样本的私有文本、公共文本、问题和评估答案必须为非空字符串")
    if not isinstance(aliases, list) or any(not isinstance(text, str) or not text.strip() for text in aliases):
        raise ValueError("accepted_aliases 必须为字符串列表")
    fmt = answer_format or row.get("answer_format", evaluation.get("answer_format", "free_text"))
    if fmt not in FORMATS:
        raise ValueError(f"未知 answer_format: {fmt}")
    instruction = row.get("receiver_inputs", {}).get("output_instruction", row.get("output_instruction", config["output_instruction"]))
    sample = Sample(task_id, *texts, question, answer, aliases, None, None, [], instruction,
                    str(manifest), fmt)
    kind = "held_out_question" if path.name == "question_test.txt" else "training_question" if path.name == "question_search.txt" else "custom_question"
    return sample, path, kind


def check_provenance(path, sample):
    for parent in path.parents:
        if parent.name.startswith("sgd_test_") and parent.name != sample.task_id:
            raise ValueError(f"前缀路径属于其他样本: {parent.name}")
    info_path = next((parent / "sample_info.json" for parent in path.parents
                      if (parent / "sample_info.json").is_file()), None)
    if info_path is None:
        return False
    info = read_json(info_path)
    if info.get("task_id", sample.task_id) != sample.task_id:
        raise ValueError(f"前缀属于其他样本: {info.get('task_id')}")
    if all(key in info for key in ("context_signature", "question", "output_instruction")):
        # Use the saved training question for this check: a new test question is allowed.
        context = sample.private_text + "\0" + sample.public_text + "\0" + info["question"] + "\0" + info["output_instruction"]
        if hashlib.sha256(context.encode()).hexdigest() != info["context_signature"]:
            raise ValueError("样本的私有/公共上下文与来源前缀不一致")
    return True


def load_prefix(path, expected_shape):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload if isinstance(payload, dict) else {}
    prefix = next((metadata[key] for key in ("soft_prefix", "soft_prefix_embeddings") if key in metadata),
                  payload if not metadata else None)
    if not isinstance(prefix, torch.Tensor) or tuple(prefix.shape) != expected_shape:
        raise ValueError(f"前缀必须是浮点张量，形状为 {expected_shape}")
    if not prefix.is_floating_point() or not torch.isfinite(prefix).all().item():
        raise ValueError("前缀包含非浮点数或 NaN/Inf")
    return prefix.detach(), {key: metadata[key] for key in ("step", "checkpoint_loss", "checkpoint_metric") if key in metadata}


def save_results(root, rows, total, excluded, status):
    valid = [row for row in rows if row["status"] == "tested"]
    correct = sum(row["answer_match"] for row in valid)
    report = {"status": status, "candidate_prefixes": total, "completed_prefixes": len(valid),
              "successful_prefixes": correct, "prefix_success_rate": correct / len(valid) if valid else None,
              "sample_success": bool(correct) if valid else None,
              "invalid_prefixes": sum(row["status"] == "invalid_prefix" for row in rows),
              "inference_errors": sum(row["status"] == "inference_error" for row in rows),
              "unprocessed_prefixes": total - len(rows), "excluded_state_files": [str(path) for path in excluded]}
    write_json(root / "summary.json", report)
    write_json(root / "results.json", rows)
    write_csv(root / "results.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["prefix_folder"]].append(row)
    folder_rows = []
    for folder, items in sorted(groups.items()):
        tested = [row for row in items if row["status"] == "tested"]
        successes = sum(row["answer_match"] for row in tested)
        folder_rows.append({"prefix_folder": folder, "attempted_prefixes": len(items),
                            "tested_prefixes": len(tested), "correct_prefixes": successes,
                            "prefix_success_rate": successes / len(tested) if tested else None,
                            "sample_success": bool(successes) if tested else None,
                            "invalid_prefixes": sum(row["status"] == "invalid_prefix" for row in items),
                            "inference_errors": sum(row["status"] == "inference_error" for row in items)})
    write_csv(root / "folder_summary.csv", folder_rows)
    return report


def evaluate(sample, files, excluded, config, output_dir, *, question_path, question_kind, backend=None):
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "evaluation_config.json", {"config": config, "task_id": sample.task_id,
               "question_file": str(question_path), "question": sample.question,
               "answer_format": sample.answer_format, "prefix_files": [str(path) for path in files],
               "decode": "greedy", "data_scope": "inferred_prefix_KV_plus_observed_public_KV"})
    rows = []
    try:
        import torch
        from experiment.backend import load_backend
        from experiment.metrics import test_prefix
        backend = backend or load_backend(config)
        public_ids = backend.encode(sample.public_text)
        observed, length, _ = backend.setup_teacher(backend.encode(sample.private_text), public_ids, include_private=False)
        ids = backend.encode_question(question_text(sample.question, sample.output_instruction, config["answer_boundary"]), config)
        shape = (1, length, backend.embedding.weight.shape[1])
        with (root / "results.jsonl").open("w", encoding="utf-8") as output, torch.no_grad():
            for index, path in enumerate(files, 1):
                row = {"task_id": sample.task_id, "question_kind": question_kind, "prefix_path": str(path),
                       "prefix_folder": str(path.parent), "correct_answer": sample.answer}
                try:
                    row["source_context_verified"] = check_provenance(path, sample)
                    prefix, metadata = load_prefix(path, shape)
                    row.update(metadata)
                except Exception as error:
                    if is_storage_error(error):
                        raise
                    row.update(status="invalid_prefix", error=f"{type(error).__name__}: {error}")
                else:
                    try:
                        row.update(test_prefix(backend, observed, public_ids, ids, prefix, sample.answer,
                                               sample.aliases, config["max_new_tokens"], config["stop_strings"], sample.answer_format))
                        row["status"] = "tested"
                    except Exception as error:
                        row.update(status="inference_error", error=f"{type(error).__name__}: {error}")
                        rows.append(row)
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        raise
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                detail = f"answer={row['answer_match']} output={row['output']!r}" if row["status"] == "tested" else row["error"]
                print(f"[测试 {index}/{len(files)}] {path} {detail}", flush=True)
        report = save_results(root, rows, len(files), excluded,
                              "completed_with_errors" if any(row["status"] != "tested" for row in rows) else "completed")
        rate = "未测试" if report["prefix_success_rate"] is None else f"{report['prefix_success_rate']:.2%}"
        print(f"[汇总] 已测试={report['completed_prefixes']}/{len(files)} 正确={report['successful_prefixes']} "
              f"前缀成功率={rate} 无效={report['invalid_prefixes']} 输出目录={root}", flush=True)
        return 1 if report["invalid_prefixes"] else 0
    except BaseException as error:
        try:
            save_results(root, rows, len(files), excluded, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        except Exception:
            pass
        write_error_status(root / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                  "error": f"{type(error).__name__}: {error}"})
        raise


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="指定单个样本与前缀根目录，递归测试全部导出前缀，仅推理")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--prefix-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="可选模型/推理配置；默认查找前缀目录及父目录的配置")
    parser.add_argument("--question-file", type=Path, help="默认使用样本目录的question_test.txt，可指定其他问题")
    parser.add_argument("--answer-format", choices=FORMATS, help="默认从样本manifest读取；搬离manifest时默认free_text")
    parser.add_argument("--model-path")
    parser.add_argument("--cuda-devices", help="物理GPU，例如3；在导入torch前设置")
    parser.add_argument("--device", choices=("cpu", "cuda:0"))
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--question-format", choices=("raw", "chat"))
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--output-dir", type=Path, help="必须是尚不存在的独立目录")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="只检查样本与文件列表，不加载模型")
    args = parser.parse_args(argv)
    config, source = evaluation_config(args.prefix_dir, args.config,
                                      {key: getattr(args, key) for key in EVALUATION_KEYS if hasattr(args, key)})
    sample, question_path, kind = load_sample(args.sample_dir, config, args.question_file, args.answer_format)
    files, excluded = discover_prefixes(args.prefix_dir, not args.no_recursive)
    root = (args.output_dir or ROOT / "outputs/prefix_tests" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])).expanduser().resolve()
    if root.exists() or any(root.is_relative_to(path.expanduser().resolve()) for path in (args.sample_dir, args.prefix_dir)):
        raise ValueError("输出目录必须尚不存在，且位于样本/前缀目录之外")
    print(json.dumps({"task_id": sample.task_id, "question_file": str(question_path), "question_kind": kind,
                      "candidate_prefixes": len(files), "excluded_state_files": len(excluded),
                      "source_config": str(source) if source else None, "device": config["device"],
                      "cuda_devices": config["cuda_devices"], "model_path": config["model_path"],
                      "output_dir": str(root)}, ensure_ascii=False, indent=2), flush=True)
    if args.inspect:
        return 0
    if config["cuda_devices"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_devices"]
    return evaluate(sample, files, excluded, config, root, question_path=question_path, question_kind=kind)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"测试失败: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
