"""Launcher uses only stdlib until GPU visibility and worker mode are resolved."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

from .config import DEFAULTS, METHODS, WEIGHTED_METHODS, load_config, validate, saved_config, require_current_attention_loss, require_current_weighted_kv, method_config
from .data import catalog, materialize
from .io import RunBusy, read_json, run_lock, write_json, write_error_status
from .question_semantics import semantic_spans

ROOT = Path(__file__).resolve().parents[1]


def parser():
    result = argparse.ArgumentParser(description="单软前缀连续训练：独立方法与联合组、三类图、损失 TopK 导出")
    result.add_argument("command", nargs="?", choices=("train", "test", "inspect", "diagnose"), default="train")
    result.add_argument("--config", type=Path, default=None)
    result.add_argument("--run-dir", type=Path, help="指定新运行目录；test 使用已有运行目录")
    result.add_argument("--resume", type=Path, help="恢复已有运行，使用其中保存的完整配置")
    result.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--foreground", dest="background", action="store_false")
    modes.add_argument("--background", dest="background", action="store_true")
    result.set_defaults(background=None)
    for flag in ("model-path", "device", "cuda-devices", "output-dir"):
        result.add_argument("--" + flag)
    result.add_argument("--base-loss", choices=("mean", "token", "none"))
    result.add_argument("--checkpoint-metric", choices=("base", "total"))
    result.add_argument("--methods", nargs="+", choices=METHODS)
    result.add_argument("--datasets", nargs="+")
    result.add_argument("--include-tasks", nargs="+")
    result.add_argument("--exclude-tasks", nargs="+")
    for flag in ("seed", "rounds", "steps-per-round", "save-topk", "log-every", "eval-every", "limit", "max-new-tokens"):
        result.add_argument("--" + flag, type=int)
    result.add_argument("--lr", type=float)
    return result


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def execute(config, root, resume, test_only, worker=False, source=None):
    try:
        with run_lock(root):
            return execute_locked(config, root, resume, test_only, worker, source)
    except RunBusy as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2


def execute_locked(config, root, resume, test_only, worker=False, source=None):
    # Import torch only after CUDA_VISIBLE_DEVICES is established.
    if config["cuda_devices"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_devices"]
    log_name = "diagnostics.log" if source else "test.log" if test_only else "train.log"
    with (root / log_name).open("a", encoding="utf-8") as log:
        previous_out, previous_err = sys.stdout, sys.stderr
        sys.stdout = Tee(log) if worker else Tee(previous_out, log)
        sys.stderr = Tee(log) if worker else Tee(previous_err, log)
        try:
            if source:
                from .diagnostics import run_diagnostics
                with run_lock(source):
                    return run_diagnostics(config, root, source)
            from .runner import run
            return run(config, root, resume=resume, test_only=test_only)
        except (KeyboardInterrupt, Exception) as error:
            try:
                traceback.print_exc()
            except Exception:
                pass
            write_error_status(root / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                              "error": f"{type(error).__name__}: {error}"})
            return 130 if isinstance(error, KeyboardInterrupt) else 1
        finally:
            sys.stdout, sys.stderr = previous_out, previous_err


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    if args.worker:
        root = args.worker.resolve()
        job = read_json(root / "worker.json")
        raw = read_json(root / "resolved_config.json")
        config = validate(saved_config(raw),
                          allow_retired=bool(job["test_only"] or job.get("source")))
        if not job["test_only"] and not job.get("source") and config["experiment_version"] != DEFAULTS["experiment_version"]:
            raise ValueError("旧运行不能使用新版问题/概率口径续训；请新建实验或用 diagnose 重评")
        if not job["test_only"] and not job.get("source"):
            require_current_attention_loss(config)
            require_current_weighted_kv(config)
        return execute(config, root, job["resume"], job["test_only"], worker=True, source=job.get("source"))
    test_only = args.command == "test"
    source = None
    if args.command == "diagnose":
        if not args.run_dir or args.resume or args.config:
            raise ValueError("diagnose 使用 --run-dir 指定原运行目录，不接受 --resume/--config")
        source = args.run_dir.resolve()
        allowed = {"model_path", "device", "cuda_devices", "max_new_tokens", "background"}
        supplied = {key: getattr(args, key) for key in DEFAULTS if hasattr(args, key) and getattr(args, key) is not None}
        if set(supplied) - allowed:
            raise ValueError("重评只允许覆盖模型路径、GPU、输出长度与前后台模式")
        config = saved_config(read_json(source / "resolved_config.json"))
        config.update(answer_boundary=DEFAULTS["answer_boundary"], stop_strings=DEFAULTS["stop_strings"],
                      score_probability=DEFAULTS["score_probability"],
                      answer_leading_spaces=DEFAULTS["answer_leading_spaces"], experiment_version=DEFAULTS["experiment_version"],
                      max_new_tokens=DEFAULTS["max_new_tokens"])
        config.update(supplied)
        config = validate(config, allow_retired=True)
        root = source / ("diagnostics_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    elif args.resume or test_only:
        if not (args.resume or args.run_dir):
            raise ValueError("test 需要 --run-dir")
        root = (args.resume or args.run_dir).resolve()
        config = read_json(root / "resolved_config.json")
        if args.resume and config.get("experiment_version", 1) != DEFAULTS["experiment_version"]:
            raise ValueError("旧运行不能使用新版问题/概率口径续训；请新建实验或用 diagnose 重评")
        if "answer_boundary" not in config:
            raise ValueError("旧运行使用不同回答边界，不能直接混用新训练/测试；请用 diagnose --run-dir 重评，或新建实验")
        config = validate(saved_config(config),
                          allow_retired=test_only and not args.resume)
        supplied = [key for key in DEFAULTS if hasattr(args, key) and getattr(args, key) is not None and key != "background"]
        if supplied or args.config:
            raise ValueError("恢复/重测使用保存的配置，不接受训练参数覆盖")
    else:
        overrides = {key: getattr(args, key) for key in DEFAULTS if hasattr(args, key) and getattr(args, key) is not None}
        if args.datasets:
            overrides["datasets"] = [str(Path(value).expanduser().resolve()) for value in args.datasets]
        if args.output_dir:
            overrides["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
        config = load_config(args.config or ROOT / "configs" / "default.json", overrides)
        root = args.run_dir.resolve() if args.run_dir else Path(config["output_dir"]) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    if args.background is not None:
        config["background"] = args.background
    if args.command == "train":
        require_current_attention_loss(config)
        require_current_weighted_kv(config)
    selected = catalog(config)
    for row in selected:
        sample = materialize(row, config["output_instruction"])
        if config["auto_test"] and config["test_question_mode"] == "held_out" and not sample.held_out_question:
            raise ValueError(f"{sample.task_id}: held_out 测试需要独立的 question_test 问题")
        if args.command in ("train", "inspect"):
            for method in config["methods"]:
                if method in WEIGHTED_METHODS and method_config(config, method)["semantic_query_mode"] == "structured":
                    semantic_spans(sample.question)
    if args.command == "inspect":
        print(json.dumps({"samples": [r["task_id"] for r in selected], "sample_count": len(selected), "config": config}, ensure_ascii=False, indent=2))
        return 0
    if not (args.resume or test_only):
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "resolved_config.json", config)
    if config["background"]:
        try:
            with run_lock(root):
                write_json(root / "worker.json", {"resume": bool(args.resume), "test_only": test_only,
                                                  "source": str(source) if source else None})
                write_json(root / "status.json", {"status": "starting"})
        except RunBusy as error:
            print(str(error), file=sys.stderr)
            return 2
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        if config["cuda_devices"] is not None:
            environment["CUDA_VISIBLE_DEVICES"] = config["cuda_devices"]
        options = {"env": environment, "cwd": str(ROOT), "stdin": subprocess.DEVNULL}
        if os.name == "nt":
            options["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            options["start_new_session"] = True
        with (root / "launcher.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", str(ROOT / "run.py"), "--worker", str(root)], stdout=log, stderr=log, **options)
        write_json(root / "process.json", {"pid": process.pid, "run_dir": str(root)})
        log_name = "diagnostics.log" if source else "test.log" if test_only else "train.log"
        print(f"后台进程 PID={process.pid}\n运行目录: {root}\n日志: {root / log_name}")
        return 0
    return execute(config, root, bool(args.resume), test_only, source=source)
