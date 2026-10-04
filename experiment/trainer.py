"""One optimizer and one prefix, trained for the configured fixed update budget."""
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .checkpoints import TopK
from .config import METHODS, require_current_attention_loss, require_current_weighted_kv
from .io import save_torch, write_csv, write_json, write_error_status
from .metrics import SupportScore
from .objectives import Objective
from .plots import render
from .question_semantics import effective_shares


def duration(seconds):
    return f"{int(seconds) // 3600:02d}:{int(seconds) // 60 % 60:02d}:{int(seconds) % 60:02d}"


def fingerprint(config, method, context_signature, question_groups=None):
    ignored = {"background", "config_path", "output_dir", "continue_on_error", "auto_test"}
    if method != "question_attention_reconstruction":
        # Attention's revision does not change the other methods' objectives.
        ignored.update(("attention_loss_version", "lambda_attention"))
    if method != "question_weighted_kv":
        ignored.update(("weighted_kv_version", "semantic_query_mode", "semantic_query_weights"))
    relevant = {k: v for k, v in config.items() if k not in ignored}
    if method == "question_weighted_kv" and question_groups is not None:
        relevant["semantic_query_positions"] = question_groups
    return hashlib.sha256(json.dumps([relevant, method, context_signature], sort_keys=True).encode()).hexdigest()


def train(backend, observed, public_ids, question_ids, initial, method, config, root, monitor,
          *, oracle_cache=None, resume=False, label="", context_signature="", on_progress=None, question_groups=None):
    if method not in METHODS:
        raise ValueError(f"方法 {method} 不属于当前实验组，禁止新训练或续训")
    require_current_attention_loss({**config, "methods": [method]})
    require_current_weighted_kv({**config, "methods": [method]})
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    latest = root / "latest.pt"
    prefix = torch.nn.Parameter(initial.to(device=backend.device, dtype=torch.float32).clone())
    optimizer = torch.optim.Adam([prefix], lr=config["lr"])
    objective = Objective(backend, observed, public_ids, question_ids, method, config, oracle_cache, question_groups=question_groups)
    pool = TopK(root / "prefixes", config["save_topk"], config["checkpoint_metric"])
    total_steps = config["rounds"] * config["steps_per_round"]
    signature = fingerprint(config, method, context_signature, question_groups)
    rows, start_step, elapsed_before = [], 0, 0.0
    probability_key = "answer_probability"
    score = None
    if resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["fingerprint"] != signature:
            raise ValueError("恢复配置或样本上下文与原训练不一致")
        with torch.no_grad():
            prefix.copy_(state["prefix"])
        optimizer.load_state_dict(state["optimizer"])
        objective.load_state_dict(state["objective"])
        pool.load_state_dict(state["topk"])
        rows, start_step, elapsed_before = state["history"], state["step"], state["elapsed"]
        score = SupportScore(config, state["score"]["initial"])
        score.load_state_dict(state["score"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        if backend.device.type == "cuda" and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in state["rng_cuda"]])
        print(f"{label} 恢复 step={start_step}/{total_steps}", flush=True)
    elif (root / "history.jsonl").exists():
        raise ValueError("输出目录已有训练记录；请使用 --resume")
    recorded_config = {"method": method, "uses_private_teacher": oracle_cache is not None, "config": config}
    if objective.query_weights is not None:
        shares = effective_shares(question_groups, config["semantic_query_weights"])
        recorded_config.update(semantic_query_positions=question_groups, query_token_coefficients=objective.query_weights,
                               effective_semantic_query_weights=shares)
        counts = {group: len(indices) for group, indices in question_groups.items()}
        print(f"{label} 语义查询 token 数={counts} 贡献比例={shares}", flush=True)
    write_json(root / "config.json", recorded_config)
    history_path = root / "history.jsonl"
    history_path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    started = time.monotonic()

    def save_resume(step):
        save_torch(latest, {"prefix": prefix.detach().cpu().clone(), "optimizer": optimizer.state_dict(),
                           "objective": objective.state_dict(), "topk": pool.state_dict(), "history": rows,
                           "score": score.state_dict(), "step": step, "fingerprint": signature,
                           "elapsed": elapsed_before + time.monotonic() - started, "rng_cpu": torch.get_rng_state(),
                           "rng_cuda": torch.cuda.get_rng_state_all() if backend.device.type == "cuda" else None})
        pool.commit_resume()

    if not rows:
        objective.refresh(prefix, 0)
        with torch.no_grad():
            _, parts, private = objective.compute(prefix)
            diagnosis = monitor(private)
        score = SupportScore(config, diagnosis[probability_key])
        initial_row = {"step": 0, "round": 0, **{k: float(v) for k, v in parts.items()}, **diagnosis,
                       **score.add(0, diagnosis[probability_key]), "diagnostics_position": "initial", "reference_step": objective.reference_step}
        rows.append(initial_row)
        with history_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(initial_row, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"{label} 初始化 base_loss={initial_row['base_loss']:.6g} answer_probability={initial_row['answer_probability']:.6%}", flush=True)
        save_resume(0)
    last_step = start_step
    try:
        for step in range(start_step + 1, total_steps + 1):
            if objective.refresh(prefix, step - 1):
                print(f"{label} 参考刷新 step={step - 1}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = objective.compute(prefix)
            if not torch.isfinite(loss):
                raise RuntimeError(f"step={step}: 训练损失非有限值")
            loss.backward()
            if prefix.grad is None or not torch.isfinite(prefix.grad).all():
                raise RuntimeError(f"step={step}: 软前缀梯度缺失或非有限值")
            torch.nn.utils.clip_grad_norm_([prefix], config["grad_clip"])
            optimizer.step()
            del loss
            record = step % config["checkpoint_every"] == 0 or step % config["log_every"] == 0 or step % config["eval_every"] == 0 or step == total_steps
            if record:
                # Recompute AFTER update. Saved tensors and plotted losses have the same state.
                with torch.no_grad():
                    _, parts, private = objective.compute(prefix)
                    row = {"step": step, "round": (step - 1) // config["steps_per_round"] + 1,
                           "round_step": (step - 1) % config["steps_per_round"] + 1,
                           **{k: float(v) for k, v in parts.items()}, "diagnostics_position": "after_update", "reference_step": objective.reference_step}
                    if step % config["eval_every"] == 0 or step == total_steps:
                        row.update(monitor(private))
                        row.update(score.add(step, row[probability_key]))
                if not all(math.isfinite(row[k]) for k in parts):
                    raise RuntimeError(f"step={step}: 更新后损失非有限值")
                if step % config["checkpoint_every"] == 0 or step == total_steps:
                    metric = row["base_loss" if config["checkpoint_metric"] == "base" else "total_loss"]
                    pool.consider(step, prefix, metric)
                rows.append(row)
                with history_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                if step % config["log_every"] == 0 or step == total_steps:
                    diagnostic = next(r for r in reversed(rows) if "answer_probability" in r)
                    elapsed = time.monotonic() - started
                    speed = (step - start_step) / max(elapsed, 1e-9)
                    eta = (total_steps - step) / max(speed, 1e-9)
                    detail = " ".join(f"{k}={v:.6g}" for k, v in row.items() if k.endswith("loss"))
                    score_text = "暂不足" if diagnostic["support_score"] is None else f"{diagnostic['support_score']:.6g}"
                    ema_text = "暂不足" if diagnostic["ema_score"] is None else f"{diagnostic['ema_score']:.6g}"
                    canonical = diagnostic.get("canonical_answer_probability", diagnostic["answer_probability"])
                    best = pool.entries[0]["loss"] if pool.entries else None
                    print(f"{label} round={row['round']}/{config['rounds']} step={row['round_step']}/{config['steps_per_round']} "
                          f"updates={step}/{total_steps} progress={100 * step / total_steps:.2f}% lr={config['lr']}\n"
                          f"  {detail}\n  answer_probability={diagnostic['answer_probability']:.6%} "
                          f"canonical_answer_probability={canonical:.6%} "
                          f"first_token_probability={diagnostic['first_token_probability']:.6%} support_score={score_text} EMA={ema_text} "
                          f"evaluated_step={diagnostic['step']}\n  saved={len(pool.entries)}/{config['save_topk']} best_loss={best} "
                          f"elapsed={duration(elapsed_before + elapsed)} speed={speed:.3f}steps/s ETA={duration(eta)}", flush=True)
            last_step = step
            if step % config["resume_every"] == 0 or step == total_steps:
                save_resume(step)
            if step % config["plot_every"] == 0 or step == total_steps:
                render(root / "figures", {method: rows})
                if on_progress:
                    on_progress(rows)
            write_json(root / "status.json", {"status": "running", "step": step, "total_steps": total_steps})
    except (KeyboardInterrupt, Exception) as error:
        # Preserve only fully observed optimizer states. Existing latest.pt is crash-safe.
        write_error_status(root / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                         "step": last_step, "error": f"{type(error).__name__}: {error}"})
        raise
    write_csv(root / "history.csv", rows)
    render(root / "figures", {method: rows})
    result = {"status": "completed", "steps": total_steps, "method": method,
              "score_probability": config["score_probability"], "probability_definition": rows[0]["probability_definition"],
              "checkpoint_metric": config["checkpoint_metric"], "prefix_count": len(pool.entries),
              "prefix_manifest": str(root / "prefixes" / "manifest.json"),
              "elapsed_seconds": elapsed_before + time.monotonic() - started}
    write_json(root / "result.json", result)
    write_json(root / "status.json", result)
    print(f"{label} 完成 updates={total_steps} saved={len(pool.entries)} manifest={result['prefix_manifest']}", flush=True)
    return rows
