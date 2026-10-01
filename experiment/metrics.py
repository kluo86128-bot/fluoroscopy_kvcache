"""Answer labels stay in diagnostics/testing, outside the optimization objective."""
import math
import re

import numpy as np
import torch

from .backend import join_cache


class AnswerMonitor:
    def __init__(self, backend, observed, question_ids, answer):
        self.backend, self.observed, self.question_ids = backend, observed, question_ids
        self.answer_ids = backend.encode(answer)

    @torch.no_grad()
    def __call__(self, prefix_cache):
        logits = self.backend.continuation_logits(join_cache(prefix_cache, self.observed), self.question_ids, self.answer_ids)
        log_probs = logits.log_softmax(-1)
        values = log_probs.gather(-1, self.answer_ids.unsqueeze(-1))[0, :, 0].double()
        log_probability = float(values.sum())
        return {"first_token_probability": math.exp(float(values[0])),
                "answer_probability": math.exp(log_probability), "answer_log_probability": log_probability,
                "answer_tokens": self.answer_ids.shape[1], "probability_path": "prefix_plus_observed_public",
                "probability_definition": "canonical_answer_sequence_product_excluding_boundary_and_stop"}


class SupportScore:
    def __init__(self, config, initial):
        self.config, self.initial = config, initial
        self.observations, self.last_ema_step, self.ema = [], None, None

    def add(self, step, probability):
        if self.observations and step <= self.observations[-1][0]:
            raise ValueError("评分观测必须按训练步数递增")
        self.observations.append((step, probability))
        window = self.config["score_window_steps"]
        rows = [(s, p) for s, p in self.observations if step - window < s <= step]
        if step < window or len(rows) < 2:
            return {"support_score": None, "ema_score": None, "score_status": "provisional_window"}
        values = np.asarray([p for _, p in rows], dtype=float)
        count = max(1, math.ceil(len(values) * self.config["score_tail_fraction"]))
        stable = float(np.quantile(values[-count:], 0.25))
        gain = max(0.0, stable - self.initial)
        preceding = [p for s, p in self.observations if s <= step - window]
        high = max(preceding[-1] if preceding else self.initial, float(np.quantile(values, 0.95)), 1e-30)
        retention = min(1.0, stable / high)
        score = 100 * stable * gain * retention
        if self.ema is None:
            self.ema = score
        else:
            delta = step - self.last_ema_step
            alpha = 1 - (1 - self.config["ema_alpha"]) ** (delta / self.config["ema_span_steps"])
            self.ema = alpha * score + (1 - alpha) * self.ema
        self.last_ema_step = step
        return {"support_score": score, "ema_score": self.ema, "stable_probability": stable,
                "positive_gain": gain, "retention": retention, "score_status": "single_trajectory_support"}

    def state_dict(self):
        return {"initial": self.initial, "observations": self.observations, "last_ema_step": self.last_ema_step, "ema": self.ema}

    def load_state_dict(self, state):
        self.initial, self.observations, self.last_ema_step, self.ema = (state[k] for k in ("initial", "observations", "last_ema_step", "ema"))


def answer_match(text, answer, aliases):
    accepted = {x.strip().casefold() for x in [answer, *aliases]}
    full = text.strip().casefold() in accepted
    # Deterministic extraction: first nonempty line, common label, first numeric value.
    line = next((x.strip() for x in text.splitlines() if x.strip()), "")
    line = re.sub(r"^(?:answer|agent|答案)\s*[:：]\s*", "", line, flags=re.I)
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", answer.strip()):
        found = re.search(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w.])", line)
        extracted = found.group(0) if found else line
    else:
        extracted = line.strip().rstrip("。.!！")
    return {"full_match": full, "answer_match": extracted.casefold() in accepted, "extracted_answer": extracted,
            "format_compliant": full}


@torch.no_grad()
def test_prefix(backend, observed, public_ids, question_ids, prefix, answer, aliases, max_tokens, stop_strings=()):
    private, _ = backend.student(prefix.to(device=backend.device), public_ids)
    generated = backend.rollout(join_cache(private, observed), question_ids, max_tokens, stop_strings=stop_strings)
    values = generated[0].tolist()
    text = backend.tokenizer.decode(values, skip_special_tokens=True)
    stops = [text.find(stop) for stop in stop_strings if stop in text]
    if stops:
        text = text[:min(stops)]
    answer_ids = backend.encode(answer)[0].tolist()
    return {"output": text, "generated_token_ids": values,
            "first_token_match": bool(values and values[0] == answer_ids[0]),
            **answer_match(text, answer, aliases)}


def summary(rows, planned_samples=None):
    identifiers = sorted({r["task_id"] for r in rows})
    result = {"planned_samples": planned_samples, "completed_samples": len(identifiers), "prefix_tests": len(rows)}
    for metric in ("first_token_match", "full_match", "answer_match"):
        successes = sum(bool(r[metric]) for r in rows)
        top1 = [r for r in rows if r["rank"] == 1]
        top1_correct = sum(bool(r[metric]) for r in top1)
        covered = sum(any(r[metric] for r in rows if r["task_id"] == name) for name in identifiers)
        result[metric] = {"prefix_correct": successes, "prefix_rate": successes / len(rows) if rows else None,
                          "top1_correct": top1_correct, "top1_rate": top1_correct / len(top1) if top1 else None,
                          "covered_samples": covered, "topk_coverage": covered / len(identifiers) if identifiers else None}
    return result
