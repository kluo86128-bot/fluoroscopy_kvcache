"""Answer labels stay in diagnostics/testing, outside the optimization objective."""
import math
import re

import numpy as np
import torch

from .backend import join_cache
from .answer_forms import (ACCEPTED_DEFINITION, CANONICAL_DEFINITION, answer_forms,
                           prefix_free_paths, format_compliant, target_mention)


class AnswerMonitor:
    def __init__(self, backend, observed, question_ids, answer, aliases=(), *,
                 probability_mode="sequence", leading_spaces=(0, 1)):
        self.backend, self.observed, self.question_ids = backend, observed, question_ids
        self.answer_ids = backend.encode(answer)
        if probability_mode not in ("sequence", "accepted_forms"):
            raise ValueError("未知概率监测模式")
        self.probability_mode = probability_mode
        texts = answer_forms(answer, aliases, leading_spaces) if probability_mode == "accepted_forms" else [answer]
        self.forms = [(text, backend.encode(text)) for text in texts]
        self.paths = prefix_free_paths(ids[0].tolist() for _, ids in self.forms)

    @torch.no_grad()
    def __call__(self, prefix_cache):
        cache = join_cache(prefix_cache, self.observed)
        canonical_path = tuple(self.answer_ids[0].tolist())
        encoded = {tuple(ids[0].tolist()): ids for _, ids in self.forms}
        encoded[canonical_path] = self.answer_ids
        probabilities = {}
        first = None
        for path, ids in encoded.items():
            # Raw model logits: no grammar mask, temperature or renormalization.
            logits = self.backend.continuation_logits(cache, self.question_ids, ids)
            values = logits.log_softmax(-1).gather(-1, ids.unsqueeze(-1))[0, :, 0].double()
            probabilities[path] = float(values.sum())
            if path == canonical_path:
                first = math.exp(float(values[0]))
        canonical_log = probabilities[canonical_path]
        if self.probability_mode == "accepted_forms":
            # Prefix-free paths are disjoint events. Accumulate in log space.
            terms = [probabilities[path] for path in self.paths]
            peak = max(terms)
            log_probability = peak + math.log(sum(math.exp(value - peak) for value in terms))
            if log_probability > 1e-5:
                raise RuntimeError("答案形式合并概率超过 1，检查 token 路径或模型 logits")
            log_probability = min(0.0, log_probability)
        else:
            log_probability = canonical_log
        details = [{"text": text, "token_ids": ids[0].tolist(),
                    "probability": math.exp(probabilities[tuple(ids[0].tolist())]),
                    "log_probability": probabilities[tuple(ids[0].tolist())],
                    "included_in_union": tuple(ids[0].tolist()) in self.paths}
                   for text, ids in self.forms]
        return {"first_token_probability": first,
                "answer_probability": math.exp(log_probability), "answer_log_probability": log_probability,
                "canonical_answer_probability": math.exp(canonical_log), "canonical_answer_log_probability": canonical_log,
                "answer_form_probabilities": details, "answer_union_path_count": len(self.paths),
                "answer_tokens": self.answer_ids.shape[1], "probability_path": "prefix_plus_observed_public",
                "probability_definition": ACCEPTED_DEFINITION if self.probability_mode == "accepted_forms" else CANONICAL_DEFINITION}


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


def answer_match(text, answer, aliases, answer_format="free_text"):
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
            "format_compliant": format_compliant(text, answer_format), **target_mention(text, answer, aliases)}


@torch.no_grad()
def test_prefix(backend, observed, public_ids, question_ids, prefix, answer, aliases, max_tokens, stop_strings=(), answer_format="free_text"):
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
            **answer_match(text, answer, aliases, answer_format)}


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
