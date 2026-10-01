"""Only receiver inputs enter non-oracle objectives; no answer labels."""
import math

import torch
import torch.nn.functional as F

from .backend import attention_summary, join_cache, kv_losses, repeat_kv


class Objective:
    def __init__(self, backend, observed, public_ids, question_ids, method, config, oracle_cache=None):
        if method == "oracle_private_prefix_distillation" and oracle_cache is None:
            raise ValueError("oracle 组缺少真实私有缓存")
        if method != "oracle_private_prefix_distillation" and oracle_cache is not None:
            raise ValueError("非 oracle 组禁止接收真实私有缓存")
        self.backend, self.observed = backend, observed
        self.public_ids, self.question_ids = public_ids, question_ids
        self.method, self.config, self.oracle_cache = method, config, oracle_cache
        self.reference = None
        self.reference_step = None

    @torch.no_grad()
    def refresh(self, prefix, step):
        if self.method == "baseline":
            return False
        if self.reference is not None:
            if self.method == "oracle_private_prefix_distillation":
                return False
            if step - self.reference_step < self.config["reference_refresh_steps"]:
                return False
        private, _ = self.backend.student(prefix, self.public_ids)
        reference_cache = join_cache(private if self.oracle_cache is None else self.oracle_cache, self.observed)
        if self.method in ("question_weighted_kv", "question_attention_reconstruction"):
            queries, complete = self.backend.probe(reference_cache, self.question_ids)
            if self.method == "question_weighted_kv":
                weights = []
                n = private[0][0].shape[2]
                length = self.observed[0][0].shape[2]
                for query, (key, _) in zip(queries, complete):
                    heads, kv_heads = query.shape[1], key.shape[1]
                    scores = query.float() @ repeat_kv(key.float(), heads).transpose(-1, -2) / math.sqrt(query.shape[-1])
                    positions = torch.arange(key.shape[2], device=query.device)
                    allowed = positions[None, :] <= (n + length + torch.arange(query.shape[2], device=query.device))[:, None]
                    scores = scores.masked_fill(~allowed[None, None], -torch.inf)
                    importance = scores.softmax(dim=-1)[..., n:n + length].mean(dim=2)
                    importance = importance.view(importance.shape[0], kv_heads, heads // kv_heads, length).mean(dim=2)
                    total = importance.sum(dim=-1, keepdim=True)
                    normalized = importance / total.clamp_min(1e-30) * length
                    normalized = torch.where(total > 0, normalized, torch.ones_like(normalized))
                    floor = self.config["weight_floor"]
                    weights.append((floor + (1 - floor) * normalized).detach())
                self.reference = {"weights": tuple(weights)}
            else:
                targets = tuple(attention_summary(q, layer) for q, layer in zip(queries, self.observed))
                self.reference = {"queries": queries, "targets": targets}
        else:
            continuation = self.backend.rollout(reference_cache, self.question_ids, self.config["rollout_tokens"])
            logits = self.backend.continuation_logits(reference_cache, self.question_ids, continuation)
            self.reference = {"continuation": continuation.detach(), "log_probs": (logits / self.config["temperature"]).log_softmax(-1).detach()}
        self.reference_step = step
        return True

    def compute(self, prefix):
        private, public = self.backend.student(prefix, self.public_ids)
        components = kv_losses(public, self.observed)
        base = components["mean_loss" if self.config["base_loss"] == "mean" else "token_loss"]
        weighted_aux = base.new_zeros(())
        if self.method == "question_weighted_kv":
            errors = []
            for predicted, observed, weight in zip(public, self.observed, self.reference["weights"]):
                for x, y in zip(predicted, observed):
                    per_token = (x.float() - y.detach().float()).square().mean(dim=-1)
                    errors.append(((per_token * weight).sum(-1) / weight.sum(-1)).mean())
            components["weighted_kv_loss"] = torch.stack(errors).mean()
            weighted_aux = self.config["lambda_weighted"] * components["weighted_kv_loss"]
        elif self.method == "question_attention_reconstruction":
            outputs, lses = [], []
            for query, layer, (target_o, target_z) in zip(self.reference["queries"], public, self.reference["targets"]):
                output, lse = attention_summary(query, layer)
                outputs.append(F.mse_loss(output, target_o))
                lses.append(F.mse_loss(lse, target_z))
            components["attention_output_loss"], components["lse_loss"] = torch.stack(outputs).mean(), torch.stack(lses).mean()
            weighted_aux = self.config["lambda_output"] * components["attention_output_loss"] + self.config["lambda_lse"] * components["lse_loss"]
        elif self.method in ("question_output_consistency", "oracle_private_prefix_distillation"):
            logits = self.backend.continuation_logits(join_cache(private, public), self.question_ids, self.reference["continuation"])
            student = (logits / self.config["temperature"]).log_softmax(-1)
            reference = self.reference["log_probs"]
            components["kl_loss"] = F.kl_div(student, reference, reduction="none", log_target=True).sum(-1).mean() * self.config["temperature"] ** 2
            weighted_aux = self.config["lambda_kl"] * components["kl_loss"]
        components.update(base_loss=base, weighted_aux_loss=weighted_aux, total_loss=base + weighted_aux)
        return components["total_loss"], components, private

    def state_dict(self):
        return {"reference": self.reference, "reference_step": self.reference_step}

    def load_state_dict(self, state):
        def transfer(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.backend.device)
            if isinstance(value, dict):
                return {k: transfer(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return tuple(transfer(v) for v in value)
            return value
        self.reference, self.reference_step = transfer(state["reference"]), state["reference_step"]
