"""Only receiver inputs enter non-oracle objectives; no answer labels."""
import math

import torch
import torch.nn.functional as F

from .backend import attention_summary, join_cache, kv_losses, repeat_kv
from .question_semantics import query_coefficients


class Objective:
    def __init__(self, backend, observed, public_ids, question_ids, method, config, oracle_cache=None, *, question_groups=None):
        if method == "oracle_private_prefix_distillation" and oracle_cache is None:
            raise ValueError("oracle 组缺少真实私有缓存")
        if method != "oracle_private_prefix_distillation" and oracle_cache is not None:
            raise ValueError("非 oracle 组禁止接收真实私有缓存")
        self.backend, self.observed = backend, observed
        self.public_ids, self.question_ids = public_ids, question_ids
        self.method, self.config, self.oracle_cache = method, config, oracle_cache
        self.reference = None
        self.reference_step = None
        self.question_groups = question_groups
        self.query_weights = None
        if method == "question_weighted_kv" and config.get("weighted_kv_version", 1) == 2 and config["semantic_query_mode"] == "structured":
            self.query_weights = query_coefficients(question_ids.shape[1], question_groups, config["semantic_query_weights"])

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
                    public_attention = scores.softmax(dim=-1)[..., n:n + length]
                    if self.query_weights is None:
                        importance = public_attention.mean(dim=2)
                    else:
                        coefficients = public_attention.new_tensor(self.query_weights)[None, None, :, None]
                        importance = (public_attention * coefficients).sum(dim=2)
                    importance = importance.view(importance.shape[0], kv_heads, heads // kv_heads, length).mean(dim=2)
                    total = importance.sum(dim=-1, keepdim=True)
                    normalized = importance / total.clamp_min(1e-30) * length
                    normalized = torch.where(total > 0, normalized, torch.ones_like(normalized))
                    floor = self.config["weight_floor"]
                    weights.append((floor + (1 - floor) * normalized).detach())
                self.reference = {"weights": tuple(weights)}
                if self.query_weights is not None:
                    self.reference["query_token_weights"] = tuple(self.query_weights)
            else:
                targets = tuple(attention_summary(q, layer, return_log_attention=self.config.get("attention_loss_version", 1) == 2)
                                for q, layer in zip(queries, self.observed))
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
        selected_base = components["mean_loss" if self.config["base_loss"] == "mean" else "token_loss"]
        base = selected_base if self.config["base_loss"] != "none" else selected_base.new_zeros(())
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
            outputs, lses, distributions = [], [], []
            match_attention = self.config.get("attention_loss_version", 1) == 2
            for query, layer, target in zip(self.reference["queries"], public, self.reference["targets"]):
                current = attention_summary(query, layer, return_log_attention=match_attention)
                output, lse = current[:2]
                target_o, target_z = target[:2]
                outputs.append(F.mse_loss(output, target_o))
                lses.append(F.mse_loss(lse, target_z))
                if match_attention:
                    # KL(reference || student): sum public positions BEFORE
                    # averaging batch, heads and question tokens, then layers.
                    distributions.append(F.kl_div(current[2], target[2], reduction="none", log_target=True)
                                         .sum(dim=-1).mean().clamp_min(0))
            components["attention_output_loss"], components["lse_loss"] = torch.stack(outputs).mean(), torch.stack(lses).mean()
            weighted_aux = self.config["lambda_output"] * components["attention_output_loss"] + self.config["lambda_lse"] * components["lse_loss"]
            if match_attention:
                components["attention_distribution_loss"] = torch.stack(distributions).mean()
                weighted_aux = weighted_aux + self.config["lambda_attention"] * components["attention_distribution_loss"]
        elif self.method in ("question_output_consistency", "oracle_private_prefix_distillation"):
            logits = self.backend.continuation_logits(join_cache(private, public), self.question_ids, self.reference["continuation"])
            student = (logits / self.config["temperature"]).log_softmax(-1)
            reference = self.reference["log_probs"]
            components["kl_loss"] = F.kl_div(student, reference, reduction="none", log_target=True).sum(-1).mean() * self.config["temperature"] ** 2
            weighted_aux = self.config["lambda_kl"] * components["kl_loss"]
        components.update(base_loss=base, weighted_aux_loss=weighted_aux, total_loss=base + weighted_aux)
        return components["total_loss"], components, private

    def state_dict(self):
        state = {"reference": self.reference, "reference_step": self.reference_step}
        if self.method == "question_attention_reconstruction":
            state["attention_loss_version"] = self.config.get("attention_loss_version", 1)
        if self.method == "question_weighted_kv":
            state["weighted_kv_version"] = self.config.get("weighted_kv_version", 1)
        return state

    def load_state_dict(self, state):
        if self.method == "question_attention_reconstruction" and state.get("attention_loss_version", 1) != self.config.get("attention_loss_version", 1):
            raise ValueError("attention 参考状态的损失版本不一致，不能跨版本续训")
        if self.method == "question_weighted_kv" and state.get("weighted_kv_version", 1) != self.config.get("weighted_kv_version", 1):
            raise ValueError("weight_kv 参考状态的版本不一致，不能跨版本续训")
        if self.method == "question_weighted_kv" and self.query_weights is not None and state["reference"] is not None and tuple(state["reference"].get("query_token_weights", ())) != tuple(self.query_weights):
            raise ValueError("weight_kv 参考状态的语义查询位置或贡献比例不一致")
        def transfer(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.backend.device)
            if isinstance(value, dict):
                return {k: transfer(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return tuple(transfer(v) for v in value)
            return value
        self.reference, self.reference_step = transfer(state["reference"]), state["reference_step"]
