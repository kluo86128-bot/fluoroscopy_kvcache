"""Differentiable cache operations and Qwen/Llama rotary query probes."""
from contextlib import contextmanager
import math

import torch


def pairs(cache):
    if hasattr(cache, "layers"):
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    if hasattr(cache, "key_cache"):
        return tuple(zip(cache.key_cache, cache.value_cache))
    return tuple(cache)


def slice_cache(cache, start, stop=None, detach=False):
    return tuple(tuple((x[:, :, start:stop].detach() if detach else x[:, :, start:stop]).clone()
                       for x in layer) for layer in cache)


def join_cache(prefix, public):
    if len(prefix) != len(public):
        raise ValueError("缓存层数不一致")
    return tuple((torch.cat((p[0], c[0]), dim=2), torch.cat((p[1], c[1]), dim=2))
                 for p, c in zip(prefix, public))


def repeat_kv(value, heads):
    if heads % value.shape[1]:
        raise ValueError("查询头数不是 KV 头数的整数倍")
    return value.repeat_interleave(heads // value.shape[1], dim=1)


def kv_losses(predicted, observed):
    if len(predicted) != len(observed) or not predicted:
        raise ValueError("公共 KV 层数不一致或为空")
    mean, token = [], []
    for student, teacher in zip(predicted, observed):
        for x, y in zip(student, teacher):
            if x.shape != y.shape:
                raise ValueError(f"公共 KV 形状不一致: {x.shape} / {y.shape}")
            x, y = x.float(), y.detach().float()
            mean.append((x.mean(dim=2) - y.mean(dim=2)).square().mean())
            token.append((x - y).square().mean())
    return {"mean_loss": torch.stack(mean).mean(), "token_loss": torch.stack(token).mean()}


def attention_summary(query, cache, *, return_log_attention=False):
    """Read public KV; optionally retain each head/query's position distribution."""
    key, value = (repeat_kv(x.float(), query.shape[1]) for x in cache)
    scores = query.float() @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
    summary = (scores.softmax(dim=-1) @ value, scores.logsumexp(dim=-1))
    if return_log_attention:
        return (*summary, scores.log_softmax(dim=-1))
    return summary


class Backend:
    def __init__(self, model, tokenizer=None, cache_factory=None):
        self.model, self.tokenizer, self.cache_factory = model, tokenizer, cache_factory
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.embedding = model.get_input_embeddings()
        self.device, self.dtype = self.embedding.weight.device, self.embedding.weight.dtype

    def encode(self, text):
        values = self.tokenizer.encode(text, add_special_tokens=False)
        if not values:
            raise ValueError("文本编码后为空")
        return torch.tensor([values], device=self.device, dtype=torch.long)

    def encode_question(self, text, config):
        return self.encode(self.question_prompt(text, config))

    def question_prompt(self, text, config):
        if config["question_format"] == "chat":
            boundary = config["answer_boundary"]
            if text.endswith(boundary):
                text = text[:-len(boundary)].rstrip()
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                enable_thinking=config["enable_thinking"])
            text += boundary
        return text

    def semantic_query_groups(self, text, config, question_ids):
        from .question_semantics import token_groups
        prompt = self.question_prompt(text, config)
        if not callable(self.tokenizer):
            raise ValueError("结构化语义查询需要支持 offset_mapping 的 tokenizer")
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        if encoded["input_ids"] != question_ids[0].tolist():
            raise ValueError("语义位置映射的 token 与实际 question 输入不一致")
        return token_groups(prompt, encoded["offset_mapping"])

    def cache(self, layers):
        if self.cache_factory is None:
            from transformers import DynamicCache
            factory = DynamicCache
        else:
            factory = self.cache_factory
        result = factory()
        for index, (key, value) in enumerate(layers):
            # clone preserves the student's graph and prevents cache mutation.
            result.update(key.clone(), value.clone(), index)
        return result

    def forward(self, *, ids=None, embeddings=None, cache=None):
        if (ids is None) == (embeddings is None):
            raise ValueError("ids/embeddings 必须且只能指定一个")
        count = ids.shape[1] if ids is not None else embeddings.shape[1]
        batch = ids.shape[0] if ids is not None else embeddings.shape[0]
        start = 0 if cache is None else cache[0][0].shape[2]
        positions = torch.arange(start, start + count, device=self.device)
        arguments = {"input_ids": ids} if ids is not None else {"inputs_embeds": embeddings}
        result = self.model(**arguments, past_key_values=None if cache is None else self.cache(cache),
                            attention_mask=torch.ones((batch, start + count), device=self.device, dtype=torch.long),
                            position_ids=positions[None].expand(batch, -1), cache_position=positions,
                            use_cache=True, return_dict=True)
        return result.logits, pairs(result.past_key_values)

    def student(self, prefix, public_ids):
        embeddings = torch.cat((prefix.to(self.dtype), self.embedding(public_ids)), dim=1)
        _, cache = self.forward(embeddings=embeddings)
        n = prefix.shape[1]
        return slice_cache(cache, 0, n), slice_cache(cache, n)

    @torch.no_grad()
    def setup_teacher(self, private_ids, public_ids, include_private=False):
        _, cache = self.forward(ids=torch.cat((private_ids, public_ids), dim=1))
        n = private_ids.shape[1]
        public = slice_cache(cache, n, detach=True)
        private = slice_cache(cache, 0, n, detach=True) if include_private else None
        return public, n, private

    @contextmanager
    def queries(self):
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("问题探针需要 Qwen2/Qwen3/Llama 风格的 model.layers.self_attn")
        captured, handles = {}, []

        def hook(index):
            def capture(module, arguments, keywords):
                hidden = keywords.get("hidden_states", arguments[0] if arguments else None)
                position = keywords.get("position_embeddings", arguments[1] if len(arguments) > 1 else None)
                if hidden is None or position is None or not hasattr(module, "q_proj"):
                    raise ValueError("无法取得真实查询与 RoPE，拒绝使用近似查询替代")
                dimension = module.head_dim
                query = module.q_proj(hidden).view(*hidden.shape[:-1], -1, dimension)
                if hasattr(module, "q_norm"):
                    query = module.q_norm(query)
                query = query.transpose(1, 2)
                cosine, sine = (x.unsqueeze(1) for x in position)
                half = dimension // 2
                rotated = torch.cat((-query[..., half:], query[..., :half]), dim=-1)
                captured[index] = (query * cosine + rotated * sine).detach()
            return capture

        try:
            for index, layer in enumerate(layers):
                handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def probe(self, reference_cache, question_ids):
        with self.queries() as captured:
            _, complete = self.forward(ids=question_ids, cache=reference_cache)
        if len(captured) != len(complete):
            raise ValueError("查询探针层数不完整")
        return tuple(captured[i] for i in range(len(complete))), complete

    @torch.no_grad()
    def rollout(self, cache, question_ids, count, stop_strings=()):
        eos = self.tokenizer.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        logits, working_cache = self.forward(ids=question_ids, cache=cache)
        token = int(logits[0, -1].argmax())
        generated = []
        for index in range(count):
            generated.append(token)
            if token in eos or index + 1 == count:
                break
            if stop_strings:
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                if any(stop in text for stop in stop_strings):
                    break
            continuation = torch.tensor([[token]], device=self.device, dtype=torch.long)
            logits, working_cache = self.forward(ids=continuation, cache=working_cache)
            token = int(logits[0, -1].argmax())
        return torch.tensor([generated], device=self.device, dtype=torch.long)

    def continuation_logits(self, cache, question_ids, continuation):
        inputs = torch.cat((question_ids, continuation[:, :-1]), dim=1)
        logits, _ = self.forward(ids=inputs, cache=cache)
        start = question_ids.shape[1] - 1
        return logits[:, start:start + continuation.shape[1]].float()


def load_backend(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无可用 GPU")
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=config["local_files_only"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], torch_dtype=getattr(torch, config["dtype"]),
        attn_implementation=config["attention_backend"], local_files_only=config["local_files_only"])
    model.to(config["device"])
    backend = Backend(model, tokenizer)
    print(f"实际设备={backend.device} dtype={backend.dtype}", flush=True)
    if backend.device.type == "cuda":
        print(f"逻辑GPU={backend.device.index} name={torch.cuda.get_device_name(backend.device)}", flush=True)
    return backend
