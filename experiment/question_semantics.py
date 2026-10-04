"""Locate receiver-provided semantic fields; never inspect answer labels."""
import re


SEMANTIC_GROUPS = ("object_state", "target_field", "required_value", "question")
SEMANTIC_WEIGHTS = dict(zip(SEMANTIC_GROUPS, (0.25, 0.30, 0.20, 0.25)))
RELATION_GROUPS = ("target_relation", "required_value", "question")
RELATION_WEIGHTS = dict(zip(RELATION_GROUPS, (0.55, 0.20, 0.25)))
FIELD_GROUPS = {"Target object": "object_state", "Target state": "object_state",
                "Target field": "target_field", "Required value": "required_value",
                "Question": "question"}
RELATION_FIELDS = {"Target relation": "target_relation", "Required value": "required_value",
                   "Question": "question"}


def semantic_spans(text):
    """Return spans of values only, excluding labels, format and answer boundary."""
    relation = bool(re.search(r"^Target relation:", text, re.M))
    if relation and any(re.search(r"^" + re.escape(label) + r":", text, re.M)
                        for label in ("Target object", "Target state", "Target field")):
        raise ValueError("Target relation 不能与旧版独立 Target 字段混用")
    fields = RELATION_FIELDS if relation else FIELD_GROUPS
    spans = {group: [] for group in (RELATION_GROUPS if relation else SEMANTIC_GROUPS)}
    for label, group in fields.items():
        matches = list(re.finditer(r"^" + re.escape(label) + r":[ \t]*([^\r\n]+)", text, re.M))
        if len(matches) != 1:
            raise ValueError(f"结构化 question 必须恰好包含一行 '{label}: ...'；请运行 prepare_questions.py，或显式使用 uniform 消融配置")
        match = matches[0]
        start, end = match.span(1)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if not any(c.isalnum() for c in text[start:end]):
            raise ValueError(f"question 的 {label} 语义内容为空")
        spans[group].append((start, end))
    return spans


def token_groups(text, offsets):
    """Map full-prompt offsets to disjoint legacy or unified-relation groups."""
    spans = semantic_spans(text)
    groups = {group: [] for group in spans}
    for index, (start, end) in enumerate(offsets):
        membership = []
        for group, ranges in spans.items():
            if any(any(c.isalnum() for c in text[max(start, a):min(end, b)])
                   for a, b in ranges if max(start, a) < min(end, b)):
                membership.append(group)
        if len(membership) > 1:
            raise ValueError("token 跨越多个语义组，无法安全分配查询贡献")
        if membership:
            groups[membership[0]].append(index)
    if any(not positions for positions in groups.values()):
        raise ValueError("结构化 question 的语义组没有可用 token")
    return groups


def effective_shares(groups, shares):
    """Merge the legacy target budgets only for unified-relation questions."""
    if not isinstance(groups, dict) or set(groups) not in (set(SEMANTIC_GROUPS), set(RELATION_GROUPS)):
        raise ValueError("weight_kv 需要四组结构化或三组目标关系查询位置")
    if set(groups) == set(RELATION_GROUPS) and set(shares) == set(SEMANTIC_GROUPS):
        return {"target_relation": shares["object_state"] + shares["target_field"],
                "required_value": shares["required_value"], "question": shares["question"]}
    if set(groups) != set(shares):
        raise ValueError("语义查询组与贡献比例不匹配")
    return dict(shares)


def query_coefficients(length, groups, shares):
    """Each group's total share is independent of its token count."""
    shares = effective_shares(groups, shares)
    coefficients, used = [0.0] * length, set()
    for group in groups:
        indices = groups[group]
        if not isinstance(indices, (list, tuple)) or not indices:
            raise ValueError(f"语义组 {group} 缺少查询位置")
        for index in indices:
            if type(index) is not int or not 0 <= index < length or index in used:
                raise ValueError(f"语义组 {group} 的查询位置越界或重复")
            used.add(index)
            coefficients[index] = shares[group] / len(indices)
    return coefficients
