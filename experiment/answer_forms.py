"""Evaluation-only answer forms and format rules. Never imported by objectives."""
import re

CANONICAL_DEFINITION = "canonical_answer_sequence_product_excluding_boundary_and_stop"
ACCEPTED_DEFINITION = "accepted_answer_token_prefix_union_v2_excluding_boundary_and_stop"
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")
FORMATS = ("free_text", "digits", "english_city", "month_ordinal")


def answer_forms(answer, aliases=(), leading_spaces=(0, 1)):
    # Fixed, finite forms; no generated history or post-hoc successful output is used.
    values = list(dict.fromkeys(value.strip() for value in (answer, *aliases)))
    if any(not value for value in values):
        raise ValueError("答案与别名不能为空")
    return list(dict.fromkeys(" " * count + value for value in values for count in leading_spaces))


def prefix_free_paths(paths):
    """Union of finite token-prefix events: a terminal ancestor covers descendants."""
    result = []
    for path in sorted(set(tuple(p) for p in paths), key=lambda p: (len(p), p)):
        if not path:
            raise ValueError("答案 token 序列不能为空")
        if not any(path[:len(parent)] == parent for parent in result):
            result.append(path)
    return result


def format_compliant(text, answer_format):
    # Ignore at most one extra leading space, just as the monitored variants do.
    value = text[1:] if text.startswith(" ") else text
    if not value or value != value.strip() or "\n" in value or "\r" in value:
        return False
    if re.match(r"^(?:answer|agent|答案)\s*[:：]|^[0-9]+[.)]\s", value, re.I):
        return False
    if answer_format == "digits":
        return re.fullmatch(r"[0-9]+", value) is not None
    if answer_format == "english_city":
        return re.fullmatch(r"[A-Za-z]+(?:[ '-][A-Za-z]+)*", value) is not None
    if answer_format == "month_ordinal":
        match = re.fullmatch(r"(" + "|".join(MONTHS) + r") ([1-9]|[12][0-9]|3[01])(st|nd|rd|th)", value)
        if not match:
            return False
        day = int(match[2])
        suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return match[3] == suffix
    if answer_format == "free_text":
        return bool(value)
    raise ValueError(f"未知答案格式: {answer_format}")


def target_mention(text, answer, aliases):
    # Lexical presence is not semantic recovery: negations and explanations need review.
    matches = [value for value in dict.fromkeys((answer, *aliases))
               if re.search(r"(?<!\w)" + re.escape(value.strip()) + r"(?!\w)", text, re.I)]
    return {"target_mentioned": bool(matches), "matched_target_forms": matches,
            "recovery_review_status": "pending_semantic_review"}
