"""Per-method diagnostics and three shared strategy comparison figures."""
from decimal import Decimal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

LABELS = {"baseline": "Baseline", "question_weighted_kv": "Weighted KV",
          "question_attention_reconstruction": "Attention", "question_output_consistency": "Consistency",
          "oracle_private_prefix_distillation": "Oracle",
          "question_joint_reconstruction": "Weighted KV + Attention"}
COLORS = {name: f"C{index}" for index, name in enumerate(LABELS)}


def percentage(value, _position=None):
    # Expand exponent notation to decimals; ticks always state their unit.
    return f"{format(Decimal(format(value, '.6g')), 'f')}%"


def probability_axis(axis, values, label="Accepted-answer form probability (%)"):
    axis.set_yscale("linear")
    axis.yaxis.set_major_formatter(FuncFormatter(percentage))
    axis.set_ylabel(label)
    if values:
        low, high = min(values), max(values)
        margin = max((high - low) * 0.1, high * 0.01, 1e-12)
        axis.set_ylim(max(0, low - margin), min(100, high + margin))


def render(root, series):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not series:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    loss_fields = ("base_loss", "weighted_kv_loss", "attention_output_loss", "lse_loss",
                   "attention_distribution_loss", "kl_loss", "total_loss")
    probability_values = []
    for index, (name, rows) in enumerate(series.items()):
        color = COLORS.get(name, f"C{index % 10}")
        label = LABELS.get(name, name)
        diagnostics = [r for r in rows if r.get("answer_probability") is not None]
        probability_values.extend(100 * r["answer_probability"] for r in diagnostics)
        accepted = any(r.get("probability_definition", "").startswith("accepted_answer") for r in diagnostics)
        axes[0].plot([r["step"] for r in diagnostics], [100 * r["answer_probability"] for r in diagnostics], color=color, marker=".", label=f"{label}: {'accepted forms' if accepted else 'canonical sequence'}")
        if accepted:
            canonical = [r for r in diagnostics if "canonical_answer_probability" in r]
            axes[0].plot([r["step"] for r in canonical], [100 * r["canonical_answer_probability"] for r in canonical],
                         color=color, linestyle=":", label=f"{label}: canonical sequence")
        if any(r.get("answer_tokens", 1) > 1 for r in diagnostics):
            probability_values.extend(100 * r["first_token_probability"] for r in diagnostics)
            axes[0].plot([r["step"] for r in diagnostics], [100 * r["first_token_probability"] for r in diagnostics], color=color, linestyle="--", marker=".", label=f"{label}: first token")
        for field in loss_fields:
            if field not in rows[0]:
                continue
            # Individual figures show every raw auxiliary loss. The comparison
            # figure uses base/total only, avoiding a legend that covers the data.
            if len(series) > 1 and field not in ("base_loss", "total_loss"):
                continue
            if field == "total_loss" and name == "baseline":
                continue
            style = "--" if field == "base_loss" else "-" if field == "total_loss" else ":"
            axes[1].plot([r["step"] for r in rows], [r[field] for r in rows], label=f"{label}: {field}",
                         color=color if len(series) > 1 else None, linestyle=style, linewidth=1)
    accepted = any(r.get("probability_definition", "").startswith("accepted_answer") for rows in series.values() for r in rows)
    probability_axis(axes[0], probability_values, "Accepted-answer form probability (%)" if accepted else "Standard-answer sequence probability (%)")
    axes[0].set_title("Training question | inferred prefix KV + observed public KV\n"
                      "Fixed answer boundary is input; sequence excludes stop tokens")
    axes[1].set_ylabel("Post-update loss")
    axes[1].set_xlabel("Gradient updates")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(root / "loss_probability.png", dpi=150)
    plt.close(fig)
    for field, filename, label in (("support_score", "support_score.png", "Trajectory support score"),
                                   ("ema_score", "support_ema.png", "Support score EMA")):
        fig, axis = plt.subplots(figsize=(10, 4))
        for name, rows in series.items():
            valid = [r for r in rows if r.get(field) is not None]
            axis.plot([r["step"] for r in valid], [r[field] for r in valid], marker=".", color=COLORS.get(name), label=LABELS.get(name, name))
        axis.set(xlabel="Gradient updates", ylabel=label,
                 title="Single trajectory; score is not a correctness probability")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        if not any(r.get(field) is not None for rows in series.values() for r in rows):
            axis.text(0.5, 0.5, "Insufficient window; no fabricated score", transform=axis.transAxes, ha="center")
        fig.tight_layout()
        fig.savefig(root / filename, dpi=150)
        plt.close(fig)


def render_comparison(root, series, *, snapshot_only=False):
    """One line per participating strategy; never mix first-token and sequence."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not series:
        return
    scope = "Saved snapshots only" if snapshot_only else "Training trajectory"
    accepted = any(r.get("probability_definition", "").startswith("accepted_answer") for rows in series.values() for r in rows)
    fields = (("answer_probability", "answer_probability_comparison.png", "Accepted-answer form probability (%)" if accepted else "Standard-answer sequence probability (%)"),
              ("support_score", "support_score_comparison.png", "Trajectory support score"),
              ("ema_score", "support_ema_comparison.png", "Support score EMA"))
    for field, filename, ylabel in fields:
        fig, axis = plt.subplots(figsize=(11, 5))
        values = []
        for index, (name, rows) in enumerate(series.items()):
            valid = [row for row in rows if row.get(field) is not None]
            y = [row[field] * (100 if field == "answer_probability" else 1) for row in valid]
            values.extend(y)
            axis.plot([row["step"] for row in valid], y, marker=".", linewidth=1.3,
                      linestyle="None" if snapshot_only else "-",
                      color=COLORS.get(name, f"C{index % 10}"), label=LABELS.get(name, name))
        axis.set(xlabel="Gradient updates", ylabel=ylabel,
                 title=f"{scope} | training question | all participating strategies")
        if field == "answer_probability":
            probability_axis(axis, values, ylabel)
        if not values:
            axis.text(0.5, 0.5, "Insufficient window; no fabricated score", transform=axis.transAxes, ha="center")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9, loc="best")
        fig.tight_layout()
        fig.savefig(root / filename, dpi=150)
        plt.close(fig)
