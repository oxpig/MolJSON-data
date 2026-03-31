#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    cache_dir = Path(tempfile.gettempdir()) / "matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


INPUT_ORDER = ["smiles", "iupac", "graph"]
INPUT_LABEL = {"smiles": "SMILES", "iupac": "IUPAC", "graph": "MolJSON"}
INPUT_COLOR = {"smiles": "#E69F00", "iupac": "#56B4E9", "graph": "#009E73"}
AXIS_LABEL_FS = 19
TICK_LABEL_FS = 19
ANNOT_FS = TICK_LABEL_FS
LEGEND_FS = 18
X_LABEL_PAD = 16


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_bool(x: object) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def _wilson_ci_pct(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = correct / total
    denom = 1.0 + (z**2 / total)
    center = (p + (z**2) / (2 * total)) / denom
    rad = z * math.sqrt((p * (1 - p) / total) + (z**2 / (4 * total * total))) / denom
    lo = max(0.0, center - rad)
    hi = min(1.0, center + rad)
    return (100.0 * lo, 100.0 * hi)


def _load_shortest_path_answers(questions_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with questions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("category", "")).strip().lower() != "shortest_path":
                continue
            rows.append(
                {
                    "uuid": str(row.get("uuid", "")).strip(),
                    "gt_shortest_path_len": pd.to_numeric(row.get("answer"), errors="coerce"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No shortest_path rows found in {questions_path}")
    out = out.dropna(subset=["gt_shortest_path_len"]).copy()
    out["gt_shortest_path_len"] = out["gt_shortest_path_len"].astype(int)
    return out


def _load_rows(checked_path: Path, questions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(checked_path, low_memory=False)
    need = {"uuid", "category", "input_format", "output_format", "is_correct", "output_tokens", "model", "effort"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{checked_path}: missing required columns {sorted(missing)}")

    sub = df[df["category"] == "shortest_path"].copy()
    sub["input_format"] = sub["input_format"].astype(str).str.lower()
    sub["output_format"] = sub["output_format"].astype(str).str.lower()
    sub = sub[sub["input_format"].isin(INPUT_ORDER) & (sub["output_format"] == "integer")].copy()
    sub["is_correct_bool"] = sub["is_correct"].map(_to_bool).astype(bool)
    sub["output_tokens_num"] = pd.to_numeric(sub["output_tokens"], errors="coerce")

    answers = _load_shortest_path_answers(questions_path)
    sub = sub.merge(answers, on="uuid", how="left")
    sub = sub.dropna(subset=["gt_shortest_path_len"]).copy()
    sub["gt_shortest_path_len"] = sub["gt_shortest_path_len"].astype(int)
    return sub


def _summarize_accuracy(rows: pd.DataFrame, model_label: str) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for fmt in INPUT_ORDER:
        s = rows[rows["input_format"] == fmt]
        if s.empty:
            continue
        grp = (
            s.groupby("gt_shortest_path_len", as_index=False)["is_correct_bool"]
            .agg(n_correct="sum", n_questions="size")
            .sort_values("gt_shortest_path_len")
        )
        for _, r in grp.iterrows():
            n_correct = int(r["n_correct"])
            n_total = int(r["n_questions"])
            lo, hi = _wilson_ci_pct(n_correct, n_total)
            out_rows.append(
                {
                    "model": model_label,
                    "effort": str(rows["effort"].iloc[0]),
                    "input_format": fmt,
                    "gt_shortest_path_len": int(r["gt_shortest_path_len"]),
                    "n_questions": n_total,
                    "n_correct": n_correct,
                    "accuracy_pct": (100.0 * n_correct / n_total) if n_total else float("nan"),
                    "ci_low_pct": lo,
                    "ci_high_pct": hi,
                }
            )
    return pd.DataFrame(out_rows)


def _summarize_tokens(rows: pd.DataFrame, model_label: str) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    r = rows.dropna(subset=["output_tokens_num"]).copy()
    for fmt in INPUT_ORDER:
        s = r[r["input_format"] == fmt]
        if s.empty:
            continue
        grp = (
            s.groupby("gt_shortest_path_len", as_index=False)["output_tokens_num"]
            .agg(mean_tokens="mean", std_tokens="std", n_questions="size")
            .sort_values("gt_shortest_path_len")
        )
        for _, row in grp.iterrows():
            n = int(row["n_questions"])
            mean = float(row["mean_tokens"])
            std = float(row["std_tokens"]) if pd.notna(row["std_tokens"]) else 0.0
            se = (std / math.sqrt(n)) if n > 0 else float("nan")
            rad = 1.96 * se if pd.notna(se) else float("nan")
            lo = max(0.0, mean - rad) if pd.notna(rad) else float("nan")
            hi = mean + rad if pd.notna(rad) else float("nan")
            out_rows.append(
                {
                    "model": model_label,
                    "effort": str(rows["effort"].iloc[0]),
                    "input_format": fmt,
                    "gt_shortest_path_len": int(row["gt_shortest_path_len"]),
                    "n_questions": n,
                    "mean_tokens": mean,
                    "ci_low_tokens": lo,
                    "ci_high_tokens": hi,
                }
            )
    return pd.DataFrame(out_rows)


def _legend_handles() -> tuple[list[Line2D], list[str]]:
    handles: list[Line2D] = []
    labels: list[str] = []
    for fmt in INPUT_ORDER:
        handles.append(Line2D([0], [0], color=INPUT_COLOR[fmt], linewidth=2.2))
        labels.append(INPUT_LABEL[fmt])
    return handles, labels


def parse_args() -> argparse.Namespace:
    package_root = _package_root()
    parser = argparse.ArgumentParser(
        description="Generate the GPT-5 shortest-path two-panel figure (accuracy and tokens)."
    )
    parser.add_argument(
        "--checked-file",
        default=str(package_root / "model_responses" / "checked" / "gpt-5-low_checked.csv"),
        help="Path to the GPT-5 checked CSV.",
    )
    parser.add_argument(
        "--questions-file",
        default=str(package_root / "questions" / "shortest_path_questions.jsonl"),
        help="Path to the shortest_path question JSONL.",
    )
    parser.add_argument(
        "--output-figure",
        default=str(
            package_root / "analysis_outputs" / "plots" / "fig-gpt5-shortest-path-accuracy-and-tokens-two-panel-horizontal.png"
        ),
        help="Output path for the figure.",
    )
    parser.add_argument(
        "--output-accuracy-table",
        default=str(package_root / "analysis_outputs" / "tables" / "accuracy_by_path_length_input_gpt5.csv"),
        help="Output path for the accuracy summary table.",
    )
    parser.add_argument(
        "--output-tokens-table",
        default=str(package_root / "analysis_outputs" / "tables" / "output_tokens_by_path_length_input_gpt5.csv"),
        help="Output path for the token summary table.",
    )
    parser.add_argument("--fig-width", type=float, default=10.8)
    parser.add_argument("--fig-height", type=float, default=5.8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    checked_path = Path(args.checked_file)
    questions_path = Path(args.questions_file)
    output_figure = Path(args.output_figure)
    output_accuracy_table = Path(args.output_accuracy_table)
    output_tokens_table = Path(args.output_tokens_table)

    if not checked_path.exists():
        raise FileNotFoundError(f"Missing checked file: {checked_path}")
    if not questions_path.exists():
        raise FileNotFoundError(f"Missing shortest_path questions file: {questions_path}")

    output_figure.parent.mkdir(parents=True, exist_ok=True)
    output_accuracy_table.parent.mkdir(parents=True, exist_ok=True)
    output_tokens_table.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(checked_path, questions_path)
    if rows.empty:
        raise ValueError(f"No shortest_path rows for {checked_path.name}")

    acc = _summarize_accuracy(rows, "gpt-5")
    tok = _summarize_tokens(rows, "gpt-5")
    acc.sort_values(["input_format", "gt_shortest_path_len"]).to_csv(output_accuracy_table, index=False)
    tok.sort_values(["input_format", "gt_shortest_path_len"]).to_csv(output_tokens_table, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(args.fig_width, args.fig_height), dpi=260, sharex=False)

    ax = axes[0]
    for fmt in INPUT_ORDER:
        s = acc[acc["input_format"] == fmt].sort_values("gt_shortest_path_len")
        if s.empty:
            continue
        x = s["gt_shortest_path_len"].to_numpy(dtype=int)
        y = s["accuracy_pct"].to_numpy(dtype=float)
        lo = s["ci_low_pct"].to_numpy(dtype=float)
        hi = s["ci_high_pct"].to_numpy(dtype=float)
        ax.plot(x, y, color=INPUT_COLOR[fmt], linewidth=2.2, zorder=3)
        ax.fill_between(x, lo, hi, color=INPUT_COLOR[fmt], alpha=0.16, zorder=2)
    xticks = sorted(acc["gt_shortest_path_len"].unique().tolist())
    if xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) if i % 2 == 0 else "" for i, x in enumerate(xticks)])
    ax.set_box_aspect(1.0)
    ax.set_title("(a)", fontsize=ANNOT_FS, pad=10, loc="left")
    ax.set_xlabel("Shortest path length", fontsize=AXIS_LABEL_FS, labelpad=X_LABEL_PAD)
    ax.set_ylabel("Accuracy (%)", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FS)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)

    ax2 = axes[1]
    for fmt in INPUT_ORDER:
        s = tok[tok["input_format"] == fmt].sort_values("gt_shortest_path_len")
        if s.empty:
            continue
        x = s["gt_shortest_path_len"].to_numpy(dtype=int)
        y = s["mean_tokens"].to_numpy(dtype=float)
        lo = s["ci_low_tokens"].to_numpy(dtype=float)
        hi = s["ci_high_tokens"].to_numpy(dtype=float)
        ax2.plot(x, y, color=INPUT_COLOR[fmt], linewidth=2.2, zorder=3)
        ax2.fill_between(x, lo, hi, color=INPUT_COLOR[fmt], alpha=0.16, zorder=2)
    xticks2 = sorted(tok["gt_shortest_path_len"].unique().tolist())
    if xticks2:
        ax2.set_xticks(xticks2)
        ax2.set_xticklabels([str(x) if i % 2 == 0 else "" for i, x in enumerate(xticks2)])
    ax2.set_box_aspect(1.0)
    ax2.set_title("(b)", fontsize=ANNOT_FS, pad=10, loc="left")
    ax2.set_xlabel("Shortest path length", fontsize=AXIS_LABEL_FS, labelpad=X_LABEL_PAD)
    ax2.set_ylabel("Output Tokens", fontsize=AXIS_LABEL_FS)
    ax2.tick_params(axis="both", labelsize=TICK_LABEL_FS)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.7)

    max_tok = float(tok["ci_high_tokens"].max()) if (not tok.empty and tok["ci_high_tokens"].notna().any()) else 0.0
    if max_tok > 0:
        y_max = float(math.ceil(max_tok / 100.0) * 100.0)
        ax2.set_ylim(0, y_max)

    handles, labels = _legend_handles()
    fig.legend(handles, labels, frameon=False, fontsize=LEGEND_FS, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0.01, 0.15, 1.0, 0.95])
    fig.subplots_adjust(wspace=0.36)
    fig.savefig(output_figure, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    print(f"Wrote {output_accuracy_table}")
    print(f"Wrote {output_tokens_table}")
    print(f"Wrote {output_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
