#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
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


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _scripts_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_scripts_root_to_path() -> None:
    root = _scripts_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _to_bool(x: object) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def _load_checked(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    need = {"dataset", "input_format", "output_format", "is_correct", "model", "effort"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing required columns {sorted(missing)}")
    out = df.copy()
    out["is_correct"] = out["is_correct"].map(_to_bool)
    return out


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    order = ["smiles", "iupac", "graph"]
    x = df[df["dataset"] == "translation_large"].copy()
    x = x[x["input_format"].isin(order) & x["output_format"].isin(order)].copy()
    x = x[x["input_format"] != x["output_format"]].copy()
    return x


def _summarize(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    g = (
        df.groupby(["effort", "input_format", "output_format"], as_index=False)
        .agg(n_questions=("is_correct", "size"), n_correct=("is_correct", "sum"))
        .copy()
    )
    g["model"] = model_label
    g["accuracy_pct"] = 100.0 * g["n_correct"] / g["n_questions"]
    cols = ["model", "effort", "input_format", "output_format", "n_questions", "n_correct", "accuracy_pct"]
    return g[cols]


def parse_args() -> argparse.Namespace:
    package_root = _package_root()
    parser = argparse.ArgumentParser(
        description="Generate the translation_large three-model matrix-row figure."
    )
    parser.add_argument(
        "--nano-checked",
        default=str(package_root / "model_responses" / "checked" / "gpt-5-nano-low_checked.csv"),
        help="Path to the GPT-5-nano checked CSV.",
    )
    parser.add_argument(
        "--mini-checked",
        default=str(package_root / "model_responses" / "checked" / "gpt-5-mini-low_checked.csv"),
        help="Path to the GPT-5-mini checked CSV.",
    )
    parser.add_argument(
        "--full-checked",
        default=str(package_root / "model_responses" / "checked" / "gpt-5-low_checked.csv"),
        help="Path to the GPT-5 checked CSV.",
    )
    parser.add_argument(
        "--output-figure",
        default=str(package_root / "analysis_outputs" / "plots" / "fig-translation-large-matrix-row-low.png"),
        help="Output path for the figure.",
    )
    parser.add_argument(
        "--output-table",
        default=str(package_root / "analysis_outputs" / "tables" / "matrix_accuracy_by_direction_model_translation_large.csv"),
        help="Output path for the summary table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _add_scripts_root_to_path()
    from utils.plotting import plot_translation_matrix

    files = [
        ("GPT-5-Nano", Path(args.nano_checked)),
        ("GPT-5-Mini", Path(args.mini_checked)),
        ("GPT-5", Path(args.full_checked)),
    ]
    for _, path in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing checked file: {path}")

    output_figure = Path(args.output_figure)
    output_table = Path(args.output_table)
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    output_table.parent.mkdir(parents=True, exist_ok=True)

    prepared: list[tuple[str, pd.DataFrame]] = []
    summary_rows: list[pd.DataFrame] = []
    for label, path in files:
        raw = _load_checked(path)
        sub = _prepare(raw)
        if sub.empty:
            raise ValueError(f"No translation_large rows after filtering for {path.name}")
        prepared.append((label, sub))
        summary_rows.append(_summarize(sub, label.lower().replace("-", "_")))

    pd.concat(summary_rows, ignore_index=True).to_csv(output_table, index=False)

    order = ["smiles", "iupac", "graph"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.8), dpi=240, sharey=True)
    for ax, (title, dfp) in zip(axes, prepared):
        plot_translation_matrix(
            dfp,
            title=title,
            input_order=order,
            output_order=order,
            label_map={"smiles": "SMILES", "iupac": "IUPAC", "graph": "MolJSON"},
            mask_diagonal=True,
            show_counts=True,
            text_size=22,
            count_text_size=16,
            ax=ax,
        )
        ax.title.set_fontsize(28)
        ax.xaxis.label.set_size(24)
        ax.yaxis.label.set_size(24)
        ax.xaxis.labelpad = 14
        for tick in ax.get_xticklabels():
            tick.set_fontsize(24)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(24)
        for text in ax.texts:
            txt = str(text.get_text())
            text.set_fontsize(16 if txt.startswith("n=") else 22)

    for ax in axes:
        ax.set_ylabel("")
        ax.set_xlabel("")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.supylabel("Input Representation", fontsize=28, x=0.03)
    fig.supxlabel("Output Representation", fontsize=28, y=0.065)
    fig.tight_layout(rect=[0.03, 0.06, 1.0, 1.0])
    fig.subplots_adjust(wspace=0.10)
    fig.savefig(output_figure, bbox_inches="tight", pad_inches=0.14, facecolor="white")
    plt.close(fig)

    print(f"Wrote {output_table}")
    print(f"Wrote {output_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
