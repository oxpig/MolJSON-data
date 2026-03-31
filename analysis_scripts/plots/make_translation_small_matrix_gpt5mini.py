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


def parse_args() -> argparse.Namespace:
    package_root = _package_root()
    parser = argparse.ArgumentParser(
        description="Generate the GPT-5-mini translation_small accuracy matrix figure."
    )
    parser.add_argument(
        "--checked-file",
        default=str(package_root / "model_responses" / "checked" / "gpt-5-mini-low_checked.csv"),
        help="Path to the checked GPT-5-mini results CSV.",
    )
    parser.add_argument(
        "--output-figure",
        default=str(
            package_root / "analysis_outputs" / "plots" / "fig-translation-small-matrix-gpt-5-mini-low-large-text.png"
        ),
        help="Output path for the matrix figure with bottom-right n-count annotations.",
    )
    parser.add_argument(
        "--output-figure-no-counts",
        default=str(
            package_root / "analysis_outputs" / "plots" / "fig-translation-small-matrix-gpt-5-mini-low-large-text-no-counts.png"
        ),
        help="Output path for the matrix figure without bottom-right n-count annotations.",
    )
    parser.add_argument(
        "--output-table",
        default=str(
            package_root / "analysis_outputs" / "tables" / "translation_small_matrix_accuracy_by_direction_gpt5mini.csv"
        ),
        help="Output path for the direction-level summary table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _add_scripts_root_to_path()
    from utils.plotting import plot_translation_matrix

    checked_file = Path(args.checked_file)
    if not checked_file.exists():
        raise FileNotFoundError(f"Missing checked file: {checked_file}")

    output_figure = Path(args.output_figure)
    output_figure_no_counts = Path(args.output_figure_no_counts)
    output_table = Path(args.output_table)
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    output_figure_no_counts.parent.mkdir(parents=True, exist_ok=True)
    output_table.parent.mkdir(parents=True, exist_ok=True)

    order = ["smiles", "iupac", "graph", "V2000_MOLBLOCK", "selfies", "inchi"]
    label_map = {
        "smiles": "SMILES",
        "iupac": "IUPAC",
        "graph": "MolJSON",
        "V2000_MOLBLOCK": "MOL\nV2000",
        "selfies": "SELFIES",
        "inchi": "InChI",
    }

    df = pd.read_csv(checked_file, low_memory=False)
    required = {"dataset", "input_format", "output_format", "is_correct", "model", "effort", "uuid"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{checked_file}: missing required columns {sorted(missing)}")

    df_plot = df[df["dataset"] == "translation_small"].copy()
    df_plot = df_plot[df_plot["input_format"].isin(order) & df_plot["output_format"].isin(order)].copy()
    if df_plot.empty:
        raise ValueError("No translation_small rows after filtering.")

    df_plot["is_correct_bool"] = df_plot["is_correct"].map(_to_bool)

    summary = (
        df_plot.groupby(["model", "effort", "input_format", "output_format"], as_index=False)
        .agg(n_questions=("is_correct_bool", "size"), n_correct=("is_correct_bool", "sum"))
        .copy()
    )
    summary["accuracy_pct"] = 100.0 * summary["n_correct"] / summary["n_questions"]
    summary.to_csv(output_table, index=False)

    def _render_matrix(out_path: Path, *, show_counts: bool) -> None:
        _, fig, ax = plot_translation_matrix(
            df_plot.assign(is_correct=df_plot["is_correct_bool"].values),
            title="",
            input_order=order,
            output_order=order,
            label_map=label_map,
            mask_diagonal=True,
            show_counts=show_counts,
            text_size=18,
            count_text_size=12,
        )

        ax.title.set_fontsize(24)
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        ax.xaxis.labelpad = 16
        for tick in ax.get_xticklabels():
            tick.set_fontsize(17)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(17)
        for text in ax.texts:
            txt = str(text.get_text())
            text.set_fontsize(12 if txt.startswith("n=") else 18)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2, facecolor="white")
        plt.close(fig)

    _render_matrix(output_figure, show_counts=True)
    _render_matrix(output_figure_no_counts, show_counts=False)

    print(f"Wrote {output_table}")
    print(f"Wrote {output_figure}")
    print(f"Wrote {output_figure_no_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
