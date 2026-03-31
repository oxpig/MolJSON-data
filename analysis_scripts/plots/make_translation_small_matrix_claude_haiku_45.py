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


def parse_args() -> argparse.Namespace:
    package_root = _package_root()
    parser = argparse.ArgumentParser(description="Plot translation accuracy matrix for Claude Haiku 4.5.")
    parser.add_argument(
        "--checked-csv",
        default=str(package_root / "model_responses" / "checked" / "claude-haiku-4-5_checked.csv"),
        help="Checked CSV produced by the packaged answer checker.",
    )
    parser.add_argument(
        "--out-fig",
        default=str(package_root / "analysis_outputs" / "plots" / "fig-translation-small-matrix-claude-haiku-4-5.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--title",
        default="Haiku 4.5 (Thinking 4096)",
        help="Figure title.",
    )
    parser.add_argument(
        "--hide-counts",
        action="store_true",
        help="Hide per-cell task counts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _add_scripts_root_to_path()
    from utils.plotting import plot_translation_matrix

    in_csv = Path(args.checked_csv)
    out_fig = Path(args.out_fig)
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing checked CSV: {in_csv}")

    df = pd.read_csv(in_csv, low_memory=False)
    df = df[(df["category"] == "translation") & (df["dataset"] == "translation_small")].copy()
    df["input_format"] = df["input_format"].astype(str).str.lower()
    df["output_format"] = df["output_format"].astype(str).str.lower()
    df = df[df["input_format"].isin(["smiles", "iupac", "graph"]) & df["output_format"].isin(["smiles", "iupac", "graph"])].copy()

    order = ["smiles", "iupac", "graph"]
    label_map = {"smiles": "SMILES", "iupac": "IUPAC", "graph": "MolJSON"}

    _, fig, _ = plot_translation_matrix(
        df,
        title=args.title,
        input_order=order,
        output_order=order,
        label_map=label_map,
        mask_diagonal=True,
        show_counts=not args.hide_counts,
        text_size=16,
        count_text_size=11,
    )

    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
