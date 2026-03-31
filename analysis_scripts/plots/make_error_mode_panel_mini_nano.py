#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import make_error_mode_panel as base


MODEL_FILES = [
    ("gpt-5-mini", "gpt-5-mini-low_checked.csv"),
    ("gpt-5-nano", "gpt-5-nano-low_checked.csv"),
]
PANEL_LABELS = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
ROW_LABELS = ["GPT-5-Mini", "GPT-5-Nano"]
ROW_LABEL_FS = base.AXIS_LABEL_FS + 5


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    package_root = _package_root()
    parser = argparse.ArgumentParser(description="Generate a combined GPT-5-mini / GPT-5-nano error-mode panel figure.")
    parser.add_argument(
        "--checked-dir",
        default=str(package_root / "model_responses" / "checked"),
    )
    parser.add_argument(
        "--out-figure",
        default=str(package_root / "analysis_outputs" / "plots" / "fig-gpt5mini-gpt5nano-error-modes-two-row.png"),
    )
    parser.add_argument(
        "--out-summary-dir",
        default=str(package_root / "analysis_outputs" / "tables" / "error_mode_panel_mini_nano"),
    )
    parser.add_argument("--fig-width", type=float, default=17.2)
    parser.add_argument("--fig-height", type=float, default=14.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checked_dir = Path(args.checked_dir)
    out_figure = Path(args.out_figure)
    out_summary_dir = Path(args.out_summary_dir)
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    out_summary_dir.mkdir(parents=True, exist_ok=True)

    panel_a_all: list[pd.DataFrame] = []
    panel_b_all: list[pd.DataFrame] = []
    panel_c_all: list[pd.DataFrame] = []
    panel_triplets: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []

    for model_key, filename in MODEL_FILES:
        path = checked_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing checked file: {path}")
        rows = base._load_checked(path)
        if rows.empty:
            raise ValueError(f"No rows left for {path.name}")
        pa = base._build_panel_a(rows)
        pb = base._build_panel_b(rows)
        pc = base._build_panel_c(rows)
        panel_a_all.append(pa)
        panel_b_all.append(pb)
        panel_c_all.append(pc)
        panel_triplets.append((pa, pb, pc))

    fig, axes = plt.subplots(2, 3, figsize=(args.fig_width, args.fig_height), dpi=260, sharey="row")
    heavy_bins = panel_triplets[0][0][["heavy_bin_idx", "heavy_bin_label"]].drop_duplicates().sort_values("heavy_bin_idx")

    for row_idx, (pa, pb, pc) in enumerate(panel_triplets):
        base._plot_line_panel(
            axes[row_idx, 0],
            pa,
            x_col="heavy_bin_idx",
            x_order=heavy_bins["heavy_bin_idx"].astype(int).tolist(),
            x_labels=heavy_bins["heavy_bin_label"].astype(str).tolist(),
            panel_label=PANEL_LABELS[row_idx][0],
            subset_note="0 rings only",
            x_label="Number of Heavy Atoms",
        )
        base._plot_line_panel(
            axes[row_idx, 1],
            pb,
            x_col="ring_count",
            x_order=[0, 1, 2, 3],
            x_labels=["0", "1", "2", "3"],
            panel_label=PANEL_LABELS[row_idx][1],
            subset_note="All molecules",
            x_label="Number of Rings",
        )
        base._plot_slope_panel(axes[row_idx, 2], pc, PANEL_LABELS[row_idx][2])

        axes[row_idx, 1].tick_params(labelleft=False)
        axes[row_idx, 2].tick_params(labelleft=False)
        axes[row_idx, 1].set_ylabel("")
        axes[row_idx, 2].set_ylabel("")

        fig.text(
            0.012,
            0.748 if row_idx == 0 else 0.325,
            ROW_LABELS[row_idx],
            rotation=90,
            va="center",
            ha="center",
            fontsize=ROW_LABEL_FS,
        )

    handles, labels = base._legend_handles()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=20,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        handlelength=3.0,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0.03, 0.09, 1.0, 0.98])
    fig.subplots_adjust(wspace=0.10, hspace=0.14)
    fig.savefig(out_figure, facecolor="white")
    plt.close(fig)

    p1 = out_summary_dir / "panel_a_zero_ring_heavy_bins5.csv"
    p2 = out_summary_dir / "panel_b_all_molecules_ringcount.csv"
    p3 = out_summary_dir / "panel_c_topology_2or3rings_nonfused_vs_fusedspiro.csv"
    pd.concat(panel_a_all, ignore_index=True).sort_values(["model", "direction", "heavy_bin_idx"]).to_csv(p1, index=False)
    pd.concat(panel_b_all, ignore_index=True).sort_values(["model", "direction", "ring_count"]).to_csv(p2, index=False)
    pd.concat(panel_c_all, ignore_index=True).sort_values(["model", "direction"]).to_csv(p3, index=False)

    print(f"Wrote {out_figure}")
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
