#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    cache_dir = Path(tempfile.gettempdir()) / "matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

RX_A_NUM = re.compile(r"^[aA][0-9]+$")
RX_ELEM_NUM = re.compile(r"^(?:" + "|".join(ELEMENTS) + r")[0-9]+$", re.IGNORECASE)
RX_LETTER_NUM = re.compile(r"^[A-Za-z][0-9]+$")
RX_DIGITS_ONLY = re.compile(r"^[0-9]+$")

INPUT_FORMATS = ["iupac", "smiles"]
INPUT_LABEL = {"smiles": "SMILES", "iupac": "IUPAC"}
# Keep the same format colors used across the v1.0.2 figures.
INPUT_COLOR = {"smiles": "#E69F00", "iupac": "#56B4E9"}

TITLE_FS = 22
AXIS_LABEL_FS = 20
TICK_FS = 16
LEGEND_FS = 18
PCT_LABEL_FS = 11


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    package_root = _package_root()
    p = argparse.ArgumentParser(
        description=(
            "Create the GPT-5 atom-ID three-panel figure: "
            "(a) atom ID occurrences, (b) unique atom IDs, (c) regex buckets."
        )
    )
    p.add_argument(
        "--results",
        default=str(package_root / "model_responses" / "checked" / "gpt-5-low_checked.csv"),
    )
    p.add_argument("--dataset", default="translation_large")
    p.add_argument("--model", default="gpt-5")
    p.add_argument("--output-format", default="graph")
    p.add_argument(
        "--out-summary",
        default=str(
            package_root
            / "analysis_outputs"
            / "tables"
            / "gpt5_atomid_length_and_waterfall_three_panel_summary.csv"
        ),
    )
    p.add_argument(
        "--out-figure",
        default=str(
            package_root
            / "analysis_outputs"
            / "plots"
            / "fig-gpt5-atomid-length-and-waterfall-three-panel.png"
        ),
    )
    p.add_argument("--dpi", type=int, default=260)
    return p.parse_args()

def _parse_atom_ids(raw: str | None) -> list[str] | None:
    text = (raw or "").strip()
    if not text or text.lower() == "nan":
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    atoms = obj.get("atoms")
    if not isinstance(atoms, list):
        return None
    ids: list[str] = []
    for atom in atoms:
        if isinstance(atom, dict) and atom.get("id") is not None:
            ids.append(str(atom["id"]))
    return ids if ids else None


def _classify_regex(atom_id: str) -> str:
    # Ordering matches the final plotted analysis: a#, {el}#, @#, #, other
    if RX_A_NUM.fullmatch(atom_id):
        return "a#"
    if RX_ELEM_NUM.fullmatch(atom_id):
        return "{el}#"
    if RX_LETTER_NUM.fullmatch(atom_id):
        return "@#"
    if RX_DIGITS_ONLY.fullmatch(atom_id):
        return "#"
    return "other"


def _resolved_rows(
    results_path: Path,
    dataset: str,
    model: str,
    output_format: str,
    input_format: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with results_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset") != dataset:
                continue
            if (row.get("category") or "").lower() != "translation":
                continue
            if (row.get("model") or "").lower() != model.lower():
                continue
            if (row.get("output_format") or "").lower() != output_format.lower():
                continue
            if (row.get("input_format") or "").lower() != input_format:
                continue
            rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    out_fig = Path(args.out_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    unique_len_counter: dict[str, Counter[int]] = {fmt: Counter() for fmt in INPUT_FORMATS}
    occ_len_counter: dict[str, Counter[int]] = {fmt: Counter() for fmt in INPUT_FORMATS}
    regex_counter: dict[str, Counter[str]] = {fmt: Counter() for fmt in INPUT_FORMATS}
    n_rows: dict[str, int] = {fmt: 0 for fmt in INPUT_FORMATS}
    n_unique_ids: dict[str, int] = {fmt: 0 for fmt in INPUT_FORMATS}
    n_occurrences: dict[str, int] = {fmt: 0 for fmt in INPUT_FORMATS}

    for fmt in INPUT_FORMATS:
        rows = _resolved_rows(
            results_path=Path(args.results),
            dataset=args.dataset,
            model=args.model,
            output_format=args.output_format,
            input_format=fmt,
        )
        n_rows[fmt] = len(rows)
        unique_ids: set[str] = set()

        for row in rows:
            ids = _parse_atom_ids(row.get("model_answer"))
            if not ids:
                continue
            for atom_id in ids:
                unique_ids.add(atom_id)
                occ_len_counter[fmt][len(atom_id)] += 1
                n_occurrences[fmt] += 1
                bucket = _classify_regex(atom_id)
                regex_counter[fmt][bucket] += 1

        n_unique_ids[fmt] = len(unique_ids)
        for atom_id in unique_ids:
            unique_len_counter[fmt][len(atom_id)] += 1

    length_values = sorted(
        set(unique_len_counter["smiles"])
        .union(unique_len_counter["iupac"])
        .union(occ_len_counter["smiles"])
        .union(occ_len_counter["iupac"])
    )
    if not length_values:
        raise SystemExit("No atom IDs found for selected subset.")

    # Compress long tails to keep the plot readable.
    length_bins = [str(i) for i in range(1, 9)] + [">=9"]

    def _bin_len(length: int) -> str:
        return str(length) if length <= 8 else ">=9"

    unique_len_binned: dict[str, Counter[str]] = {fmt: Counter() for fmt in INPUT_FORMATS}
    occ_len_binned: dict[str, Counter[str]] = {fmt: Counter() for fmt in INPUT_FORMATS}
    for fmt in INPUT_FORMATS:
        for L, c in unique_len_counter[fmt].items():
            unique_len_binned[fmt][_bin_len(int(L))] += int(c)
        for L, c in occ_len_counter[fmt].items():
            occ_len_binned[fmt][_bin_len(int(L))] += int(c)

    waterfall_order = ["a#", "{el}#", "@#", "#", "other"]
    summary_rows: list[dict[str, object]] = []

    for fmt in INPUT_FORMATS:
        for length_label in length_bins:
            summary_rows.append(
                {
                    "panel": "unique_length",
                    "input_format": fmt,
                    "x": length_label,
                    "count": int(unique_len_binned[fmt].get(length_label, 0)),
                    "denominator": int(n_unique_ids[fmt]),
                    "pct": 100.0 * unique_len_binned[fmt].get(length_label, 0) / max(1, n_unique_ids[fmt]),
                }
            )
            summary_rows.append(
                {
                    "panel": "occurrence_length",
                    "input_format": fmt,
                    "x": length_label,
                    "count": int(occ_len_binned[fmt].get(length_label, 0)),
                    "denominator": int(n_occurrences[fmt]),
                    "pct": 100.0 * occ_len_binned[fmt].get(length_label, 0) / max(1, n_occurrences[fmt]),
                }
            )
        for bucket in waterfall_order:
            c = int(regex_counter[fmt].get(bucket, 0))
            d = int(n_occurrences[fmt])
            summary_rows.append(
                {
                    "panel": "regex_waterfall",
                    "input_format": fmt,
                    "x": bucket,
                    "count": c,
                    "denominator": d,
                    "pct": 100.0 * c / max(1, d),
                }
            )

    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17.8, 6.6), dpi=args.dpi)

    bar_width = 0.38
    offset_map = {
        fmt: (i - (len(INPUT_FORMATS) - 1) / 2.0) * bar_width
        for i, fmt in enumerate(INPUT_FORMATS)
    }
    x_len = np.arange(len(length_bins), dtype=float)

    # (a) Total atom-ID occurrences by length.
    ax = axes[0]
    for fmt in INPUT_FORMATS:
        y = np.array([occ_len_binned[fmt].get(k, 0) for k in length_bins], dtype=float)
        ax.bar(x_len + offset_map[fmt], y, width=bar_width, color=INPUT_COLOR[fmt], label=INPUT_LABEL[fmt], alpha=0.95)
    ax.set_box_aspect(1.0)
    ax.set_xticks(x_len)
    ax.set_xticklabels(length_bins, fontsize=TICK_FS)
    ax.set_xlabel("Atom ID length", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Atom ID occurrences", fontsize=AXIS_LABEL_FS)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=TITLE_FS)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=TICK_FS)

    # (b) Unique atom-ID counts by length.
    ax = axes[1]
    for fmt in INPUT_FORMATS:
        y = np.array([unique_len_binned[fmt].get(k, 0) for k in length_bins], dtype=float)
        ax.bar(x_len + offset_map[fmt], y, width=bar_width, color=INPUT_COLOR[fmt], label=INPUT_LABEL[fmt], alpha=0.95)
    ax.set_box_aspect(1.0)
    ax.set_xticks(x_len)
    ax.set_xticklabels(length_bins, fontsize=TICK_FS)
    ax.set_xlabel("Atom ID length", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Unique atom IDs", fontsize=AXIS_LABEL_FS)
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=TITLE_FS)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=TICK_FS)

    # (c) Regex buckets as a share of all atom IDs.
    ax = axes[2]
    x3 = np.arange(len(waterfall_order), dtype=float)
    for i, fmt in enumerate(INPUT_FORMATS):
        y = np.array(
            [100.0 * regex_counter[fmt].get(k, 0) / max(1, n_occurrences[fmt]) for k in waterfall_order],
            dtype=float,
        )
        off = (i - (len(INPUT_FORMATS) - 1) / 2.0) * bar_width
        bars = ax.bar(x3 + off, y, width=bar_width, color=INPUT_COLOR[fmt], label=INPUT_LABEL[fmt], alpha=0.95)
        for bar, val in zip(bars, y):
            if val <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.35,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=PCT_LABEL_FS,
            )
    ax.set_box_aspect(1.0)
    ax.set_xticks(x3)
    ax.set_xticklabels(waterfall_order, fontsize=TICK_FS)
    ax.set_xlabel("Regex bucket", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("% of all atom IDs", fontsize=AXIS_LABEL_FS)
    ax.text(0.02, 0.98, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=TITLE_FS)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_ylim(0, 100)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FS,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_fig, dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote: {out_fig}")
    print(f"Wrote: {out_summary}")
    for fmt in INPUT_FORMATS:
        print(f"{fmt}: rows={n_rows[fmt]}, unique_atom_ids={n_unique_ids[fmt]}, occurrences={n_occurrences[fmt]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
