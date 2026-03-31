#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
import numpy as np


FORMATS = ["smiles", "iupac", "graph"]
FORMAT_LABEL = {"smiles": "SMILES", "iupac": "IUPAC", "graph": "MolJSON"}
FORMAT_COLOR = {"smiles": "#E69F00", "iupac": "#56B4E9", "graph": "#009E73"}


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    package_root = _package_root()
    parser = argparse.ArgumentParser(
        description=(
            "Plot aggregate constrained-generation accuracy by model and output format "
            "using the established analysis style."
        )
    )
    parser.add_argument(
        "--nano-checked-csv",
        default=str(package_root / "model_responses" / "checked" / "gpt-5-nano-low-constrained-generation_checked.csv"),
    )
    parser.add_argument(
        "--mini-checked-csv",
        default=str(package_root / "model_responses" / "checked" / "gpt-5-mini-low-constrained-generation_checked.csv"),
    )
    parser.add_argument(
        "--gpt5-checked-csv",
        default=str(package_root / "model_responses" / "checked" / "gpt-5-low-constrained-generation_checked.csv"),
    )
    parser.add_argument(
        "--out-summary-csv",
        default=str(
            package_root / "analysis_outputs" / "tables" / "constrained_generation_aggregate_accuracy_by_model_format_summary.csv"
        ),
    )
    parser.add_argument(
        "--out-fig",
        default=str(
            package_root / "analysis_outputs" / "plots" / "constrained_generation_aggregate_accuracy_by_model_format.png"
        ),
    )
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.35,
        help="Multiply all plot text sizes by this factor.",
    )
    return parser.parse_args()


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    den = 1.0 + (z * z) / n
    ctr = (p + (z * z) / (2.0 * n)) / den
    rad = (z / den) * np.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    return float(max(0.0, ctr - rad)), float(min(1.0, ctr + rad))


def is_true(raw: str) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes"}


def aggregate_checked(path: Path) -> dict[str, tuple[int, int]]:
    counts = {fmt: {"n": 0, "k": 0} for fmt in FORMATS}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fmt = str(row.get("resolved_output_format", "") or "").strip().lower()
            if fmt not in counts:
                continue
            counts[fmt]["n"] += 1
            if is_true(row.get("is_correct", "")):
                counts[fmt]["k"] += 1
    return {fmt: (int(v["n"]), int(v["k"])) for fmt, v in counts.items()}


def main() -> int:
    args = parse_args()
    fs = max(0.1, float(args.font_scale))

    model_sources = [
        ("GPT-5-nano", Path(args.nano_checked_csv)),
        ("GPT-5-mini", Path(args.mini_checked_csv)),
        ("GPT-5", Path(args.gpt5_checked_csv)),
    ]
    for _, path in model_sources:
        if not path.exists():
            raise FileNotFoundError(f"Missing checked file: {path}")

    rows: list[dict[str, object]] = []
    for model_label, checked_path in model_sources:
        agg = aggregate_checked(checked_path)
        for fmt in FORMATS:
            n, k = agg[fmt]
            lo, hi = wilson_interval(k, n)
            rows.append(
                {
                    "model": model_label,
                    "format": fmt,
                    "label": FORMAT_LABEL[fmt],
                    "n": n,
                    "correct": k,
                    "accuracy": (k / n) if n else 0.0,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )

    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "format", "label", "n", "correct", "accuracy", "ci95_low", "ci95_high"],
        )
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(model_sources), dtype=float)
    width = 0.20
    offsets = np.array([-0.23, 0.0, 0.23])

    fig, ax = plt.subplots(figsize=(8.6, 8.0), dpi=260)

    for i, fmt in enumerate(FORMATS):
        vals: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        hi_abs: list[float] = []
        for model_label, _ in model_sources:
            r = next(rr for rr in rows if rr["model"] == model_label and rr["format"] == fmt)
            acc = 100.0 * float(r["accuracy"])
            lo = 100.0 * float(r["ci95_low"])
            hi = 100.0 * float(r["ci95_high"])
            vals.append(acc)
            lows.append(acc - lo)
            highs.append(hi - acc)
            hi_abs.append(hi)

        xpos = x + offsets[i]
        ax.bar(
            xpos,
            vals,
            width=width,
            color=FORMAT_COLOR[fmt],
            edgecolor="none",
            linewidth=0.0,
            label=FORMAT_LABEL[fmt],
            zorder=2,
        )
        ax.errorbar(
            xpos,
            vals,
            yerr=np.array([lows, highs]),
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=3.0,
            zorder=3,
        )
        label_dy = {
            ("GPT-5", "graph"): -1.0,
            ("GPT-5-nano", "iupac"): +3.6,
        }
        label_dx = -0.018
        for j, (xx, yy, upper) in enumerate(zip(xpos, vals, hi_abs)):
            model_label = model_sources[j][0]
            dy = float(label_dy.get((model_label, fmt), 0.0))
            y_try = upper + 0.4
            if y_try <= 99.2:
                y_lab = min(99.0, max(0.0, y_try + dy))
                ax.text(float(xx + label_dx), y_lab, f"{yy:.1f}%", ha="center", va="bottom", fontsize=11 * fs)
            else:
                y_lab = min(99.0, max(0.0, 98.9 + dy))
                ax.text(float(xx + label_dx), y_lab, f"{yy:.1f}%", ha="center", va="top", fontsize=11 * fs)

    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in model_sources], fontsize=16 * fs)
    ax.set_ylabel("Accuracy (%)", fontsize=20 * fs)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=16 * fs)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
    if args.title:
        ax.set_title(args.title, fontsize=20 * fs)
    ax.legend(frameon=False, fontsize=15 * fs, ncol=1, loc="upper left")

    fig.tight_layout()
    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, facecolor="white")
    plt.close(fig)

    print(f"Wrote {out_summary}")
    print(f"Wrote {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
