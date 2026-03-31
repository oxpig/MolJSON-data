from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from utils.answer_checking import AnswerCheckConfig, check_answers_to_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check model answers against question files and write results. "
            "IUPAC outputs are evaluated with the local OPSIN CLI only."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Deprecated package-root override. If set, questions are read from <data-dir>/questions and responses from <data-dir>/model_responses/raw.",
    )
    parser.add_argument(
        "--questions-dir",
        default=None,
        help="Path to question JSONL directory (default: <package-root>/questions).",
    )
    parser.add_argument(
        "--responses-dir",
        default=None,
        help="Path to raw response CSV directory (default: <package-root>/model_responses/raw).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for checked CSVs (default: <package-root>/model_responses/checked).",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional glob pattern for model CSVs (relative to responses dir). By default, runs the packaged standard response files including Haiku.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for answer checking.",
    )
    parser.add_argument(
        "--threads",
        action="store_true",
        help="Use threads instead of processes.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=20,
        help="Task chunksize for parallel checking.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def load_questions(jsonl_paths: list[Path]) -> list[dict]:
    questions: list[dict] = []
    for path in jsonl_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return questions


def load_results(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        row.setdefault("error", "")
    return rows


def default_response_csvs(responses_dir: Path) -> list[Path]:
    names = [
        "claude-haiku-4-5.csv",
        "gpt-5-low.csv",
        "gpt-5-mini-low.csv",
        "gpt-5-nano-low.csv",
    ]
    return [responses_dir / name for name in names if (responses_dir / name).exists()]


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    package_root = Path(args.data_dir) if args.data_dir else script_dir.parent
    questions_dir = Path(args.questions_dir) if args.questions_dir else (package_root / "questions")
    responses_dir = Path(args.responses_dir) if args.responses_dir else (package_root / "model_responses" / "raw")
    output_dir = Path(args.output_dir) if args.output_dir else (package_root / "model_responses" / "checked")
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_paths = sorted(questions_dir.glob("*.jsonl"))
    if not jsonl_paths:
        print(f"No question files found in {questions_dir}")
        return 1

    csv_paths = sorted(responses_dir.glob(args.pattern)) if args.pattern else default_response_csvs(responses_dir)
    if not csv_paths:
        if args.pattern:
            print(f"No model CSVs found for pattern {args.pattern} in {responses_dir}")
        else:
            print(f"No default standard model CSVs found in {responses_dir}")
        return 1

    questions = load_questions(jsonl_paths)
    if not questions:
        print("No questions loaded.")
        return 1

    cfg = AnswerCheckConfig(
        show_progress=not args.no_progress,
        workers=args.workers,
        use_processes=not args.threads,
        chunksize=args.chunksize,
    )

    for csv_path in csv_paths:
        results = load_results(csv_path)
        if not results:
            print(f"{csv_path.name}: no rows found, skipping.")
            continue

        df_checked = check_answers_to_df(questions, results, cfg=cfg)
        out_path = output_dir / f"{csv_path.stem}_checked.csv"
        df_checked.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
