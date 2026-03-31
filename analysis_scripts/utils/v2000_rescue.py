from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


# Heuristic for lines that look like molfile atom records:
#   x y z element [optional extra integer fields]
ATOM_LINE_RE = re.compile(
    r"^\s*[+-]?\d+(?:\.\d+)?\s+[+-]?\d+(?:\.\d+)?\s+[+-]?\d+(?:\.\d+)?\s+[A-Za-z\*]{1,3}(?:\s+.*)?$"
)

# Small helper regex used when checking bond-line tokens.
INT_RE = re.compile(r"^[+-]?\d+$")


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = package_root()
    parser = argparse.ArgumentParser(
        description=(
            "Rescue V2000 molfile outputs by detecting atom and bond blocks directly, "
            "inferring the counts line, and rebuilding a clean molblock."
        )
    )
    parser.add_argument(
        "--checked-csv",
        default=str(root / "model_responses" / "checked" / "gpt-5-mini-low_checked.csv"),
        help="Checked CSV containing V2000_MOLBLOCK rows to analyse.",
    )
    parser.add_argument(
        "--questions-dir",
        default=str(root / "questions"),
        help="Directory containing packaged question JSONLs.",
    )
    parser.add_argument(
        "--out-summary",
        default=str(root / "analysis_outputs" / "tables" / "v2000_rescue" / "gpt5mini_v2000_rescue_summary.csv"),
        help="Where to write the aggregated rescue summary CSV.",
    )
    parser.add_argument(
        "--out-details",
        default=str(root / "analysis_outputs" / "tables" / "v2000_rescue" / "gpt5mini_v2000_rescue_details.csv"),
        help="Where to write the row-level rescue details CSV.",
    )
    return parser.parse_args()


def load_questions(questions_dir: Path) -> dict[str, dict]:
    """Load packaged question JSONLs so we can recover the ground-truth molecule."""
    out: dict[str, dict] = {}
    for path in sorted(questions_dir.glob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                q = json.loads(line)
                out[q["uuid"]] = q
    return out


def get_ground_truth_smiles(question: dict) -> str:
    """Read the gold molecule SMILES from packaged question metadata."""
    meta = question.get("meta") or {}
    if isinstance(meta, dict):
        mol = meta.get("molecule")
        if isinstance(mol, dict) and isinstance(mol.get("smiles"), str):
            return mol["smiles"]
    return ""


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string with RDKit so equality checks are stable."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def extract_molblock(model_answer: object) -> Optional[str]:
    """
    Extract the raw molblock text from the stored JSON payload.

    The packaged checked CSV stores the original model answer string, and for V2000
    rows the expected payload shape is {"V2000_MOLBLOCK": "..."}.
    """
    if model_answer is None:
        return None

    raw = str(model_answer).strip()
    if not raw or raw.lower() == "nan":
        return None

    obj = json.loads(raw)
    if not isinstance(obj, dict):
        return None

    value = obj.get("V2000_MOLBLOCK")
    return value if isinstance(value, str) else None


def mol_from_block(molblock: Optional[str]):
    """
    Parse a molblock strictly with RDKit.

    We keep strictParsing=True because the purpose of the rescue is to rebuild a
    structurally valid CTAB block, not to rely on RDKit's permissive parser.
    """
    if molblock is None:
        return None
    try:
        return Chem.MolFromMolBlock(
            molblock,
            sanitize=True,
            removeHs=False,
            strictParsing=True,
        )
    except Exception:
        return None


def mol_to_smiles(mol) -> Optional[str]:
    """Convert a parsed RDKit molecule to canonical SMILES."""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1"}


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def is_atom_line(line: str) -> bool:
    """
    Decide whether a line looks like a molfile atom record.

    We deliberately use a simple shape-based heuristic:
    - three numeric coordinates
    - then an element token like C, N, Cl, Br, or *
    - optionally followed by more fields
    """
    if not line.strip():
        return False
    if not ATOM_LINE_RE.match(line):
        return False

    parts = line.split()
    if len(parts) < 4:
        return False

    try:
        float(parts[0])
        float(parts[1])
        float(parts[2])
    except ValueError:
        return False

    return True


def is_bond_line(line: str) -> bool:
    """
    Decide whether a line looks like a molfile bond record.

    We only require the basic leading structure:
    - atom index 1
    - atom index 2
    - bond order
    """
    parts = line.split()
    if len(parts) < 3:
        return False
    if not all(INT_RE.match(part) for part in parts):
        return False

    try:
        a1 = int(parts[0])
        a2 = int(parts[1])
        order = int(parts[2])
    except ValueError:
        return False

    return a1 >= 1 and a2 >= 1 and order >= 0


def counts_line(n_atoms: int, n_bonds: int) -> str:
    """Create a conventional V2000 counts line from inferred atom/bond counts."""
    return f"{n_atoms:>3}{n_bonds:>3}  0  0  0  0            999 V2000"


def reconstruct_from_inferred_blocks(
    molblock: Optional[str],
) -> tuple[Optional[str], str, Optional[int], Optional[int], Optional[int]]:
    """
    Rebuild a molblock by detecting atom and bond blocks directly.

    This rescue ignores the original counts line entirely.

    For every line in the text:
    1. treat it as a possible atom-block start if it looks like an atom line
    2. take the longest contiguous atom-line run
    3. immediately after that, take the longest contiguous bond-line run
    4. synthesize a fresh counts line from those run lengths
    5. rebuild a clean molblock and test whether RDKit can parse it

    We keep the best candidate that parses strictly. If nothing parses, we still
    return the best-looking candidate so the caller can record what was attempted.
    """
    if molblock is None:
        return None, "missing_molblock", None, None, None

    lines = molblock.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # Best candidate by raw block size, even if RDKit still rejects it.
    best_any: tuple[int, int, int, int] | None = None

    # Best candidate among those that RDKit can parse strictly.
    best_parsed: tuple[int, int, int, int, str] | None = None

    for atom_start, line in enumerate(lines):
        if not is_atom_line(line):
            continue

        # Extend the atom block as long as the following lines still look like atoms.
        atom_end = atom_start
        while atom_end < len(lines) and is_atom_line(lines[atom_end]):
            atom_end += 1
        n_atoms = atom_end - atom_start
        if n_atoms <= 0:
            continue

        # Extend the bond block immediately after the atom block.
        bond_end = atom_end
        while bond_end < len(lines) and is_bond_line(lines[bond_end]):
            bond_end += 1
        n_bonds = bond_end - atom_end

        score = n_atoms + n_bonds
        if best_any is None or score > best_any[0]:
            best_any = (score, atom_start, atom_end, bond_end)

        # Preserve any trailing property lines after the inferred bond block.
        rebuilt_lines = [
            "",
            "",
            "",
            counts_line(n_atoms, n_bonds),
            *lines[atom_start:atom_end],
            *lines[atom_end:bond_end],
            *lines[bond_end:],
        ]
        rebuilt = "\n".join(rebuilt_lines)

        # Prefer candidates that parse, and among them prefer bigger blocks and
        # earlier starts.
        parsed = mol_from_block(rebuilt)
        if parsed is not None:
            parsed_score = (score, -atom_start)
            if best_parsed is None or parsed_score > (best_parsed[0], -best_parsed[1]):
                best_parsed = (score, atom_start, n_atoms, n_bonds, rebuilt)

    if best_parsed is not None:
        _, atom_start, n_atoms, n_bonds, rebuilt = best_parsed
        return rebuilt, "inferred_blocks_rebuilt", atom_start, n_atoms, n_bonds

    if best_any is None:
        return None, "no_atom_block_candidate", None, None, None

    # Nothing parsed, but we still return the strongest-looking candidate for
    # diagnostics and downstream inspection.
    _, atom_start, atom_end, bond_end = best_any
    n_atoms = atom_end - atom_start
    n_bonds = bond_end - atom_end
    rebuilt = "\n".join(
        [
            "",
            "",
            "",
            counts_line(n_atoms, n_bonds),
            *lines[atom_start:atom_end],
            *lines[atom_end:bond_end],
            *lines[bond_end:],
        ]
    )
    return rebuilt, "inferred_blocks_unparsed", atom_start, n_atoms, n_bonds


def run_rescue(checked_csv: Path, questions_dir: Path) -> tuple[list[dict], list[dict]]:
    """
    Evaluate all V2000 rows before and after inferred-count rescue.

    The "before" result is strict RDKit parsing of the raw model molblock.
    The "after" result is strict RDKit parsing of the rebuilt molblock with an
    inferred counts line.
    """
    questions = load_questions(questions_dir)
    details: list[dict] = []

    with checked_csv.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("output_format") != "V2000_MOLBLOCK":
                continue

            uuid = row["uuid"]
            question = questions[uuid]

            gold_smiles = canonicalize_smiles(get_ground_truth_smiles(question))
            pred_block = extract_molblock(row.get("model_answer"))

            # Baseline: parse the raw model molblock exactly as stored.
            pred_before = mol_to_smiles(mol_from_block(pred_block))

            # Rescue: rebuild from inferred atom/bond blocks, then parse again.
            rebuilt_block, rebuild_status, atom_start_idx, n_atoms, n_bonds = reconstruct_from_inferred_blocks(pred_block)
            pred_after = mol_to_smiles(mol_from_block(rebuilt_block))

            details.append(
                {
                    "uuid": uuid,
                    "dataset": row.get("dataset", ""),
                    "input_format": row.get("input_format", ""),
                    "output_format": row.get("output_format", ""),
                    "orig_is_correct": as_bool(row.get("is_correct")),
                    "orig_error_detail": row.get("error_detail", ""),
                    "atom_block_start_idx_before": atom_start_idx if atom_start_idx is not None else "",
                    "n_atoms_inferred": n_atoms if n_atoms is not None else "",
                    "n_bonds_inferred": n_bonds if n_bonds is not None else "",
                    "rebuild_status": rebuild_status,
                    "strict_parse_before": pred_before is not None,
                    "strict_parse_after": pred_after is not None,
                    "gold_smiles": gold_smiles or "",
                    "pred_smiles_before": pred_before or "",
                    "pred_smiles_after": pred_after or "",
                    "match_before": bool(gold_smiles and pred_before and gold_smiles == pred_before),
                    "match_after": bool(gold_smiles and pred_after and gold_smiles == pred_after),
                }
            )

    by_input: dict[str, list[dict]] = defaultdict(list)
    for row in details:
        by_input[str(row["input_format"])].append(row)

    summary_rows: list[dict] = []
    for input_format, rows in sorted(by_input.items()):
        orig_correct = sum(r["orig_is_correct"] for r in rows)
        match_after = sum(r["match_after"] for r in rows)
        newly_rescued = sum((not r["orig_is_correct"]) and r["match_after"] for r in rows)
        parse_before = sum(r["strict_parse_before"] for r in rows)
        parse_after = sum(r["strict_parse_after"] for r in rows)
        rebuild_counts = Counter(str(r["rebuild_status"]) for r in rows)
        summary_rows.append(
            {
                "input_format": input_format,
                "n_rows": len(rows),
                "orig_correct_count": orig_correct,
                "orig_correct_pct": pct(orig_correct, len(rows)),
                "match_after_count": match_after,
                "match_after_pct": pct(match_after, len(rows)),
                "newly_rescued_count": newly_rescued,
                "newly_rescued_pct_points": pct(match_after, len(rows)) - pct(orig_correct, len(rows)),
                "strict_parse_before_count": parse_before,
                "strict_parse_after_count": parse_after,
                "strict_parse_gain_count": parse_after - parse_before,
                "inferred_blocks_rebuilt_count": rebuild_counts["inferred_blocks_rebuilt"],
                "inferred_blocks_unparsed_count": rebuild_counts["inferred_blocks_unparsed"],
                "no_atom_block_candidate_count": rebuild_counts["no_atom_block_candidate"],
                "missing_molblock_count": rebuild_counts["missing_molblock"],
            }
        )

    total = len(details)
    orig_correct = sum(r["orig_is_correct"] for r in details)
    match_after = sum(r["match_after"] for r in details)
    newly_rescued = sum((not r["orig_is_correct"]) and r["match_after"] for r in details)
    parse_before = sum(r["strict_parse_before"] for r in details)
    parse_after = sum(r["strict_parse_after"] for r in details)
    rebuild_counts = Counter(str(r["rebuild_status"]) for r in details)
    summary_rows.append(
        {
            "input_format": "ALL",
            "n_rows": total,
            "orig_correct_count": orig_correct,
            "orig_correct_pct": pct(orig_correct, total),
            "match_after_count": match_after,
            "match_after_pct": pct(match_after, total),
            "newly_rescued_count": newly_rescued,
            "newly_rescued_pct_points": pct(match_after, total) - pct(orig_correct, total),
            "strict_parse_before_count": parse_before,
            "strict_parse_after_count": parse_after,
            "strict_parse_gain_count": parse_after - parse_before,
            "inferred_blocks_rebuilt_count": rebuild_counts["inferred_blocks_rebuilt"],
            "inferred_blocks_unparsed_count": rebuild_counts["inferred_blocks_unparsed"],
            "no_atom_block_candidate_count": rebuild_counts["no_atom_block_candidate"],
            "missing_molblock_count": rebuild_counts["missing_molblock"],
        }
    )

    return summary_rows, details


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    summary_rows, detail_rows = run_rescue(Path(args.checked_csv), Path(args.questions_dir))
    write_csv(Path(args.out_summary), summary_rows)
    write_csv(Path(args.out_details), detail_rows)

    for row in summary_rows:
        print(row)
    print(f"Wrote {args.out_summary}")
    print(f"Wrote {args.out_details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
