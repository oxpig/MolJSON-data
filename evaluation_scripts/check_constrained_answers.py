#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MOLJSON_SRC = ROOT / "MolJSON" / "src"
if str(MOLJSON_SRC) not in sys.path:
    sys.path.insert(0, str(MOLJSON_SRC))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdMolDescriptors  # noqa: E402
from moljson import MolFromJSON  # noqa: E402
from utils.opsin import OpsinClient  # noqa: E402


VALID_HALOGENS = ("F", "Cl", "Br")
LOWER_TO_HALOGEN = {h.lower(): h for h in VALID_HALOGENS}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Check constrained-generation responses (smiles, iupac, MolJSON) "
            "against constraints stored in question JSONL using the local OPSIN CLI for IUPAC."
        )
    )
    p.add_argument("--questions", required=True, help="Questions JSONL with 'constraints' field.")
    p.add_argument("--responses-csv", required=True, help="Responses CSV from submission.")
    p.add_argument("--out-csv", required=True, help="Checked output CSV.")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{i} line must be an object.")
            rows.append(obj)
    return rows


def canonicalize_smiles(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    s = str(smiles)
    if s == "":
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def parse_json_text(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def normalize_format(fmt: Any) -> str:
    f = str(fmt or "").strip().lower()
    if f in {"graph", "smiles", "iupac"}:
        return f
    return ""


def extract_scalar_value(payload: Any, key: str) -> str | None:
    if isinstance(payload, dict):
        if isinstance(payload.get(key), str):
            return str(payload.get(key))
        return None
    return None


def extract_graph_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        if "atoms" in payload and "bonds" in payload:
            return payload
    return None


def collect_unique_iupac_answers(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        if normalize_format(row.get("output_format")) != "iupac":
            continue
        model_answer = str(row.get("model_answer", "") or "")
        if not model_answer:
            continue
        payload = parse_json_text(model_answer)
        if payload is None:
            continue
        iupac_text = extract_scalar_value(payload, "iupac")
        if iupac_text:
            names.add(iupac_text)
    return sorted(names)


def lookup_precomputed_opsin_result(
    iupac_name: str,
    *,
    success_map: dict[str, str],
    error_map: dict[str, str],
) -> tuple[str | None, str | None]:
    if iupac_name in success_map:
        return success_map[iupac_name], None
    if iupac_name in error_map:
        return None, error_map[iupac_name]
    return None, None


def project_constraints(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    out: dict[str, Any] = {}
    for key in (
        "connected_components",
        "rings",
        "fused_ring_systems",
        "ring_topology",
        "spiro_centers",
        "halogens_bonded_to_ring_atoms",
        "ring_sizes",
        "halogen_counts",
        "shortest_halogen_paths",
    ):
        if key in raw:
            out[key] = raw[key]
    return out


def count_halogens(mol: Chem.Mol) -> dict[str, int]:
    out = {h: 0 for h in VALID_HALOGENS}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in out:
            out[sym] += 1
    return out


def count_connected_components(mol: Chem.Mol) -> int:
    return int(len(Chem.GetMolFrags(mol)))


def ring_size_histogram(mol: Chem.Mol) -> dict[int, int]:
    out: dict[int, int] = {}
    for ring in mol.GetRingInfo().AtomRings():
        size = int(len(ring))
        out[size] = out.get(size, 0) + 1
    return out


def shortest_halogen_distance(mol: Chem.Mol, h1: str, h2: str) -> int | None:
    idx1 = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == h1]
    idx2 = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == h2]
    if len(idx1) != 1 or len(idx2) != 1:
        return None

    path = Chem.rdmolops.GetShortestPath(mol, int(idx1[0]), int(idx2[0]))
    if not path:
        return None
    return len(path) - 1


def count_fused_ring_systems(mol: Chem.Mol) -> int:
    """
    Count fused ring systems.

    A fused relationship is defined as two rings sharing at least one bond (edge).
    A fused ring system is one connected component in the ring graph formed by
    those fused relationships.
    """
    ring_info = mol.GetRingInfo()
    bond_rings = [set(r) for r in ring_info.BondRings()]
    n = len(bond_rings)
    if n == 0:
        return 0

    # Build adjacency among rings that share at least one bond.
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if bond_rings[i].intersection(bond_rings[j]):
                adj[i].add(j)
                adj[j].add(i)

    # Only rings participating in at least one fused edge contribute.
    participating = {i for i in range(n) if adj[i]}
    if not participating:
        return 0

    visited: set[int] = set()
    systems = 0
    for start in sorted(participating):
        if start in visited:
            continue
        systems += 1
        stack = [start]
        visited.add(start)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
    return systems


def count_spiro_centers(mol: Chem.Mol) -> int:
    return int(rdMolDescriptors.CalcNumSpiroAtoms(mol))


def classify_ring_topology(mol: Chem.Mol) -> str | None:
    """
    Classify benchmark ring topologies.

    Returns:
    - two_separate_rings
    - two_separate_rings_single_bond
    - two_fused_rings_shared_edge
    or None if the molecule does not match one of these classes.
    """
    ring_info = mol.GetRingInfo()
    atom_rings = [set(r) for r in ring_info.AtomRings()]
    bond_rings = [set(r) for r in ring_info.BondRings()]
    if count_spiro_centers(mol) != 0:
        return None

    if len(atom_rings) == 2 and len(bond_rings) == 2:
        shared_bonds = bond_rings[0].intersection(bond_rings[1])
        shared_atoms = atom_rings[0].intersection(atom_rings[1])
        if shared_bonds:
            return "two_fused_rings_shared_edge"
        if shared_atoms:
            return None
        inter_ring_bonds = 0
        ring0 = atom_rings[0]
        ring1 = atom_rings[1]
        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            if (a in ring0 and b in ring1) or (a in ring1 and b in ring0):
                inter_ring_bonds += 1
        if inter_ring_bonds == 1:
            return "two_separate_rings_single_bond"
        return "two_separate_rings"
    return None


def halogens_bonded_to_ring_atoms(mol: Chem.Mol) -> bool:
    """
    Return True iff every F/Cl/Br atom in the molecule is directly bonded to
    a ring atom.
    """
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in VALID_HALOGENS:
            continue
        nbrs = list(atom.GetNeighbors())
        if len(nbrs) != 1:
            return False
        if not nbrs[0].IsInRing():
            return False
    return True


def describe_ring_topology(
    observed_ring_topology: str | None,
    *,
    observed_rings: int,
    observed_fused_ring_systems: int,
    observed_spiro_centers: int,
) -> str:
    if observed_ring_topology is not None:
        return observed_ring_topology
    return (
        "unclassified_ring_topology("
        f"rings={observed_rings}, "
        f"fused_ring_systems={observed_fused_ring_systems}, "
        f"spiro_centers={observed_spiro_centers})"
    )


def evaluate_constraints(mol: Chem.Mol, constraints: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    observed_paths: list[dict[str, Any]] = []

    observed_connected_components = count_connected_components(mol)
    observed_rings = int(mol.GetRingInfo().NumRings())
    observed_fused_ring_systems = int(count_fused_ring_systems(mol))
    observed_ring_topology = classify_ring_topology(mol)
    observed_spiro_centers = int(count_spiro_centers(mol))
    observed_ring_topology_text = describe_ring_topology(
        observed_ring_topology,
        observed_rings=observed_rings,
        observed_fused_ring_systems=observed_fused_ring_systems,
        observed_spiro_centers=observed_spiro_centers,
    )
    observed_halogens_bonded_to_ring_atoms = halogens_bonded_to_ring_atoms(mol)
    observed_ring_sizes = ring_size_histogram(mol)
    observed_halo = count_halogens(mol)

    target_connected = constraints.get("connected_components")
    if isinstance(target_connected, int) and observed_connected_components != target_connected:
        failures.append(
            "connected_components expected "
            f"{target_connected}, got {observed_connected_components}"
        )

    target_rings = constraints.get("rings")
    if isinstance(target_rings, int) and observed_rings != target_rings:
        failures.append(f"rings expected {target_rings}, got {observed_rings}")

    target_fused = constraints.get("fused_ring_systems")
    if isinstance(target_fused, int) and observed_fused_ring_systems != target_fused:
        failures.append(
            "fused_ring_systems expected "
            f"{target_fused}, got {observed_fused_ring_systems}"
        )

    target_ring_topology = constraints.get("ring_topology")
    if isinstance(target_ring_topology, str):
        if target_ring_topology == "two_separate_rings":
            allowed_observed = {"two_separate_rings", "two_separate_rings_single_bond"}
            if observed_ring_topology not in allowed_observed:
                failures.append(
                    "ring_topology expected two_separate_rings, got "
                    f"{observed_ring_topology_text}"
                )
        elif observed_ring_topology != target_ring_topology:
            failures.append(
                "ring_topology expected "
                f"{target_ring_topology}, got {observed_ring_topology_text}"
            )

    target_spiro = constraints.get("spiro_centers")
    if isinstance(target_spiro, int) and observed_spiro_centers != target_spiro:
        failures.append(
            "spiro_centers expected "
            f"{target_spiro}, got {observed_spiro_centers}"
        )

    if (
        constraints.get("halogens_bonded_to_ring_atoms") is True
        and not observed_halogens_bonded_to_ring_atoms
    ):
        failures.append("halogens_bonded_to_ring_atoms expected True, got False")

    target_ring_sizes = constraints.get("ring_sizes")
    if isinstance(target_ring_sizes, dict):
        for k, target_n in target_ring_sizes.items():
            size = int(k)
            got = int(observed_ring_sizes.get(size, 0))
            if got != int(target_n):
                failures.append(
                    f"ring_sizes[{size}] expected {int(target_n)}, got {got}"
                )

    target_halo = constraints.get("halogen_counts", {})
    if isinstance(target_halo, dict):
        for h, target_n in target_halo.items():
            got = observed_halo.get(h, 0)
            if got != target_n:
                failures.append(f"{h} count expected {target_n}, got {got}")

    for pc in constraints.get("shortest_halogen_paths", []):
        h1, h2 = pc["between"]
        target_d = pc["distance_bonds"]
        count1 = observed_halo.get(h1, 0)
        count2 = observed_halo.get(h2, 0)
        if count1 != 1 or count2 != 1:
            observed_paths.append({"between": [h1, h2], "distance_bonds": None})
            failures.append(
                f"shortest path {h1}-{h2} requires exactly one {h1} and one {h2}, "
                f"got {h1}={count1}, {h2}={count2}"
            )
            continue
        got_d = shortest_halogen_distance(mol, h1, h2)
        observed_paths.append({"between": [h1, h2], "distance_bonds": got_d})
        if got_d is None:
            failures.append(f"shortest path {h1}-{h2} expected {target_d}, got missing")
        elif got_d != target_d:
            failures.append(f"shortest path {h1}-{h2} expected {target_d}, got {got_d}")

    return {
        "is_correct": len(failures) == 0,
        "failure_messages": failures,
        "observed_connected_components": observed_connected_components,
        "observed_rings": observed_rings,
        "observed_fused_ring_systems": observed_fused_ring_systems,
        "observed_ring_topology": observed_ring_topology,
        "observed_spiro_centers": observed_spiro_centers,
        "observed_halogens_bonded_to_ring_atoms": observed_halogens_bonded_to_ring_atoms,
        "observed_ring_sizes": observed_ring_sizes,
        "observed_halogen_counts": observed_halo,
        "observed_shortest_halogen_paths": observed_paths,
    }


def check_one(
    row: dict[str, Any],
    *,
    question_map: dict[str, dict[str, Any]],
    opsin_client: OpsinClient,
    opsin_success_map: dict[str, str] | None = None,
    opsin_error_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    def failed(error_detail: str, constraints_json: str) -> dict[str, Any]:
        return {
            "resolved_question_uuid": q_uuid,
            "resolved_output_format": output_format,
            "is_correct": False,
            "error_detail": error_detail,
            "pred_smiles": None,
            "opsin_smiles": None,
            "constraints_json": constraints_json,
            "observed_connected_components": None,
            "observed_rings": None,
            "observed_fused_ring_systems": None,
            "observed_ring_topology": None,
            "observed_spiro_centers": None,
            "observed_halogens_bonded_to_ring_atoms": None,
            "observed_ring_sizes": None,
            "observed_f_count": None,
            "observed_cl_count": None,
            "observed_br_count": None,
            "observed_shortest_halogen_paths": None,
            "constraint_failures": None,
        }

    raw_uuid = str(row.get("uuid", "")).strip()
    q_uuid = raw_uuid.rsplit("__", 1)[0] if "__" in raw_uuid else raw_uuid
    question = question_map.get(q_uuid)
    output_format = normalize_format(row.get("output_format"))
    if not isinstance(question, dict):
        return failed("question_not_found", "{}")
    if not isinstance(question.get("constraints"), dict):
        return failed("invalid_constraints", json.dumps(question.get("constraints"), ensure_ascii=True))
    constraints = project_constraints(question["constraints"])

    model_error = str(row.get("error", "") or "").strip()
    if model_error:
        return failed("response_error", json.dumps(constraints, ensure_ascii=True))

    model_answer = str(row.get("model_answer", "") or "")
    if not model_answer:
        return failed("empty_model_answer", json.dumps(constraints, ensure_ascii=True))

    payload = parse_json_text(model_answer)
    if payload is None:
        return failed("model_answer_not_json", json.dumps(constraints, ensure_ascii=True))

    pred_smiles: str | None = None
    opsin_smiles: str | None = None

    if output_format == "smiles":
        pred_smiles = canonicalize_smiles(extract_scalar_value(payload, "smiles"))
        if pred_smiles is None:
            return failed("smiles_parse_failed", json.dumps(constraints, ensure_ascii=True))

    elif output_format == "iupac":
        iupac_text = extract_scalar_value(payload, "iupac")
        if not iupac_text:
            return failed("iupac_parse_failed", json.dumps(constraints, ensure_ascii=True))
        opsin_success_map = opsin_success_map or {}
        opsin_error_map = opsin_error_map or {}
        opsin_smiles, opsin_error = lookup_precomputed_opsin_result(
            iupac_text,
            success_map=opsin_success_map,
            error_map=opsin_error_map,
        )
        if opsin_smiles is None and opsin_error is None:
            try:
                opsin_smiles = opsin_client.iupac_to_smiles(iupac_text)
            except Exception as e:
                msg = str(e).strip()
                opsin_error = msg if msg else e.__class__.__name__
        if opsin_smiles is None:
            return failed(opsin_error or "opsin_lookup_failed", json.dumps(constraints, ensure_ascii=True))
        pred_smiles = canonicalize_smiles(opsin_smiles)
        if pred_smiles is None:
            out = failed("opsin_smiles_parse_failed", json.dumps(constraints, ensure_ascii=True))
            out["opsin_smiles"] = opsin_smiles
            return out

    elif output_format == "graph":
        graph_payload = extract_graph_payload(payload)
        if graph_payload is None:
            return failed("graph_parse_failed", json.dumps(constraints, ensure_ascii=True))
        try:
            mol = MolFromJSON(graph_payload)
        except Exception:
            return failed("mol_from_graph_failed", json.dumps(constraints, ensure_ascii=True))
        pred_smiles = canonicalize_smiles(
            Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
        )
        if pred_smiles is None:
            return failed("graph_to_smiles_failed", json.dumps(constraints, ensure_ascii=True))

    else:
        return failed(f"unknown_output_format:{output_format}", json.dumps(constraints, ensure_ascii=True))

    mol_eval = Chem.MolFromSmiles(pred_smiles)
    if mol_eval is None:
        out = failed("pred_smiles_parse_failed", json.dumps(constraints, ensure_ascii=True))
        out["pred_smiles"] = pred_smiles
        out["opsin_smiles"] = opsin_smiles
        return out

    eval_result = evaluate_constraints(mol_eval, constraints)
    halo = eval_result["observed_halogen_counts"]
    failures = eval_result["failure_messages"]
    return {
        "resolved_question_uuid": q_uuid,
        "resolved_output_format": output_format,
        "is_correct": bool(eval_result["is_correct"]),
        "error_detail": None if eval_result["is_correct"] else "constraint_mismatch",
        "pred_smiles": pred_smiles,
        "opsin_smiles": opsin_smiles,
        "constraints_json": json.dumps(constraints, ensure_ascii=True),
        "observed_connected_components": eval_result["observed_connected_components"],
        "observed_rings": eval_result["observed_rings"],
        "observed_fused_ring_systems": eval_result["observed_fused_ring_systems"],
        "observed_ring_topology": eval_result["observed_ring_topology"],
        "observed_spiro_centers": eval_result["observed_spiro_centers"],
        "observed_halogens_bonded_to_ring_atoms": eval_result["observed_halogens_bonded_to_ring_atoms"],
        "observed_ring_sizes": json.dumps(eval_result["observed_ring_sizes"], ensure_ascii=True),
        "observed_f_count": halo.get("F"),
        "observed_cl_count": halo.get("Cl"),
        "observed_br_count": halo.get("Br"),
        "observed_shortest_halogen_paths": json.dumps(
            eval_result["observed_shortest_halogen_paths"], ensure_ascii=True
        ),
        "constraint_failures": "; ".join(failures) if failures else None,
    }


def main() -> int:
    args = parse_args()
    questions = read_jsonl(Path(args.questions))
    q_map = {str(q.get("uuid", "")).strip(): q for q in questions if str(q.get("uuid", "")).strip()}

    opsin_client = OpsinClient()

    in_csv = Path(args.responses_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with in_csv.open("r", newline="", encoding="utf-8") as f:
        resp_rows = list(csv.DictReader(f))

    iupac_names = collect_unique_iupac_answers(resp_rows)
    opsin_success_map: dict[str, str] = {}
    opsin_error_map: dict[str, str] = {}
    if iupac_names:
        opsin_success_map, opsin_error_map = opsin_client.batch_lookup(iupac_names)

    out_rows: list[dict[str, Any]] = []
    correct = 0
    for row in resp_rows:
        checked = check_one(
            row,
            question_map=q_map,
            opsin_client=opsin_client,
            opsin_success_map=opsin_success_map,
            opsin_error_map=opsin_error_map,
        )
        merged = dict(row)
        merged.update(checked)
        out_rows.append(merged)
        if bool(checked.get("is_correct")):
            correct += 1

    fields = [
        "uuid",
        "output_format",
        "model",
        "effort",
        "model_answer",
        "output_tokens",
        "error",
        "resolved_question_uuid",
        "resolved_output_format",
        "is_correct",
        "error_detail",
        "pred_smiles",
        "opsin_smiles",
        "constraints_json",
        "observed_connected_components",
        "observed_rings",
        "observed_fused_ring_systems",
        "observed_ring_topology",
        "observed_spiro_centers",
        "observed_halogens_bonded_to_ring_atoms",
        "observed_ring_sizes",
        "observed_f_count",
        "observed_cl_count",
        "observed_br_count",
        "observed_shortest_halogen_paths",
        "constraint_failures",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    n = len(out_rows)
    pct = (100.0 * correct / n) if n else 0.0
    print(f"Wrote {out_csv}")
    print(f"Correct: {correct}/{n} ({pct:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
