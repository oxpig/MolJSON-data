# utils/answer_checking.py
from __future__ import annotations

import json
import io
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import redirect_stderr

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit import rdBase
from rdkit.Chem import inchi as rd_inchi
from tqdm.auto import tqdm

import selfies as sf

ROOT = Path(__file__).resolve().parents[1]
MOLJSON_SRC = ROOT / "MolJSON" / "src"
if str(MOLJSON_SRC) not in sys.path:
    sys.path.insert(0, str(MOLJSON_SRC))

from moljson import MolFromJSON
from utils.opsin import OpsinClient

Json = Dict[str, Any]

RDLogger.DisableLog("rdApp.*")
rdBase.LogToPythonStderr()


@dataclass
class AnswerCheckConfig:
    opsin_batch_workers: int = 8
    opsin_batch_chunk_size: int = 10000
    show_progress: bool = True
    workers: int = 1
    use_processes: bool = True
    chunksize: int = 20


class AnswerVerifier:
    _GRAPH_OUTPUT_FORMATS = {"graph", "moleculegraph"}

    def __init__(
        self,
        questions: List[Json],
        cfg: Optional[AnswerCheckConfig] = None,
    ):
        self.cfg = cfg or AnswerCheckConfig()
        self.question_dict: Dict[str, Json] = {q["uuid"]: q for q in questions}

        self._canon_cache: Dict[str, Optional[str]] = {}
        self._opsin_client = OpsinClient()

    # -----------------------
    # Utilities
    # -----------------------
    def _is_graph_format(self, output_format: str) -> bool:
        return output_format.strip().lower() in self._GRAPH_OUTPUT_FORMATS

    def _get_ground_truth_smiles(self, q: Json) -> str:
        meta = q.get("meta") or {}
        if isinstance(meta, dict):
            mol = meta.get("molecule")
            if isinstance(mol, dict):
                s = mol.get("smiles")
                if isinstance(s, str):
                    return s

        return ""

    # -----------------------
    # Parsing model_answer
    # -----------------------
    def _parse_output_text_as_json(self, output_text: Any) -> Optional[Any]:
        if not isinstance(output_text, str):
            return None
        if not output_text:
            return None
        try:
            return json.loads(output_text)
        except Exception:
            return None

    def _extract_non_graph_answer(
        self,
        output_text: Any,
        output_format: str,
    ) -> Optional[str]:
        parsed = self._parse_output_text_as_json(output_text)
        if not isinstance(parsed, dict):
            return None

        val = parsed.get(output_format)
        if not isinstance(val, str):
            return None
        return val

    def _extract_graph_dict(self, output_text: Any) -> Optional[Json]:
        parsed = self._parse_output_text_as_json(output_text)
        if not isinstance(parsed, dict):
            return None

        if "atoms" in parsed and "bonds" in parsed:
            return parsed  # type: ignore[return-value]

        return None

    def _extract_integer_answer(self, output_text: Any) -> Optional[int]:
        """
        Accept the packaged shortest-path response shape:
          - JSON: {"integer": "7"}
        """
        if not isinstance(output_text, str):
            return None

        parsed = self._parse_output_text_as_json(output_text)
        if not isinstance(parsed, dict):
            return None

        v = parsed.get("integer")
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return None
        return None

    # -----------------------
    # Canonicalization
    # -----------------------
    def canonicalise_smiles(self, smiles: Any) -> Optional[str]:
        key = smiles if isinstance(smiles, str) else repr(smiles)
        if key in self._canon_cache:
            return self._canon_cache[key]

        if not isinstance(smiles, str):
            self._canon_cache[key] = None
            return None

        if smiles == "":
            self._canon_cache[key] = None
            return None

        m = Chem.MolFromSmiles(smiles)
        out = None if m is None else Chem.MolToSmiles(m, canonical=True)
        self._canon_cache[key] = out
        return out

    def _rdkit_error_message(self, stderr_text: str) -> str:
        """Condense RDKit stderr into a readable checker error; we do not use this message text in downstream analysis."""
        lines = [ln.strip() for ln in stderr_text.splitlines() if ln.strip()]
        clean_lines = [re.sub(r"^\[[^\]]+\]\s*", "", ln) for ln in lines]
        if not clean_lines:
            return "MolFromSmiles returned None (no RDKit log)"

        def _strip_prefix(ln: str) -> str:
            if "SMILES Parse Error:" in ln:
                return ln.split("SMILES Parse Error:", 1)[1].strip()
            return ln.strip()

        def _score(ln: str) -> tuple[int, int]:
            msg = _strip_prefix(ln)
            lower = msg.lower()
            if "explicit valence" in lower or "valence" in lower:
                return (90, len(msg))
            if "can't kekulize" in lower or "kekulize" in lower:
                return (80, len(msg))
            if "marked aromatic" in lower or "aromatic" in lower:
                return (70, len(msg))
            if "duplicated ring closure" in lower:
                return (65, len(msg))
            if "unclosed ring" in lower or "ring closure" in lower or "ring" in lower:
                return (60, len(msg))
            if "extra open parentheses" in lower or "extra close parentheses" in lower:
                return (50, len(msg))
            if "syntax error" in lower:
                return (40, len(msg))
            if "failed parsing smiles" in lower:
                return (0, 0)
            if "smiles parse error" in ln.lower():
                return (10, len(msg))
            return (20, len(msg))

        chosen = max(clean_lines, key=_score)

        if "SMILES Parse Error:" in chosen:
            msg = chosen.split("SMILES Parse Error:", 1)[1].strip()
        else:
            msg = chosen

        msg = re.sub(r"\s*while parsing:.*$", "", msg).strip()
        msg = re.sub(r"\s*for input:.*$", "", msg).strip()
        msg = re.sub(r"\s*at position \d+.*$", "", msg).strip()
        return msg if msg else "MolFromSmiles returned None"

    def _smiles_to_canonical_with_error(self, smiles: Any) -> tuple[Optional[str], Optional[str]]:
        if not isinstance(smiles, str):
            return None, "SMILES output was not a string"

        if smiles == "":
            return None, "SMILES output was empty"

        stderr_buf = io.StringIO()
        RDLogger.EnableLog("rdApp.error")
        try:
            with redirect_stderr(stderr_buf):
                mol = Chem.MolFromSmiles(smiles)
        finally:
            RDLogger.DisableLog("rdApp.error")

        if mol is None:
            return None, self._rdkit_error_message(stderr_buf.getvalue())
        return Chem.MolToSmiles(mol, canonical=True), None

    def check_smiles_equal(self, pred_smiles: str, ground_truth_smiles: str) -> bool:
        gt = self.canonicalise_smiles(ground_truth_smiles)
        pred = self.canonicalise_smiles(pred_smiles)
        return (gt is not None) and (pred is not None) and (pred == gt)

    # -----------------------
    # OPSIN
    # -----------------------
    def _get_opsin_result(self, iupac_name: str) -> tuple[Optional[str], Optional[str]]:
        try:
            smiles = self._opsin_client.iupac_to_smiles(iupac_name)
        except Exception as e:
            msg = str(e).strip()
            return None, msg if msg else e.__class__.__name__
        if isinstance(smiles, str) and smiles:
            return smiles, None
        return None, "OPSIN returned no SMILES"

    # -----------------------
    # Conversions to SMILES for checking
    # -----------------------
    def _smiles_from_inchi(self, inchi_str: str) -> Optional[str]:
        """Convert InChI to canonical SMILES; on the packaged response set, varying sanitize/removeHs did not change accuracy."""
        m = rd_inchi.MolFromInchi(inchi_str, sanitize=True, removeHs=False)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)

    def _smiles_from_selfies(self, selfies_str: str) -> Optional[str]:
        """Decode SELFIES with compatible=True because model outputs sometimes use older SELFIES variants, which lets more strings parse."""
        try:
            s = sf.decoder(selfies_str, compatible=True)
        except Exception:
            return None
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)

    def _smiles_from_v2000_molblock(self, molblock: str) -> Optional[str]:
        """Parse V2000 molblocks to canonical SMILES; removeHs=False matches the packaged checked outputs."""
        m = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=False)  # strictParsing=False can slightly increase accuracy on the packaged responses.
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)

    # -----------------------
    # Main checker
    # -----------------------
    def check_answer(
        self,
        uuid: str,
        model_answer: Any,
        *,
        precomputed_opsin_smiles: Optional[str] = None,
        precomputed_opsin_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        if uuid not in self.question_dict:
            raise KeyError(f"uuid={uuid} not found in questions")

        q = self.question_dict[uuid]
        output_format = str(q.get("output_format", "")).strip()
        output_format_l = output_format.lower()
        ground_truth_smiles = self._get_ground_truth_smiles(q)

        # NEW: integer tasks (shortest_path)
        if output_format_l in ("integer", "int"):
            gt = q.get("answer", None)
            try:
                gt_int = int(gt)
            except Exception:
                gt_int = None

            pred_int = self._extract_integer_answer(model_answer)
            if pred_int is None or gt_int is None:
                return {"is_correct": False, "opsin_smiles": None, "pred_smiles": None, "error_detail": "integer_parse_failed"}

            ok = (pred_int == gt_int)
            return {
                "is_correct": ok,
                "opsin_smiles": None,
                "pred_smiles": None,
                "error_detail": None if ok else "integer_mismatch",
            }

        if model_answer is None:
            return {"is_correct": False, "opsin_smiles": None, "pred_smiles": None, "error_detail": "empty_answer"}

        # GRAPH path
        if self._is_graph_format(output_format):
            graph = self._extract_graph_dict(model_answer)
            if graph is None:
                return {
                    "is_correct": False,
                    "opsin_smiles": None,
                    "pred_smiles": None,
                    "error_detail": "MolJSON graph payload missing atoms/bonds",
                }

            try:
                mol = MolFromJSON(graph)
                pred_smiles = Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
                if not pred_smiles:
                    return {
                        "is_correct": False,
                        "opsin_smiles": None,
                        "pred_smiles": None,
                        "error_detail": "MolFromJSON returned None",
                    }
                ok = self.check_smiles_equal(pred_smiles, ground_truth_smiles)
                return {
                    "is_correct": ok,
                    "opsin_smiles": None,
                    "pred_smiles": pred_smiles,
                    "error_detail": None if ok else "graph_smiles_mismatch",
                }
            except Exception as e:
                msg = str(e).strip()
                detail = f"{e.__class__.__name__}: {msg}" if msg else e.__class__.__name__
                return {"is_correct": False, "opsin_smiles": None, "pred_smiles": None, "error_detail": detail}

        # NON-GRAPH: parse {"<output_format>": "..."}
        intended = self._extract_non_graph_answer(model_answer, output_format)
        if intended is None or not intended:
            return {"is_correct": False, "opsin_smiles": None, "pred_smiles": None, "error_detail": "non_graph_parse_failed"}

        # SMILES
        if output_format_l == "smiles":
            pred_smiles, parse_error = self._smiles_to_canonical_with_error(intended)
            if pred_smiles is None:
                return {
                    "is_correct": False,
                    "opsin_smiles": None,
                    "pred_smiles": None,
                    "error_detail": parse_error or "SMILES parse failed",
                }
            ok = self.check_smiles_equal(pred_smiles, ground_truth_smiles)
            return {
                "is_correct": ok,
                "opsin_smiles": None,
                "pred_smiles": pred_smiles,
                "error_detail": None if ok else "smiles_mismatch",
            }

        # IUPAC
        if output_format_l == "iupac":
            opsin_smiles: Optional[str] = None
            opsin_error: Optional[str] = None
            if isinstance(precomputed_opsin_smiles, str) and precomputed_opsin_smiles:
                opsin_smiles = precomputed_opsin_smiles
                opsin_error = None
            elif isinstance(precomputed_opsin_error, str) and precomputed_opsin_error:
                opsin_smiles = None
                opsin_error = precomputed_opsin_error

            if opsin_smiles is None and opsin_error is None:
                opsin_smiles, opsin_error = self._get_opsin_result(intended)

            if not opsin_smiles or not isinstance(opsin_smiles, str):
                return {
                    "is_correct": False,
                    "opsin_smiles": None,
                    "pred_smiles": None,
                    "error_detail": opsin_error or "OPSIN lookup failed",
                }
            ok = self.check_smiles_equal(opsin_smiles, ground_truth_smiles)
            return {
                "is_correct": ok,
                "opsin_smiles": opsin_smiles,
                "pred_smiles": opsin_smiles,
                "error_detail": None if ok else "iupac_smiles_mismatch",
            }

        # SELFIES -> SMILES
        if output_format_l == "selfies":
            pred_smiles = self._smiles_from_selfies(intended)
            if not pred_smiles:
                return {"is_correct": False, "opsin_smiles": None, "pred_smiles": None, "error_detail": "selfies_decode_failed"}
            ok = self.check_smiles_equal(pred_smiles, ground_truth_smiles)
            return {
                "is_correct": ok,
                "opsin_smiles": None,
                "pred_smiles": pred_smiles,
                "error_detail": None if ok else "selfies_smiles_mismatch",
            }

        # INCHI -> SMILES
        if output_format_l == "inchi":
            pred_smiles = self._smiles_from_inchi(intended)
            if not pred_smiles:
                return {"is_correct": False, "opsin_smiles": None, "pred_smiles": None, "error_detail": "inchi_parse_failed"}
            ok = self.check_smiles_equal(pred_smiles, ground_truth_smiles)
            return {
                "is_correct": ok,
                "opsin_smiles": None,
                "pred_smiles": pred_smiles,
                "error_detail": None if ok else "inchi_smiles_mismatch",
            }

        # V2000 MOLBLOCK -> SMILES
        if output_format == "V2000_MOLBLOCK" or output_format_l == "v2000_molblock":
            pred_smiles = self._smiles_from_v2000_molblock(intended)
            if not pred_smiles:
                return {"is_correct": False, "opsin_smiles": None, "pred_smiles": None, "error_detail": "molblock_parse_failed"}
            ok = self.check_smiles_equal(pred_smiles, ground_truth_smiles)
            return {
                "is_correct": ok,
                "opsin_smiles": None,
                "pred_smiles": pred_smiles,
                "error_detail": None if ok else "molblock_smiles_mismatch",
            }

        return {"is_correct": False, "opsin_smiles": None, "pred_smiles": None, "error_detail": f"unknown_output_format:{output_format}"}

_WORKER_VERIFIER: Optional[AnswerVerifier] = None


def _init_worker(
    questions: List[Json],
    cfg_dict: Dict[str, Any],
) -> None:
    global _WORKER_VERIFIER
    cfg = AnswerCheckConfig(**cfg_dict)
    _WORKER_VERIFIER = AnswerVerifier(questions, cfg=cfg)


def _check_row(args: tuple[int, str, Any, Optional[str], Optional[str]]) -> tuple[int, Dict[str, Any]]:
    idx, uuid, model_answer, precomputed_opsin_smiles, precomputed_opsin_error = args
    if _WORKER_VERIFIER is None:
        raise RuntimeError("Worker verifier not initialized")
    checked = _WORKER_VERIFIER.check_answer(
        uuid=uuid,
        model_answer=model_answer,
        precomputed_opsin_smiles=precomputed_opsin_smiles,
        precomputed_opsin_error=precomputed_opsin_error,
    )
    return idx, checked


def _flatten_questions_for_merge(questions: List[Json]) -> pd.DataFrame:
    """
    Build an analysis-friendly question dataframe from the NEW question format.

    Pulls:
      - core: uuid, category, input_format, output_format, answer, verification_method
      - dataset: meta.dataset (if present)
      - molecule identity: smiles/inchi/inchikey/cid (from meta.molecule)
      - common features: n_rings/n_atoms/has_charged (from meta.features)
      - (optional) stored input: meta.input (if present)
    """
    rows: List[Dict[str, Any]] = []

    for q in questions:
        meta = q.get("meta") or {}
        mol = meta.get("molecule") if isinstance(meta, dict) else None
        feats = meta.get("features") if isinstance(meta, dict) else None
        main = mol if isinstance(mol, dict) else {}

        row = {
            # core
            "uuid": q.get("uuid"),
            "category": q.get("category"),
            "input_format": q.get("input_format"),
            "output_format": q.get("output_format"),
            "answer": q.get("answer"),
            "verification_method": q.get("verification_method"),

            # NEW: dataset label (optional)
            "dataset": meta.get("dataset") if isinstance(meta, dict) else None,

            # identity
            "smiles": main.get("smiles") if isinstance(main, dict) else None,
            "inchi": main.get("inchi") if isinstance(main, dict) else None,
            "inchikey": main.get("inchikey") if isinstance(main, dict) else None,
            "cid": main.get("cid") if isinstance(main, dict) else None,
        }

        # features (optional)
        if isinstance(feats, dict):
            row["n_rings"] = feats.get("n_rings")
            row["n_atoms"] = feats.get("n_atoms")
            row["has_charged"] = feats.get("has_charged")

        # (optional) keep input string if you stored it
        if isinstance(meta, dict) and "input" in meta:
            row["input"] = meta.get("input")

        rows.append(row)

    return pd.DataFrame(rows)


def _collect_unique_iupac_answers(
    verifier: AnswerVerifier,
    df: pd.DataFrame,
) -> list[str]:
    names: set[str] = set()
    for row in df.itertuples(index=False):
        output_format = str(getattr(row, "output_format", "") or "").strip().lower()
        if output_format != "iupac":
            continue
        intended = verifier._extract_non_graph_answer(row.model_answer, "iupac")
        if not intended:
            continue
        names.add(intended)
    return sorted(names)


def _collect_precomputed_opsin_results(
    verifier: AnswerVerifier,
    df: pd.DataFrame,
    opsin_success_map: Dict[str, str],
    opsin_error_map: Dict[str, str],
) -> list[tuple[Optional[str], Optional[str]]]:
    out: list[tuple[Optional[str], Optional[str]]] = []
    for row in df.itertuples(index=False):
        output_format = str(getattr(row, "output_format", "") or "").strip().lower()
        if output_format != "iupac":
            out.append((None, None))
            continue

        intended = verifier._extract_non_graph_answer(row.model_answer, "iupac")
        if not intended:
            out.append((None, None))
            continue

        if intended in opsin_success_map:
            out.append((opsin_success_map[intended], None))
        elif intended in opsin_error_map:
            out.append((None, opsin_error_map[intended]))
        else:
            out.append((None, None))
    return out



def check_answers_to_df(
    questions: List[Json],
    results: List[Json],
    *,
    cfg: Optional[AnswerCheckConfig] = None,
) -> pd.DataFrame:
    """
    Analysis-friendly df_checked:
    - includes result fields (model, effort, model_answer, etc.)
    - includes question core fields + useful meta fields for plotting/analysis
    - includes checker outputs (is_correct, error_detail, etc.)
    """
    cfg = cfg or AnswerCheckConfig()

    df_q = _flatten_questions_for_merge(questions)
    df_r = pd.DataFrame(results)

    # Required question fields
    for col in ["uuid", "category", "input_format", "output_format", "answer", "verification_method"]:
        if col not in df_q.columns:
            raise KeyError(f"questions missing required column after flatten: {col}")

    # Required result fields
    for col in ["uuid", "model", "effort", "model_answer", "output_tokens", "error"]:
        if col not in df_r.columns:
            raise KeyError(f"results missing required column: {col}")

    if "output_format" in df_r.columns:
        df_r = df_r.drop(columns=["output_format"])

    df = df_r.merge(df_q, on="uuid", how="left")

    if df["output_format"].isna().any():
        missing = df.loc[df["output_format"].isna(), "uuid"].head(10).tolist()
        raise KeyError(f"Some results UUIDs not found in questions (showing up to 10): {missing}")

    bootstrap_verifier = AnswerVerifier(questions, cfg=cfg)
    iupac_names = _collect_unique_iupac_answers(bootstrap_verifier, df)
    opsin_success_map: Dict[str, str] = {}
    opsin_error_map: Dict[str, str] = {}
    if iupac_names:
        opsin_success_map, opsin_error_map = bootstrap_verifier._opsin_client.batch_lookup(
            iupac_names,
            workers=cfg.opsin_batch_workers,
            chunk_size=cfg.opsin_batch_chunk_size,
        )
    precomputed_opsin_results = _collect_precomputed_opsin_results(
        bootstrap_verifier,
        df,
        opsin_success_map,
        opsin_error_map,
    )

    verifier = AnswerVerifier(questions, cfg=cfg)

    if cfg.workers > 1:
        worker_cfg = AnswerCheckConfig(**asdict(cfg))
        worker_cfg.show_progress = False
        cfg_dict = asdict(worker_cfg)
        tasks = [
            (
                i,
                str(row.uuid),
                row.model_answer,
                precomputed_opsin_results[i][0],
                precomputed_opsin_results[i][1],
            )
            for i, row in enumerate(df.itertuples(index=False))
        ]
        executor_cls = ProcessPoolExecutor if cfg.use_processes else ThreadPoolExecutor
        with executor_cls(
            max_workers=cfg.workers,
            initializer=_init_worker,
            initargs=(questions, cfg_dict),
        ) as executor:
            results = executor.map(_check_row, tasks, chunksize=cfg.chunksize)
            if cfg.show_progress:
                results = tqdm(results, total=len(tasks), desc="Checking answers")
            checked_rows = [None] * len(tasks)
            for idx, checked in results:
                checked_rows[idx] = checked
    else:
        iterator = df.itertuples(index=False)
        if cfg.show_progress:
            iterator = tqdm(iterator, total=len(df), desc="Checking answers")

        checked_rows = []
        for idx, row in enumerate(iterator):
            checked = verifier.check_answer(
                uuid=str(row.uuid),
                model_answer=row.model_answer,
                precomputed_opsin_smiles=precomputed_opsin_results[idx][0],
                precomputed_opsin_error=precomputed_opsin_results[idx][1],
            )
            checked_rows.append(checked)

    df_checked_extra = pd.DataFrame(checked_rows)
    df = pd.concat([df.reset_index(drop=True), df_checked_extra.reset_index(drop=True)], axis=1)

    drop_output_columns = [col for col in ("prompt", "answer") if col in df.columns]
    if drop_output_columns:
        df = df.drop(columns=drop_output_columns)

    # Keep a sensible column order, but don't drop other columns
    preferred = [
        "uuid",
        "category",
        "model",
        "effort",
        "input_format",
        "output_format",
        "verification_method",
        "is_correct",
        "error_detail",
        "model_answer",
        "output_tokens",
        "error",
        "smiles",
        "inchi",
        "inchikey",
        "cid",
        "n_rings",
        "n_atoms",
        "has_charged",
        "input",
        "opsin_smiles",
        "pred_smiles",
    ]

    # return all columns, but ordered with preferred first
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


if __name__ == "__main__":
    pass
