#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MOLJSON_SRC = ROOT / "evaluation_scripts" / "MolJSON" / "src"
if str(MOLJSON_SRC) not in sys.path:
    sys.path.insert(0, str(MOLJSON_SRC))

from moljson import GetSchema  # noqa: E402

GRAPH_ALIASES = {"graph", "moljson", "moleculegraph"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Submit constrained-generation prompts using structured JSON output schemas "
            "for smiles, iupac, and MolJSON."
        )
    )
    p.add_argument("--questions", required=True, help="Input questions JSONL.")
    p.add_argument("--out-csv", required=True, help="Output CSV for responses.")
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--thinking-budget-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--timeout-s", type=int, default=300)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--formats", default="smiles,iupac,moljson")
    p.add_argument("--limit", type=int, default=0, help="Optional max number of rows after expansion.")
    p.add_argument(
        "--no-suffix-uuid",
        action="store_true",
        help="Keep original uuid when expanding formats (default adds __format suffix).",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Disable resume behavior. Safety guard: this requires a new/empty --out-csv "
            "to avoid accidental appends."
        ),
    )
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} invalid json: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{i} line must be an object.")
            out.append(obj)
    return out


def normalize_format(fmt: str) -> str:
    f = str(fmt).strip().lower()
    if f in GRAPH_ALIASES:
        return "graph"
    if f in {"smiles", "iupac"}:
        return f
    raise ValueError(f"Unsupported format: {fmt}")


def parse_formats_arg(raw: str) -> list[str]:
    if not raw.strip():
        return []
    out: list[str] = []
    for item in raw.split(","):
        if not item.strip():
            continue
        out.append(normalize_format(item))
    return out


def expand_questions(
    questions: list[dict[str, Any]],
    *,
    formats: list[str],
    suffix_uuid: bool,
) -> list[dict[str, Any]]:
    if not formats:
        return questions
    out: list[dict[str, Any]] = []
    for q in questions:
        base_uuid = str(q.get("uuid", "")).strip()
        if not base_uuid:
            raise KeyError("All rows must include non-empty uuid.")
        for fmt in formats:
            qq = dict(q)
            qq["output_format"] = fmt
            if suffix_uuid:
                qq["uuid"] = f"{base_uuid}__{fmt}"
            out.append(qq)
    return out


def sanitize_schema_for_anthropic(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: sanitize_schema_for_anthropic(v) for k, v in node.items()}

        # Anthropic rejects min/max constraints in this flow. For bounded integer
        # fields, convert to an explicit enum when the range is small.
        t = out.get("type")
        min_v = out.get("minimum")
        max_v = out.get("maximum")
        if (
            t == "integer"
            and isinstance(min_v, int)
            and isinstance(max_v, int)
            and min_v <= max_v
            and (max_v - min_v) <= 32
        ):
            out["enum"] = list(range(min_v, max_v + 1))

        for k in {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}:
            out.pop(k, None)
        return out
    if isinstance(node, list):
        return [sanitize_schema_for_anthropic(x) for x in node]
    return node


def schema_for_format(fmt: str, graph_schema: dict[str, Any]) -> dict[str, Any]:
    if fmt == "graph":
        return sanitize_schema_for_anthropic(graph_schema)
    return {
        "type": "object",
        "properties": {
            fmt: {
                "type": "string",
                "description": (
                    f"Molecule written as {fmt} ONLY. "
                    "Do not ask clarifying questions. Do not write any comments."
                ),
            }
        },
        "required": [fmt],
        "additionalProperties": False,
    }


def extract_output_tokens(resp: Any) -> int | None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    v = getattr(usage, "output_tokens", None)
    return v if isinstance(v, int) else None


def extract_text_content(resp: Any) -> str:
    parts: list[str] = []
    content = getattr(resp, "content", None)
    if not isinstance(content, list):
        return ""
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts).strip()


def is_retryable(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    return any(x in name for x in ("ratelimit", "timeout", "apierror", "serviceunavailable", "overloaded")) or any(
        x in msg for x in ("429", "rate limit", "timeout", "temporarily", "try again", "503", "502", "500", "504")
    )


def backoff_seconds(attempt: int) -> float:
    return min(10.0, 0.5 * (2 ** max(0, attempt - 1))) * (0.7 + 0.6 * random.random())


def question_key(q: dict[str, Any]) -> str:
    uuid = str(q.get("uuid", "")).strip()
    if not uuid:
        raise KeyError("Question missing uuid.")
    fmt = normalize_format(str(q.get("output_format", "")))
    return f"{uuid}::{fmt}"


def row_key(row: dict[str, Any]) -> str | None:
    uuid = str(row.get("uuid", "")).strip()
    fmt_raw = str(row.get("output_format", "")).strip()
    if not uuid or not fmt_raw:
        return None
    try:
        fmt = normalize_format(fmt_raw)
    except Exception:
        return None
    return f"{uuid}::{fmt}"


def load_existing_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    out: set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row_key(row)
            if key:
                out.add(key)
    return out


def load_done_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    out: set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row_key(row)
            err = str(row.get("error", "")).strip()
            answer = str(row.get("model_answer", "")).strip()
            if key and answer and not err:
                out.add(key)
    return out


async def submit_one(
    client: "AsyncAnthropic",
    q: dict[str, Any],
    *,
    model: str,
    thinking_budget_tokens: int,
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
    graph_schema: dict[str, Any],
) -> dict[str, Any]:
    uuid = str(q.get("uuid", "")).strip()
    prompt = str(q.get("prompt", "")).strip()
    if not uuid:
        raise KeyError("Question missing uuid.")
    if not prompt:
        raise KeyError(f"Question {uuid} missing prompt.")

    fmt = normalize_format(str(q.get("output_format", "")))
    schema = schema_for_format(fmt, graph_schema)

    err: str | None = None
    output_text: str | None = None
    output_tokens: int | None = None

    for attempt in range(1, max_retries + 2):
        try:
            resp = await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    thinking={"type": "enabled", "budget_tokens": thinking_budget_tokens},
                    output_config={"format": {"type": "json_schema", "schema": schema}},
                ),
                timeout=timeout_s,
            )
            raw_text = extract_text_content(resp)
            if not raw_text:
                raise ValueError("No JSON text content found in Anthropic response.")
            parsed = json.loads(raw_text)
            if not isinstance(parsed, dict):
                raise ValueError("JSON response was not an object.")
            output_text = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            output_tokens = extract_output_tokens(resp)
            err = None
            break
        except Exception as e:
            msg = str(e).strip()
            err = msg if msg else e.__class__.__name__
            if attempt <= max_retries and is_retryable(e):
                await asyncio.sleep(backoff_seconds(attempt))
                continue
            break

    return {
        "uuid": uuid,
        "output_format": fmt,
        "model": model,
        "effort": str(thinking_budget_tokens),
        "model_answer": output_text,
        "output_tokens": output_tokens,
        "error": err,
    }


async def run(args: argparse.Namespace) -> int:
    try:
        from anthropic import AsyncAnthropic
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'anthropic'. Install with: pip install anthropic"
        ) from e

    if int(args.thinking_budget_tokens) < 1024:
        raise ValueError("--thinking-budget-tokens must be >= 1024 for manual thinking mode.")

    questions = read_jsonl(Path(args.questions))
    questions = expand_questions(
        questions,
        formats=parse_formats_arg(args.formats),
        suffix_uuid=(not args.no_suffix_uuid),
    )
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
    if not questions:
        print("No questions to submit.")
        return 1

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    target_keys = {question_key(q) for q in questions}
    existing_keys = load_existing_keys(out_csv)
    unexpected_keys = existing_keys.difference(target_keys)
    if unexpected_keys:
        sample = ", ".join(sorted(unexpected_keys)[:5])
        raise ValueError(
            "Refusing to write into existing --out-csv because it contains rows from a "
            f"different question set (sample keys: {sample}). Use a fresh output filename."
        )

    if args.no_resume and existing_keys:
        raise ValueError(
            "--no-resume requires a new/empty --out-csv. Existing rows detected; "
            "use a fresh output filename."
        )

    if out_csv.exists() and not args.no_resume:
        done = load_done_keys(out_csv)
        before = len(questions)
        questions = [q for q in questions if question_key(q) not in done]
        print(f"Resume enabled: {before - len(questions)} already-complete rows skipped.")

    if not questions:
        print("All rows already complete.")
        return 0

    graph_schema = GetSchema()
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(max(1, int(args.concurrency)))
    lock = asyncio.Lock()
    total = len(questions)
    completed = 0

    fields = ["uuid", "output_format", "model", "effort", "model_answer", "output_tokens", "error"]
    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
            f.flush()

        async def worker(q: dict[str, Any]) -> None:
            nonlocal completed
            async with sem:
                row = await submit_one(
                    client,
                    q,
                    model=args.model,
                    thinking_budget_tokens=args.thinking_budget_tokens,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout_s,
                    max_retries=args.max_retries,
                    graph_schema=graph_schema,
                )
            async with lock:
                writer.writerow(row)
                f.flush()
                completed += 1
                if completed == 1 or completed % 10 == 0 or completed == total:
                    print(f"Completed {completed}/{total}")

        tasks = [asyncio.create_task(worker(q)) for q in questions]
        try:
            await asyncio.gather(*tasks)
        finally:
            await client.close()

    print(f"Wrote results to {out_csv}")
    return 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
