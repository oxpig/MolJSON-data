"""OPSIN wrapper utilities.

OPSIN must be installed separately from https://github.com/dan2097/opsin.
The analyses in this work used OPSIN 2.9.0.
"""

from __future__ import annotations
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _truncate(text: str, limit: int) -> str:
    if len(text) > limit:
        return f"{text[:limit]}..."
    return text


def _candidate_java_bins() -> list[Path]:
    candidates: list[Path] = [
        Path("/opt/homebrew/opt/openjdk/bin/java"),
        Path("/usr/local/opt/openjdk/bin/java"),
    ]
    java_from_path = shutil.which("java")
    if java_from_path:
        candidates.append(Path(java_from_path))
    candidates.append(Path("/usr/bin/java"))
    return candidates


def _candidate_jar_paths() -> list[Path]:
    home = Path.home()
    candidates: list[Path] = []
    for root in (
        home / "tools" / "opsin",
        Path("/usr/local/share/opsin"),
        Path("/opt/homebrew/share/opsin"),
    ):
        if root.exists():
            candidates.extend(sorted(root.glob("opsin-cli-*-jar-with-dependencies.jar"), reverse=True))
    return candidates


def _resolve_java_bin(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"Java binary not found at {explicit}")
    for candidate in _candidate_java_bins():
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Java binary not found. Set OPSIN_JAVA_BIN or install Java so `java` is on PATH."
    )


def _resolve_jar_path(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"OPSIN CLI jar not found at {explicit}")
    for candidate in _candidate_jar_paths():
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "OPSIN CLI jar not found. Set OPSIN_CLI_JAR to the downloaded OPSIN jar path."
    )


class OpsinClient:
    def __init__(
        self,
        *,
        java_bin: str | Path | None = None,
        jar_path: str | Path | None = None,
    ) -> None:
        self.java_bin = _resolve_java_bin(Path(java_bin) if java_bin else None)
        self.jar_path = _resolve_jar_path(Path(jar_path) if jar_path else None)

    def iupac_to_smiles(self, iupac_name: str) -> str:
        validated = self._validate_name(iupac_name)
        with tempfile.TemporaryDirectory(prefix="opsin_single_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "input.txt"
            output_path = tmpdir_path / "output.txt"
            stderr_path = tmpdir_path / "stderr.txt"
            input_path.write_text(validated + "\n", encoding="utf-8")

            with stderr_path.open("w", encoding="utf-8") as stderr_handle:
                result = subprocess.run(
                    [
                        str(self.java_bin),
                        "-jar",
                        str(self.jar_path),
                        "-n",
                        str(input_path),
                        str(output_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_handle,
                    text=True,
                )

            if result.returncode != 0:
                stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
                if stderr_text:
                    raise ValueError(_truncate(stderr_text, 1000))
                raise ValueError(f"OPSIN CLI exited with status {result.returncode}")

            output_lines = output_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if output_lines:
                smiles, _, _name = output_lines[0].partition("\t")
                smiles = smiles.strip()
                if smiles:
                    return smiles

            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
            if stderr_text:
                raise ValueError(_truncate(stderr_text, 1000))
            raise ValueError("No SMILES found")

    def batch_lookup(
        self,
        names: list[str],
        *,
        workers: int = 8,
        chunk_size: int = 10000,
    ) -> tuple[dict[str, str], dict[str, str]]:
        validated_names = sorted({self._validate_name(name) for name in names if isinstance(name, str) and name})
        if not validated_names:
            return {}, {}

        success: dict[str, str] = {}
        errors: dict[str, str] = {}
        safe_names = [name for name in validated_names if self._is_batch_safe_name(name)]
        unsafe_names = [name for name in validated_names if not self._is_batch_safe_name(name)]

        chunks = [
            safe_names[i : i + max(1, chunk_size)]
            for i in range(0, len(safe_names), max(1, chunk_size))
        ]
        max_workers = max(1, workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_batch_chunk, idx, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                _chunk_id, chunk_success, chunk_errors = future.result()
                success.update(chunk_success)
                errors.update(chunk_errors)

        for name in unsafe_names:
            try:
                success[name] = self.iupac_to_smiles(name)
            except Exception as e:
                msg = str(e).strip()
                errors[name] = _truncate(msg, 1000) if msg else e.__class__.__name__
        return success, errors

    def _run_batch_chunk(
        self,
        chunk_id: int,
        names: list[str],
    ) -> tuple[int, dict[str, str], dict[str, str]]:
        success: dict[str, str] = {}
        errors: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix=f"opsin_chunk_{chunk_id:04d}_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "input.txt"
            output_path = tmpdir_path / "output.txt"
            stderr_path = tmpdir_path / "stderr.txt"
            input_path.write_text("".join(name + "\n" for name in names), encoding="utf-8")

            with stderr_path.open("w", encoding="utf-8") as stderr_handle:
                result = subprocess.run(
                    [
                        str(self.java_bin),
                        "-jar",
                        str(self.jar_path),
                        "-n",
                        str(input_path),
                        str(output_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_handle,
                    text=True,
                )

            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
            stderr_lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
            output_lines = []
            if output_path.exists():
                output_lines = output_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()

            pending = set(names)
            if stderr_text:
                fallback_error = _truncate(stderr_text, 1000)
            elif result.returncode != 0:
                fallback_error = f"OPSIN CLI exited with status {result.returncode}"
            else:
                fallback_error = "OPSIN local CLI returned no SMILES"

            per_name_errors: dict[str, str] = {}
            for line in stderr_lines:
                for name in names:
                    if line.startswith(name):
                        per_name_errors[name] = _truncate(line, 1000)
                        break

            for line in output_lines:
                smiles, _, tail = line.partition("\t")
                if tail not in pending:
                    continue
                smiles = smiles.strip()
                pending.remove(tail)
                if smiles:
                    success[tail] = smiles
                else:
                    errors[tail] = per_name_errors.get(tail, fallback_error)

            # Retry unresolved names individually so each row gets a name-specific
            # OPSIN result or error instead of inheriting the shared chunk stderr.
            for name in names:
                if name not in pending:
                    continue
                try:
                    success[name] = self.iupac_to_smiles(name)
                except Exception as e:
                    msg = str(e).strip()
                    if msg:
                        errors[name] = _truncate(msg, 1000)
                    else:
                        errors[name] = per_name_errors.get(name, fallback_error)
        return chunk_id, success, errors

    @staticmethod
    def _validate_name(iupac_name: str) -> str:
        if not isinstance(iupac_name, str):
            raise TypeError("iupac_name must be a string")
        if not iupac_name:
            raise ValueError("iupac_name must be a non-empty string")
        return iupac_name

    @staticmethod
    def _is_batch_safe_name(iupac_name: str) -> bool:
        return ("\n" not in iupac_name) and ("\r" not in iupac_name)


def iupac_to_smiles(
    iupac_name: str,
    *,
    java_bin: str | Path | None = None,
    jar_path: str | Path | None = None,
) -> str:
    client = OpsinClient(java_bin=java_bin, jar_path=jar_path)
    return client.iupac_to_smiles(iupac_name)
