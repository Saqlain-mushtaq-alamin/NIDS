"""
src/utils/provenance.py — Provenance stamping and results validation.

Enforces experimental rigor across all phases:
- Every results JSON/CSV records provenance metadata.
- Rejects writing headline results based on simulated data.
- Enforces N >= 500 rows per test set (and >= 500 attack + >= 500 benign in Phase 5).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute the SHA256 hex digest of a file."""
    path = Path(file_path)
    if not path.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def count_file_rows(file_path: Union[str, Path]) -> int:
    """Count data lines (excluding header) in a CSV file."""
    path = Path(file_path)
    if not path.is_file():
        return 0
    lines = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            lines += 1
    return max(0, lines - 1)


def get_git_commit_hash() -> str:
    """Retrieve the current HEAD git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown_commit"


def create_provenance_stamp(
    data_kind: str,
    dataset_names: List[str],
    source_files: Dict[str, Dict[str, Any]],
    n_rows: int,
    n_train: int,
    n_val: int,
    n_test: int,
    class_counts: Dict[str, int],
    seed_list: List[int],
    script_name: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct a verified provenance dictionary.

    Args:
        data_kind: 'real' or 'simulated'.
        dataset_names: list of datasets involved.
        source_files: dict mapping relative file path to {'rows': int, 'sha256': str}.
        n_rows: total rows loaded.
        n_train: rows in train set.
        n_val: rows in val set.
        n_test: rows in test set.
        class_counts: class label distribution.
        seed_list: seeds used in the experiment.
        script_name: name of the calling driver script.
        extra_metadata: optional extra fields.

    Returns:
        Dict with complete provenance record.
    """
    if data_kind not in ("real", "simulated"):
        raise ValueError(f"data_kind must be 'real' or 'simulated', got '{data_kind}'")

    stamp = {
        "data_kind": data_kind,
        "dataset_names": dataset_names,
        "source_files": source_files,
        "n_rows": n_rows,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "class_counts": class_counts,
        "seed_list": seed_list,
        "git_commit": get_git_commit_hash(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script_name": script_name,
    }
    if extra_metadata:
        stamp.update(extra_metadata)
    return stamp


def validate_results_provenance(
    stamp: Dict[str, Any],
    is_headline: bool = False,
    is_phase5: bool = False,
    min_test_rows: int = 500,
) -> None:
    """
    Validate that results meet scientific validity standards.
    Raises ValueError if invalid.
    """
    data_kind = stamp.get("data_kind", "")
    if is_headline and data_kind == "simulated":
        raise ValueError(
            "SCIENTIFIC INTEGRITY VIOLATION: Headline experiment results cannot "
            "be generated from simulated data (data_kind='simulated')."
        )

    n_test = stamp.get("n_test", 0)
    if n_test < min_test_rows:
        raise ValueError(
            f"SCIENTIFIC INTEGRITY VIOLATION: Test set has {n_test} rows; "
            f"minimum {min_test_rows} required."
        )

    if is_phase5:
        class_counts = stamp.get("class_counts", {})
        attack_count = sum(v for k, v in class_counts.items() if k.lower() != "benign")
        benign_count = sum(v for k, v in class_counts.items() if k.lower() == "benign")
        if attack_count < 500 or benign_count < 500:
            raise ValueError(
                f"SCIENTIFIC INTEGRITY VIOLATION: Phase 5 requires >= 500 attack "
                f"and >= 500 benign flows in test evaluation (got attack={attack_count}, "
                f"benign={benign_count})."
            )


def save_stamped_json(
    payload: Dict[str, Any],
    output_path: Union[str, Path],
    provenance_stamp: Dict[str, Any],
    is_headline: bool = False,
    is_phase5: bool = False,
) -> None:
    """Validate provenance and write JSON results."""
    validate_results_provenance(provenance_stamp, is_headline=is_headline, is_phase5=is_phase5)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload_copy = dict(payload)
    payload_copy["provenance"] = provenance_stamp
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload_copy, f, indent=2)
    logger.info("Saved validated results with provenance to %s", out)


def save_stamped_csv(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    provenance_stamp: Dict[str, Any],
    is_headline: bool = False,
    is_phase5: bool = False,
) -> None:
    """Validate provenance and write CSV results alongside a companion provenance JSON."""
    validate_results_provenance(provenance_stamp, is_headline=is_headline, is_phase5=is_phase5)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    # Save companion provenance
    prov_file = out.with_suffix(".provenance.json")
    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump(provenance_stamp, f, indent=2)
    logger.info("Saved validated CSV to %s (provenance: %s)", out, prov_file)

