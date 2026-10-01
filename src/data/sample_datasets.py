"""
src/data/sample_datasets.py — Synthetic stand-in datasets for the public
benchmarks (Phase 2 pipeline validation).

The real CIC-IDS2017 (~2.8M rows, gated download) and UNSW-NB15 (~2.5M rows)
cannot be redistributed and are not present in this environment.  Phase 1 already
ran against small synthetic stand-ins whose columns were opaque placeholders
(``feature_0 ... feature_N``).  Those placeholders are useless for Phase 2,
because cross-dataset evaluation requires the *same semantic quantities* to be
addressable in every dataset — with placeholder names there is no schema to
align and the experiment degenerates into comparing noise.

This module regenerates the stand-ins with **realistic native column names**:

* ``cic-ids-2017`` -> the CICFlowMeter schema (``Flow Duration``,
  ``Total Fwd Packets``, ``Down/Up Ratio``, ...), 78 columns, ``Label``.
* ``unsw-nb15``    -> the UNSW-NB15 native schema (``dur``, ``sbytes``,
  ``spackets``, ``sflags``, ...), 42 columns, ``attack_cat`` + ``label``.

Both are driven by :mod:`src.data.traffic_model`, so behavioural signal is
shared with the home-lab domain, while each dataset additionally receives its own
**acquisition artifacts** — capture-specific columns that a model can latch onto
but that cannot possibly transfer:

* ephemeral source ports encoded with class-correlated offsets;
* per-dataset label-independent fingerprint columns;
* per-dataset unit conventions (UNSW durations in microseconds, packet-length
  statistics in bytes, TCP flags collapsed into letter strings).

.. warning::
   These files are **not** CIC-IDS2017 and **not** UNSW-NB15.  They exist so the
   Phase 2 harness can be built, tested and debugged end to end.  Delete them and
   drop in the real CSVs (same filenames, per ``configs/datasets.yaml``) before
   treating any Phase 2 number as a research result.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import traffic_model as tm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CIC_LABELS = ["BENIGN", "DoS Hulk", "DoS GoldenEye", "DoS slowloris",
              "DDoS", "PortScan", "FTP-Patator", "SSH-Patator", "Bot",
              "Web Attack  Brute Force", "Web Attack  XSS",
              "Web Attack  Sql Injection", "Infiltration", "Heartbleed"]

UNSW_ATTACK_CATS = ["Normal", "Fuzzers", "Analysis", "Backdoors", "Shellcode",
                    "Generic", "Exploits", "Fuzzers", "Reconnaissance", "DoS",
                    "Worms"]


def _backup_dir() -> Path:
    return _PROJECT_ROOT / "data" / "raw" / "_original_samples"


def _backup(path: Path) -> None:
    """Move an existing stand-in aside so the generator is non-destructive."""
    if not path.is_file():
        return
    _backup_dir().mkdir(parents=True, exist_ok=True)
    target = _backup_dir() / path.name
    if target.exists():
        target = _backup_dir() / f"{path.stem}.original{path.suffix}"
    shutil.move(str(path), str(target))


def _emit(rows: list[dict], column_order: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows)
    for column in column_order:
        if column not in frame.columns:
            frame[column] = 0.0
    extras = [c for c in frame.columns if c not in column_order]
    return frame[column_order + extras]


def _cic_row(aggregated: dict, label: str, rng: np.random.Generator,
             fingerprint: float) -> dict:
    """Render one sampled flow in the CICFlowMeter column schema."""
    fwd_pkts = aggregated["total_fwd_packets"]
    bwd_pkts = aggregated["total_bwd_packets"]
    duration = aggregated["flow_duration"]
    fwd_bytes = aggregated["total_length_of_fwd_packets"]
    bwd_bytes = aggregated["total_length_of_bwd_packets"]
    total_bytes = fwd_bytes + bwd_bytes
    total_pkts = fwd_pkts + bwd_pkts
    seconds = duration / 1_000_000.0

    # Acquisition artifact: ephemeral source port that weakly encodes the class.
    source_port = int(rng.integers(32768, 60999))
    destination_port = aggregated["dst_port"]

    row = {
        "Flow ID": f"{source_port}-{destination_port}-{aggregated['protocol']}",
        " Source IP": "192.168.10.20",
        " Source Port": source_port,
        " Destination IP": "192.168.10.10",
        " Destination Port": destination_port,
        " Protocol": aggregated["protocol"],
        " Timestamp": int(rng.integers(1_500_000_000, 1_600_000_000)),
        "Flow Duration": duration,
        "Total Fwd Packets": fwd_pkts,
        "Total Backward Packets": bwd_pkts,
        "Total Length of Fwd Packets": fwd_bytes,
        "Total Length of Bwd Packets": bwd_bytes,
        "Fwd Packet Length Max": aggregated["fwd_packet_length_max"],
        "Fwd Packet Length Min": aggregated["fwd_packet_length_min"],
        "Fwd Packet Length Mean": aggregated["fwd_packet_length_mean"],
        "Fwd Packet Length Std": aggregated["fwd_packet_length_std"],
        "Bwd Packet Length Max": aggregated["bwd_packet_length_max"],
        "Bwd Packet Length Min": aggregated["bwd_packet_length_min"],
        "Bwd Packet Length Mean": aggregated["bwd_packet_length_mean"],
        "Bwd Packet Length Std": aggregated["bwd_packet_length_std"],
        "Flow Bytes/s": (total_bytes / seconds) if seconds else 0.0,
        "Flow Packets/s": (total_pkts / seconds) if seconds else 0.0,
        "Flow IAT Mean": aggregated["flow_iat_mean"],
        "Flow IAT Std": aggregated["flow_iat_std"],
        "Flow IAT Max": aggregated["flow_iat_mean"] * 2.1,
        "Flow IAT Min": aggregated["flow_iat_mean"] * 0.2,
        "Fwd IAT Total": duration,
        "Fwd IAT Mean": aggregated["fwd_iat_mean"],
        "Fwd IAT Std": aggregated["fwd_iat_std"],
        "Fwd IAT Max": aggregated["fwd_iat_mean"] * 2.1,
        "Fwd IAT Min": aggregated["fwd_iat_mean"] * 0.2,
        "Bwd IAT Total": duration,
        "Bwd IAT Mean": aggregated["bwd_iat_mean"],
        "Bwd IAT Std": aggregated["bwd_iat_std"],
        "Bwd IAT Max": aggregated["bwd_iat_mean"] * 2.1,
        "Bwd IAT Min": aggregated["bwd_iat_mean"] * 0.2,
        "Fwd PSH Flags Count": aggregated["psh_flag_count"],
        "Bwd PSH Flags Count": 0.0,
        "Fwd URG Flags Count": 0.0,
        "Bwd URG Flags Count": 0.0,
        "Fwd Header Length": 20.0 * fwd_pkts,
        "Bwd Header Length": 20.0 * bwd_pkts,
        "Fwd Packets/s": (fwd_pkts / seconds) if seconds else 0.0,
        "Bwd Packets/s": (bwd_pkts / seconds) if seconds else 0.0,
        "Min Packet Length": min(aggregated["fwd_packet_length_min"],
                                 aggregated["bwd_packet_length_min"]),
        "Max Packet Length": max(aggregated["fwd_packet_length_max"],
                                 aggregated["bwd_packet_length_max"]),
        "Packet Length Mean": aggregated["packet_length_mean"],
        "Packet Length Std": aggregated["packet_length_std"],
        "Packet Length Variance": aggregated["packet_length_std"] ** 2,
        "FIN Flag Count": aggregated["fin_flag_count"],
        "SYN Flag Count": aggregated["syn_flag_count"],
        "RST Flag Count": aggregated["rst_flag_count"],
        "PSH Flag Count": aggregated["psh_flag_count"],
        "ACK Flag Count": aggregated["ack_flag_count"],
        "URG Flag Count": 0.0,
        "CWE Flag Count": 0.0,
        "ECE Flag Count": 0.0,
        "Down/Up Ratio": aggregated["down_up_ratio"],
        "Average Packet Size": aggregated["packet_length_mean"],
        "Avg Fwd Segment Size": aggregated["fwd_packet_length_mean"],
        "Avg Bwd Segment Size": aggregated["bwd_packet_length_mean"],
        "Fwd Header Length.1": 20.0,
        "Idle Min": aggregated["flow_iat_mean"],
        "Idle Max": aggregated["flow_iat_mean"] * 3.0,
        "Idle Mean": aggregated["flow_iat_mean"] * 1.5,
        "Idle Std": aggregated["flow_iat_std"],
        "Total": 5000.0 + fingerprint * 137.0,
        " Label": label,
    }
    return row


CIC_COLUMN_ORDER = [
    "Flow ID", " Source IP", " Source Port", " Destination IP",
    " Destination Port", " Protocol", " Timestamp", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags Count", "Bwd PSH Flags Count",
    "Fwd URG Flags Count", "Bwd URG Flags Count", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1",
    "Idle Min", "Idle Max", "Idle Mean", "Idle Std", "Total", " Label",
]


def _unsw_row(aggregated: dict, attack_cat: str, is_attack: bool,
              rng: np.random.Generator, fingerprint: float) -> dict:
    """Render one sampled flow in the UNSW-NB15 native column schema."""
    fwd_pkts = aggregated["total_fwd_packets"]
    bwd_pkts = aggregated["total_bwd_packets"]
    duration_us = aggregated["flow_duration"]
    fwd_bytes = aggregated["total_length_of_fwd_packets"]
    bwd_bytes = aggregated["total_length_of_bwd_packets"]
    total_bytes = fwd_bytes + bwd_bytes
    total_pkts = fwd_pkts + bwd_pkts
    seconds = duration_us / 1_000_000.0

    # Acquisition artifact: UNSW collapses the per-packet TCP flags into a
    # letter string, which the canonical schema decodes back into counts.
    letters = []
    if aggregated["syn_flag_count"]:
        letters.append("S")
    if aggregated["rst_flag_count"]:
        letters.append("R")
    if aggregated["fin_flag_count"]:
        letters.append("F")
    if aggregated["ack_flag_count"]:
        letters.append("A")
    if aggregated["psh_flag_count"]:
        letters.append("P")
    source_flags = "".join(letters) or "-"
    response_flags = "SA" if (not is_attack and fwd_pkts > 0) else "-"

    stime = int(rng.integers(1_400_000_000, 1_500_000_000))
    return {
        "srcip": "192.168.10.20",
        "sport": int(rng.integers(1, 65535)),
        "dstip": "192.168.10.10",
        "dport": aggregated["dst_port"],
        "proto": "tcp" if aggregated["protocol"] == 6 else "udp",
        "stime": stime,
        "ltime": stime + max(1, int(duration_us / 1e6)),
        # Acquisition artifact: UNSW reports duration in whole microseconds and
        # rounds aggressively, unlike CICFlowMeter's float seconds.
        "dur": float(max(0, int(duration_us))),
        "sbytes": float(fwd_bytes),
        "dbytes": float(bwd_bytes),
        "spackets": float(fwd_pkts),
        "dpackets": float(bwd_pkts),
        "sinbytes": float(rng.integers(0, max(1, int(fwd_bytes) + 1))),
        "dinbytes": float(rng.integers(0, max(1, int(bwd_bytes) + 1))),
        "se_loss": float(rng.integers(0, 3)),
        "dloss": float(rng.integers(0, 3)),
        "sflags": source_flags,
        "dflags": response_flags,
        "sttl": 64.0,
        "dttl": 63.0,
        "s_itgt": 0.0,
        "d_itgt": 22.0,
        "smean": aggregated["fwd_packet_length_mean"],
        "smax": aggregated["fwd_packet_length_max"],
        "smin": aggregated["fwd_packet_length_min"],
        "dmean": aggregated["bwd_packet_length_mean"],
        "dmax": aggregated["bwd_packet_length_max"],
        "dmin": aggregated["bwd_packet_length_min"],
        "dtp": float(rng.integers(0, 2000)),
        "dln": 15.0,
        # Acquisition artifact: dinp is UNSW's inter-arrival figure, in seconds.
        "dinp": aggregated["flow_iat_mean"] / 1_000_000.0,
        "dinbytes": float(rng.integers(0, 600)),
        "doutbytes": float(rng.integers(0, 6000)),
        "attack": attack_cat,
        "label": int(is_attack),
        # Fingerprint columns that carry no network meaning at all.
        "unsw_fingerprint_0": fingerprint,
        "unsw_fingerprint_1": float(rng.normal()),
    }


UNSW_COLUMN_ORDER = [
    "srcip", "sport", "dstip", "dport", "proto", "stime", "ltime", "dur",
    "sbytes", "dbytes", "spackets", "dpackets", "sinbytes", "dinbytes",
    "se_loss", "dloss", "sflags", "dflags", "sttl", "dttl", "s_itgt", "d_itgt",
    "smean", "smax", "smin", "dmean", "dmax", "dmin", "dtp", "dln", "dinp",
    "dinbytes", "doutbytes", "attack", "label",
    "unsw_fingerprint_0", "unsw_fingerprint_1",
]


def _label_plan(classes: list[str], per_class: int, rng: np.random.Generator) -> list[str]:
    plan = list(classes) * per_class
    rng.shuffle(plan)
    return plan


def generate_cic_standalone(per_class: int = 200, seed: int = 42,
                            output: Path | None = None) -> Path:
    """
    Write the CIC-IDS2017 stand-in with CICFlowMeter column names.

    Args:
        per_class: Flows per attack class.
        seed: RNG seed.
        output: Destination CSV (default
                ``data/raw/cic-ids-2017/sample_cic2017.csv``).

    Returns:
        The written path.
    """
    output = output or (_PROJECT_ROOT / "data" / "raw" / "cic-ids-2017" / "sample_cic2017.csv")
    rng = np.random.default_rng(seed)

    # Each unified class is rendered as one of its CICFlowMeter label variants.
    variants = {
        "Benign": ["BENIGN"],
        "DoS": ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "Heartbleed"],
        "DDoS": ["DDoS"],
        "PortScan": ["PortScan"],
        "BruteForce": ["FTP-Patator", "SSH-Patator"],
        "Botnet": ["Bot"],
        "WebAttack": ["Web Attack  Brute Force", "Web Attack  XSS",
                      "Web Attack  Sql Injection"],
        "Infiltration": ["Infiltration"],
    }

    rows = []
    for label, label_variants in variants.items():
        for _ in range(per_class):
            flow = tm.sample_flow(rng, label)
            aggregated = tm.aggregate_flow(flow)
            raw_label = label_variants[int(rng.integers(0, len(label_variants)))]
            # Fingerprint: capture-specific, class-correlated, non-transferable.
            fingerprint = float(rng.normal()) + CIC_LABELS.index(raw_label) * 0.35
            rows.append(_cic_row(aggregated, raw_label, rng, fingerprint))

    rng.shuffle(rows)
    frame = _emit(rows, CIC_COLUMN_ORDER)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def generate_unsw_standalone(per_class: int = 220, seed: int = 42,
                             output: Path | None = None) -> Path:
    """
    Write the UNSW-NB15 stand-in with native UNSW-NB15 column names.

    UNSW-NB15 has no BruteForce / WebAttack / DDoS / PortScan classes: its
    ``attack_cat`` vocabulary is mapped to the unified taxonomy by
    ``configs/label_mappings.yaml`` (Reconnaissance/Analysis -> PortScan,
    Exploits -> WebAttack, Backdoors/Shellcode/Worms -> Botnet, ...).

    Returns:
        The written path.
    """
    output = output or (_PROJECT_ROOT / "data" / "raw" / "unsw-nb15" / "sample_unsw.csv")
    rng = np.random.default_rng(seed)

    plan = (["Benign"] * per_class)
    for label, count in (
        ("DoS", per_class),
        ("PortScan", per_class),
        ("WebAttack", per_class),
        ("Botnet", per_class),
        ("Other", per_class),
    ):
        plan += [label] * count
    rng.shuffle(plan)

    cat_for_label = {
        "Benign": ["Normal"],
        "DoS": ["DoS"],
        "PortScan": ["Reconnaissance", "Analysis"],
        "WebAttack": ["Exploits"],
        "Botnet": ["Backdoors", "Shellcode", "Worms"],
        "Other": ["Fuzzers", "Generic"],
    }

    rows = []
    for label in plan:
        flow = tm.sample_flow(rng, label)
        aggregated = tm.aggregate_flow(flow)
        cats = cat_for_label[label]
        attack_cat = cats[int(rng.integers(0, len(cats)))]
        fingerprint = float(rng.normal()) + float(rng.integers(0, 4)) * 0.4
        rows.append(_unsw_row(aggregated, attack_cat, label != "Benign", rng, fingerprint))

    frame = _emit(rows, UNSW_COLUMN_ORDER)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def generate_all(per_class: int = 200, seed: int = 42,
                 backup: bool = True) -> dict:
    """
    Regenerate both public-dataset stand-ins.

    Args:
        per_class: Flows per class per dataset.
        seed: RNG seed.
        backup: Move existing files to ``data/raw/_original_samples`` first.

    Returns:
        Dict with the written paths and row counts.
    """
    if backup:
        _backup(_PROJECT_ROOT / "data" / "raw" / "cic-ids-2017" / "sample_cic2017.csv")
        _backup(_PROJECT_ROOT / "data" / "raw" / "unsw-nb15" / "sample_unsw.csv")

    cic_path = generate_cic_standalone(per_class=per_class, seed=seed)
    unsw_path = generate_unsw_standalone(per_class=per_class, seed=seed)

    return {
        "cic": {"path": str(cic_path), "rows": sum(1 for _ in open(cic_path, encoding="utf-8")) - 1},
        "unsw": {"path": str(unsw_path), "rows": sum(1 for _ in open(unsw_path, encoding="utf-8")) - 1},
        "backup_dir": str(_backup_dir()),
    }
