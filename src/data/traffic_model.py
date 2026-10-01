"""
src/data/traffic_model.py — Shared, documented behavioural model of network flows.

Every synthetic dataset used in this project (the CIC-IDS2017 stand-in, the
UNSW-NB15 stand-in and the home-lab capture simulator) draws its *behavioural*
features from this one model.  That is deliberate: it lets the cross-dataset
harness exercise a meaningful experiment instead of comparing unrelated noise.

Two kinds of variation are modelled, and they are kept strictly separate:

``behaviour``
    Physically meaningful per-class flow characteristics — packet-count and
    packet-length distributions, inter-arrival times, flow duration, TCP flag
    mix.  These are properties of the *attack*, not of the capture, and they are
    expected to transfer across networks.

``acquisition_artifact``
    Capture-specific nuisance structure — ephemeral source ports, local
    addressing, tool fingerprints, per-dataset column naming and units.  These
    are properties of the *labelling and collection pipeline* and are expected
    NOT to transfer.

.. warning::
   The datasets built on top of this module are **synthetic stand-ins** used to
   validate the Phase 2 pipeline end to end.  They are not CIC-IDS2017, not
   UNSW-NB15 and not real captures.  No scientific claim about cross-dataset
   generalization may be derived from them; replace them with the real datasets
   (see ``configs/datasets.yaml``) before drawing conclusions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BehaviorProfile:
    """
    Flow-level behaviour for one attack class.

    All durations are microseconds, all lengths are TCP/UDP payload bytes.

    Attributes:
        name: Unified taxonomy label.
        fwd_packets: (min, max) forward packet count.
        bwd_packets: (min, max) backward packet count.
        fwd_length: (mean, std, min, max) forward payload bytes.
        bwd_length: (mean, std, min, max) backward payload bytes.
        iat_mean_us: (mean, std) mean forward inter-arrival time.
        duration_us: (mean, std, min, max) flow duration.
        flags: Probability mass over TCP flags ``S``/``A``/``P``/``R``/``F``.
        dst_ports: Destination ports used by the class.
        protocols: ``6`` for TCP, ``17`` for UDP.
        difficulty: Extra within-class spread (lognormal sigma) applied to
            features so classes are not trivially separable.
    """

    name: str
    fwd_packets: tuple[int, int]
    bwd_packets: tuple[int, int]
    fwd_length: tuple[float, float, float, float]
    bwd_length: tuple[float, float, float, float]
    iat_mean_us: tuple[float, float]
    duration_us: tuple[float, float, float, float]
    flags: dict = field(default_factory=dict)
    dst_ports: tuple[int, ...] = (80,)
    protocols: tuple[int, ...] = (6,)
    difficulty: float = 1.0


# TCP flag bit masks in the network-byte-order flags octet.
FLAG_FIN = 0x01
FLAG_SYN = 0x02
FLAG_RST = 0x04
FLAG_PSH = 0x08
FLAG_ACK = 0x10


PROFILES: dict[str, BehaviorProfile] = {
    "Benign": BehaviorProfile(
        name="Benign",
        fwd_packets=(3, 9),
        bwd_packets=(3, 14),
        fwd_length=(420.0, 220.0, 60.0, 1400.0),
        bwd_length=(1150.0, 640.0, 120.0, 8000.0),
        iat_mean_us=(26000.0, 18000.0),
        duration_us=(1_900_000.0, 1_400_000.0, 20_000.0, 9_000_000.0),
        flags={"S": 1.0, "A": 1.0, "P": 1.0, "R": 0.02, "F": 0.35},
        dst_ports=(80, 443, 22, 21, 53),
        protocols=(6,),
        difficulty=1.0,
    ),
    "DoS": BehaviorProfile(
        name="DoS",
        fwd_packets=(1, 2),
        bwd_packets=(0, 1),
        fwd_length=(24.0, 26.0, 0.0, 90.0),
        bwd_length=(20.0, 14.0, 0.0, 60.0),
        iat_mean_us=(240.0, 180.0),
        duration_us=(1_500_000.0, 1_200_000.0, 1_000.0, 6_000_000.0),
        flags={"S": 1.0, "A": 0.9, "P": 0.05, "R": 0.08, "F": 0.03},
        dst_ports=(80,),
        protocols=(6,),
        difficulty=1.1,
    ),
    "DDoS": BehaviorProfile(
        name="DDoS",
        fwd_packets=(4, 22),
        bwd_packets=(0, 1),
        fwd_length=(62.0, 44.0, 0.0, 220.0),
        bwd_length=(0.0, 0.0, 0.0, 40.0),
        iat_mean_us=(140.0, 90.0),
        duration_us=(1_100_000.0, 900_000.0, 2_000.0, 5_000_000.0),
        flags={"S": 1.0, "A": 1.0, "P": 0.5, "R": 0.12, "F": 0.02},
        dst_ports=(80,),
        protocols=(6,),
        difficulty=1.15,
    ),
    "PortScan": BehaviorProfile(
        name="PortScan",
        fwd_packets=(1, 1),
        bwd_packets=(0, 1),
        fwd_length=(0.0, 0.0, 0.0, 0.0),
        bwd_length=(0.0, 0.0, 0.0, 40.0),
        iat_mean_us=(1_200.0, 900.0),
        duration_us=(1_100.0, 700.0, 100.0, 9_000.0),
        flags={"S": 1.0, "A": 0.15, "P": 0.0, "R": 0.45, "F": 0.05},
        dst_ports=tuple(range(1, 1025)),
        protocols=(6,),
        difficulty=1.2,
    ),
    "BruteForce": BehaviorProfile(
        name="BruteForce",
        fwd_packets=(2, 4),
        bwd_packets=(2, 4),
        fwd_length=(190.0, 90.0, 40.0, 620.0),
        bwd_length=(120.0, 70.0, 30.0, 400.0),
        iat_mean_us=(4_800.0, 3_400.0),
        duration_us=(620_000.0, 480_000.0, 40_000.0, 3_000_000.0),
        flags={"S": 1.0, "A": 1.0, "P": 1.0, "R": 0.06, "F": 0.55},
        dst_ports=(22, 21),
        protocols=(6,),
        difficulty=1.05,
    ),
    "WebAttack": BehaviorProfile(
        name="WebAttack",
        fwd_packets=(3, 10),
        bwd_packets=(1, 6),
        fwd_length=(310.0, 210.0, 90.0, 1100.0),
        bwd_length=(740.0, 520.0, 80.0, 5200.0),
        iat_mean_us=(9_800.0, 7_200.0),
        duration_us=(760_000.0, 560_000.0, 30_000.0, 4_500_000.0),
        flags={"S": 1.0, "A": 1.0, "P": 1.0, "R": 0.03, "F": 0.4},
        dst_ports=(80, 8080, 443),
        protocols=(6,),
        difficulty=1.0,
    ),
    "Botnet": BehaviorProfile(
        name="Botnet",
        fwd_packets=(1, 3),
        bwd_packets=(1, 3),
        fwd_length=(140.0, 80.0, 20.0, 480.0),
        bwd_length=(110.0, 70.0, 20.0, 400.0),
        iat_mean_us=(32_000.0, 22_000.0),
        duration_us=(46_000_000.0, 26_000_000.0, 6_000_000.0, 120_000_000.0),
        flags={"S": 0.85, "A": 1.0, "P": 0.9, "R": 0.05, "F": 0.2},
        dst_ports=(4444, 8443, 1337, 31337),
        protocols=(6,),
        difficulty=1.1,
    ),
    "Infiltration": BehaviorProfile(
        name="Infiltration",
        fwd_packets=(2, 7),
        bwd_packets=(2, 9),
        fwd_length=(760.0, 380.0, 200.0, 4000.0),
        bwd_length=(900.0, 460.0, 180.0, 6000.0),
        iat_mean_us=(48_000.0, 34_000.0),
        duration_us=(6_500_000.0, 4_200_000.0, 200_000.0, 30_000_000.0),
        flags={"S": 0.9, "A": 1.0, "P": 0.9, "R": 0.02, "F": 0.3},
        dst_ports=(445, 5555, 4444),
        protocols=(6,),
        difficulty=1.25,
    ),
    # Catch-all bucket: UNSW-NB15 Fuzzers/Generic land here.  Deliberately a
    # broad mixture so the class stays genuinely hard.
    "Other": BehaviorProfile(
        name="Other",
        fwd_packets=(1, 14),
        bwd_packets=(0, 10),
        fwd_length=(300.0, 340.0, 0.0, 3000.0),
        bwd_length=(520.0, 620.0, 0.0, 5000.0),
        iat_mean_us=(15_000.0, 20_000.0),
        duration_us=(1_800_000.0, 2_600_000.0, 500.0, 40_000_000.0),
        flags={"S": 0.95, "A": 1.0, "P": 0.7, "R": 0.08, "F": 0.2},
        dst_ports=(80, 443, 22, 21, 53, 445, 8080),
        protocols=(6,),
        difficulty=2.6,
    ),
}


def _uniform_int(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    low, high = bounds
    return int(rng.integers(low, high + 1))


def _truncated_normal(rng: np.random.Generator, mean: float, std: float,
                      minimum: float, maximum: float) -> float:
    if std <= 0:
        return float(np.clip(mean, minimum, maximum))
    value = rng.normal(mean, std)
    return float(np.clip(value, minimum, maximum))


def sample_packet_length(rng: np.random.Generator, bounds: tuple[float, float, float, float],
                         sigma: float = 1.0) -> float:
    """Draw one payload length from ``(mean, std, min, max)``."""
    mean, std, minimum, maximum = bounds
    value = rng.normal(mean, std * sigma)
    return float(np.clip(value, minimum, maximum))


def sample_flow(rng: np.random.Generator, label: str) -> dict:
    """
    Draw one flow for ``label``.

    Returns a dict with per-packet forward/backward timing and length
    descriptors (``fwd_lengths``, ``bwd_lengths``, ``iat_list``) plus the
    aggregate values needed to emit a native-schema row.
    """
    profile = PROFILES[label]
    sigma = profile.difficulty

    n_fwd = _uniform_int(rng, profile.fwd_packets)
    n_bwd = _uniform_int(rng, profile.bwd_packets)

    fwd_lengths = [
        sample_packet_length(rng, profile.fwd_length, sigma) for _ in range(n_fwd)
    ]
    bwd_lengths = [
        sample_packet_length(rng, profile.bwd_length, sigma) for _ in range(n_bwd)
    ]

    iat_mean = max(1.0, _truncated_normal(
        rng, profile.iat_mean_us[0], profile.iat_mean_us[1] * sigma, 1.0, 5_000_000.0,
    ))
    duration = _truncated_normal(
        rng, profile.duration_us[0], profile.duration_us[1] * sigma,
        profile.duration_us[2], profile.duration_us[3],
    )
    duration = max(duration, iat_mean * max(n_fwd + n_bwd - 1, 0))

    timestamps = np.cumsum(rng.normal(iat_mean, iat_mean * 0.35, size=n_fwd + n_bwd))
    timestamps = np.clip(timestamps, 0.0, None)
    # Scale so the flow spans exactly `duration`.  The relative inter-arrival
    # structure is preserved, and a pcap-derived flow_duration therefore matches
    # this one exactly (see src/data/feature_extractor._flow_to_record).
    span = float(timestamps[-1]) if timestamps.size and timestamps[-1] > 0 else 0.0
    if span > 0:
        timestamps = timestamps * (duration / span)
    fwd_times = timestamps[:n_fwd].tolist()
    bwd_times = timestamps[n_fwd:].tolist()

    flags: list[int] = []
    for index in range(n_fwd):
        mask = 0
        if rng.random() < profile.flags.get("S", 0.0):
            mask |= FLAG_SYN if index == 0 else FLAG_ACK
        if rng.random() < profile.flags.get("A", 0.0):
            mask |= FLAG_ACK
        if rng.random() < profile.flags.get("P", 0.0):
            mask |= FLAG_PSH
        if rng.random() < profile.flags.get("R", 0.0):
            mask |= FLAG_RST
        if rng.random() < profile.flags.get("F", 0.0):
            mask |= FLAG_FIN
        if mask == 0:
            mask = FLAG_ACK
        flags.append(mask)

    return {
        "label": label,
        "duration_us": duration,
        "fwd_lengths": fwd_lengths,
        "bwd_lengths": bwd_lengths,
        "fwd_times": fwd_times,
        "bwd_times": bwd_times,
        "flags": flags,
        "dst_port": int(rng.choice(profile.dst_ports)),
        "protocol": int(rng.choice(profile.protocols)),
    }


def iat_statistics(times: list[float]) -> tuple[float, float]:
    """Return ``(mean, std)`` inter-arrival times in microseconds."""
    if len(times) < 2:
        return (0.0, 0.0)
    ordered = np.sort(np.asarray(times, dtype=np.float64))
    deltas = np.diff(ordered) * 1_000_000.0
    return float(np.mean(deltas)), float(np.std(deltas))

def aggregate_flow(flow: dict) -> dict:
    """
    Reduce a sampled flow into the canonical feature dictionary.

    Mirrors :func:`src.data.feature_extractor._flow_to_record` so pcap-derived
    and CSV-derived feature values are directly comparable.
    """
    fwd_lengths = np.asarray(flow["fwd_lengths"], dtype=np.float64)
    bwd_lengths = np.asarray(flow["bwd_lengths"], dtype=np.float64)
    fwd_bytes = float(fwd_lengths.sum())
    bwd_bytes = float(bwd_lengths.sum())
    all_lengths = np.concatenate([fwd_lengths, bwd_lengths]) if len(fwd_lengths) + len(bwd_lengths) else np.array([0.0])

    fwd_iat_mean, fwd_iat_std = iat_statistics(flow["fwd_times"])
    bwd_iat_mean, bwd_iat_std = iat_statistics(flow["bwd_times"])
    flow_iat_mean, flow_iat_std = iat_statistics(flow["fwd_times"] + flow["bwd_times"])
    duration_us = float(flow["duration_us"])
    seconds = duration_us / 1_000_000.0
    flags = flow["flags"]

    def _count(mask: int) -> float:
        return float(sum(1 for value in flags if value & mask))

    return {
        "flow_duration": duration_us,
        "total_fwd_packets": float(len(fwd_lengths)),
        "total_bwd_packets": float(len(bwd_lengths)),
        "total_length_of_fwd_packets": fwd_bytes,
        "total_length_of_bwd_packets": bwd_bytes,
        "total_length_of_packets": fwd_bytes + bwd_bytes,
        "fwd_packet_length_mean": float(fwd_lengths.mean()) if len(fwd_lengths) else 0.0,
        "fwd_packet_length_max": float(fwd_lengths.max()) if len(fwd_lengths) else 0.0,
        "fwd_packet_length_min": float(fwd_lengths.min()) if len(fwd_lengths) else 0.0,
        "fwd_packet_length_std": float(fwd_lengths.std()) if len(fwd_lengths) else 0.0,
        "bwd_packet_length_mean": float(bwd_lengths.mean()) if len(bwd_lengths) else 0.0,
        "bwd_packet_length_max": float(bwd_lengths.max()) if len(bwd_lengths) else 0.0,
        "bwd_packet_length_min": float(bwd_lengths.min()) if len(bwd_lengths) else 0.0,
        "bwd_packet_length_std": float(bwd_lengths.std()) if len(bwd_lengths) else 0.0,
        "packet_length_mean": float(all_lengths.mean()),
        "packet_length_std": float(all_lengths.std()),
        "flow_iat_mean": flow_iat_mean,
        "flow_iat_std": flow_iat_std,
        "fwd_iat_mean": fwd_iat_mean,
        "fwd_iat_std": fwd_iat_std,
        "bwd_iat_mean": bwd_iat_mean,
        "bwd_iat_std": bwd_iat_std,
        "down_up_ratio": (bwd_bytes / fwd_bytes) if fwd_bytes else 0.0,
        "syn_flag_count": _count(FLAG_SYN),
        "ack_flag_count": _count(FLAG_ACK),
        "rst_flag_count": _count(FLAG_RST),
        "fin_flag_count": _count(FLAG_FIN),
        "psh_flag_count": _count(FLAG_PSH),
        "flow_bytes_per_sec": ((fwd_bytes + bwd_bytes) / seconds) if seconds else 0.0,
        "flow_packets_per_sec": ((len(fwd_lengths) + len(bwd_lengths)) / seconds) if seconds else 0.0,
        "dst_port": flow["dst_port"],
        "protocol": flow["protocol"],
    }
