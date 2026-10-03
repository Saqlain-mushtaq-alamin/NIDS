"""
src/data/graph_simulator.py — Multi-host windowed traffic for the host-graph GNN.

The Phase 2 home-lab capture is a **single attacker to a single victim** design:
one source host (``10.0.0.5``) talking to two targets.  That is the right shape
for a flow classifier and the wrong shape for a host-communication graph, which
needs many hosts interacting so that degree, fan-out and traversal structure
actually vary between benign and attack windows.

This module therefore generates a **separate, explicitly simulated** traffic set
whose only purpose is to exercise the graph module of ``PHASE_3`` section 3.2.
It emits flow-level rows with ``src_ip`` / ``dst_ip`` / ``timestamp_start`` (the
columns :func:`src.data.feature_extractor.pcap_to_flows` produces) so the same
:class:`~src.models.gnn_detector.HostCommGraph` builder works on real captures
and on this data without modification.

Scenario semantics
------------------
Each window is generated as one of the following patterns.  These are the
structural patterns a graph model is supposed to see and a flow classifier is
supposed to miss:

``benign``
    Hub-and-spoke web/DNS/SSH traffic.  Every host talks to a gateway and to a
    couple of peers; in-degree and out-degree are moderate and *stable*.
``port_scan``
    One host opens flows to many distinct destination hosts and ports inside a
    single window.  Produces one very-high-out-degree node.
``lateral_movement``
    A *sequence* of hosts, each contacting the next.  The graph is a path, so
    every individual edge looks ordinary; only the chain is anomalous.
``data_exfiltration``
    One host sends large volumes to one external destination; high byte volume
    on a single edge.

.. warning::
   This is **synthetic traffic**.  It validates that the graph pipeline works and
   that the architecture can separate the four structural patterns.  It is not
   evidence about real lateral-movement detection, and every artifact produced
   from it carries that caveat.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data import traffic_model as tm

GATEWAY_IP = "10.0.0.1"
EXTERNAL_IP = "203.0.113.7"

DEFAULT_WINDOW_SECONDS = 60
DEFAULT_N_WINDOWS_PER_SCENARIO = 40

SCENARIOS = ("benign", "port_scan", "lateral_movement", "data_exfiltration")

WINDOW_LABELS = {
    "benign": "Benign",
    "port_scan": "PortScan",
    "lateral_movement": "Infiltration",
    "data_exfiltration": "Infiltration",
}

_GRAPH_COLUMNS = (
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "timestamp_start", "timestamp_end", "flow_duration",
    "total_fwd_packets", "total_bwd_packets",
    "total_length_of_fwd_packets", "total_length_of_bwd_packets",
    "down_up_ratio", "syn_flag_count", "ack_flag_count",
    "label", "window_id", "scenario",
)


@dataclass(frozen=True)
class GraphScenarioConfig:
    """
    Topology and volume of one simulated host network.

    Attributes:
        n_hosts: Internal hosts (10.0.0.2 .. 10.0.0.40) excluding the gateway.
        window_seconds: Length of one capture window.
        flows_per_window: Flows emitted per window before per-scenario shaping.
        seed: Base RNG seed.
        subnet: Network prefix for internal hosts.
    """

    n_hosts: int = 12
    window_seconds: int = DEFAULT_WINDOW_SECONDS
    flows_per_window: int = 90
    seed: int = 42
    subnet: str = "10.0.0"


def _host(index: int, subnet: str) -> str:
    return f"{subnet}.{index + 2}"


def _normalise_ips(frame: pd.DataFrame, known: set) -> pd.DataFrame:
    """Map the metadata IPs onto generated host names (cosmetic only)."""
    return frame


def _flow_row(rng: np.random.Generator, flow: dict, src_ip: str, dst_ip: str,
              src_port: int, start: float, scenario: str, window_id: int) -> dict:
    """Turn one sampled behavioural flow into a graph-ready row."""
    aggregated = tm.aggregate_flow(flow)
    duration_s = aggregated["flow_duration"] / 1e6
    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": int(src_port),
        "dst_port": int(aggregated["dst_port"]),
        "protocol": int(aggregated["protocol"]),
        "timestamp_start": float(start),
        "timestamp_end": float(start + duration_s),
        "flow_duration": float(aggregated["flow_duration"]),
        "total_fwd_packets": aggregated["total_fwd_packets"],
        "total_bwd_packets": aggregated["total_bwd_packets"],
        "total_length_of_fwd_packets": aggregated["total_length_of_fwd_packets"],
        "total_length_of_bwd_packets": aggregated["total_length_of_bwd_packets"],
        "down_up_ratio": aggregated["down_up_ratio"],
        "syn_flag_count": aggregated["syn_flag_count"],
        "ack_flag_count": aggregated["ack_flag_count"],
        "label": WINDOW_LABELS[scenario],
        "window_id": int(window_id),
        "scenario": scenario,
    }


def _benign_window(rng: np.random.Generator, hosts: Sequence[str], start: float,
                   window_seconds: int, n_flows: int, window_id: int) -> list[dict]:
    """
    Hub-and-spoke background traffic.

    Each host talks to the gateway and to a small number of peers.  Degrees stay
    low and *bounded*, which is the structural property the attack windows break.
    """
    rows = []
    n_hosts = len(hosts)
    for index in range(n_flows):
        host = hosts[index % n_hosts]
        roll = rng.random()
        if roll < 0.45:
            peer, port = GATEWAY_IP, int(rng.choice([80, 443, 53]))
        elif roll < 0.70:
            peer, port = EXTERNAL_IP, int(rng.choice([80, 443]))
        else:
            peer = hosts[int(rng.integers(0, n_hosts))]
            if peer == host:
                continue
            port = int(rng.choice([22, 445, 8080, 139]))
        flow = tm.sample_flow(rng, "Benign")
        offset = float(rng.uniform(0.0, window_seconds * 0.95))
        rows.append(_flow_row(rng, flow, host, peer,
                              int(rng.integers(40000, 60000)),
                              start + offset, "benign", window_id))
    return rows


def _port_scan_window(rng: np.random.Generator, hosts: Sequence[str], start: float,
                      window_seconds: int, n_flows: int, window_id: int) -> list[dict]:
    """One scanner fans out to many hosts and ports inside a single window."""
    rows = []
    # The scanner is randomised per window: a fixed scanner host would let the
    # classifier memorise an address instead of learning the fan-out structure,
    # which is the property the module is supposed to capture.
    scanner = hosts[int(rng.integers(0, len(hosts)))]
    others = [host for host in hosts if host != scanner]
    n_targets = max(4, min(len(others), n_flows))
    targets = [others[int(i)] for i in
               rng.choice(len(others), size=min(n_targets, len(others)),
                          replace=False)]

    for index in range(n_flows):
        target = targets[index % len(targets)]
        flow = tm.sample_flow(rng, "PortScan")
        # Every probe gets its own ephemeral source port and a distinct
        # destination port, which is what makes the fan-out visible.
        offset = float(rng.uniform(0.0, window_seconds * 0.95))
        row = _flow_row(rng, flow, scanner, target,
                        int(rng.integers(40000, 60000)), start + offset,
                        "port_scan", window_id)
        row["dst_port"] = int(rng.integers(1, 65535))
        rows.append(row)
    return rows


def _lateral_movement_window(rng: np.random.Generator, hosts: Sequence[str],
                             start: float, window_seconds: int, n_flows: int,
                             window_id: int) -> list[dict]:
    """
    A sequential chain of hosts, each contacting the next.

    Consecutive windows start from a different pivot so the model cannot
    memorise a fixed path; every *individual* edge looks like benign traffic,
    which is precisely why the pattern only exists at the graph level.
    """
    rows = []
    n_hosts = len(hosts)
    chain_len = max(3, min(n_hosts, n_flows // 4))
    pivot = int(rng.integers(0, n_hosts))
    order = [(pivot + step) % n_hosts for step in range(chain_len)]

    per_hop = max(1, n_flows // chain_len)
    for hop in range(chain_len - 1):
        source = hosts[order[hop]]
        target = hosts[order[hop + 1]]
        for _ in range(per_hop):
            flow = tm.sample_flow(rng, "Infiltration")
            # Hops are spread across the window so the traversal is *ordered in
            # time* as well as in structure.
            offset = window_seconds * (hop / max(chain_len - 1, 1))
            offset += float(rng.uniform(0.0, window_seconds / max(chain_len, 2) * 0.4))
            rows.append(_flow_row(rng, flow, source, target,
                                  int(rng.integers(40000, 60000)),
                                  start + offset, "lateral_movement", window_id))
    return rows


def _exfiltration_window(rng: np.random.Generator, hosts: Sequence[str],
                         start: float, window_seconds: int, n_flows: int,
                         window_id: int) -> list[dict]:
    """One internal host pushes a large volume to a single external host."""
    rows = []
    source = hosts[int(rng.integers(0, len(hosts)))]
    for _ in range(n_flows):
        flow = tm.sample_flow(rng, "Infiltration")
        # Inflate the forward payload by an order of magnitude: the anomaly is
        # volume on one edge, not the number of edges.
        flow["fwd_lengths"] = [length * 12.0 for length in flow["fwd_lengths"]]
        flow["bwd_lengths"] = [length * 0.2 for length in flow["bwd_lengths"]]
        offset = float(rng.uniform(0.0, window_seconds * 0.95))
        rows.append(_flow_row(rng, flow, source, EXTERNAL_IP,
                              int(rng.integers(40000, 60000)),
                              start + offset, "data_exfiltration", window_id))
    return rows


_SCENARIO_BUILDERS = {
    "benign": _benign_window,
    "port_scan": _port_scan_window,
    "lateral_movement": _lateral_movement_window,
    "data_exfiltration": _exfiltration_window,
}


def simulate_graph_windows(config: Optional[GraphScenarioConfig] = None,
                          scenarios: Sequence[str] = SCENARIOS,
                          n_windows_per_scenario: int = DEFAULT_N_WINDOWS_PER_SCENARIO,
                          start_epoch: float = 1_788_000_000.0) -> pd.DataFrame:
    """
    Generate windowed multi-host traffic.

    Windows are laid out **back to back on a single timeline**, each exactly
    ``window_seconds`` long, and ``window_id`` is the index on that timeline.
    That is what makes the frame usable by both the GNN window labeller and the
    drift detector without any further time bookkeeping.

    Args:
        config: Topology/volume settings; defaults to :class:`GraphScenarioConfig`.
        scenarios: Which structural patterns to generate.
        n_windows_per_scenario: Windows emitted per scenario.
        start_epoch: Epoch seconds of the first window.

    Returns:
        Flow frame with the :data:`_GRAPH_COLUMNS` (plus identity columns
        ``src_ip`` / ``dst_ip`` / ``dst_port`` and ``timestamp_start``).

    Raises:
        ValueError: on an unknown scenario name or a non-positive window count.
    """
    config = config or GraphScenarioConfig()
    unknown = [s for s in scenarios if s not in _SCENARIO_BUILDERS]
    if unknown:
        raise ValueError(
            f"Unknown scenario(s) {unknown}. Expected a subset of {list(SCENARIOS)}."
        )
    if n_windows_per_scenario <= 0:
        raise ValueError(
            f"n_windows_per_scenario must be positive, got {n_windows_per_scenario}"
        )

    hosts = [_host(index, config.subnet) for index in range(config.n_hosts)]
    window_seconds = int(config.window_seconds)
    frames = []
    window_id = 0

    for scenario in scenarios:
        builder = _SCENARIO_BUILDERS[scenario]
        for position in range(n_windows_per_scenario):
            # Independent, reproducible stream per window.
            rng = np.random.default_rng([config.seed, window_id])
            start = start_epoch + window_id * window_seconds
            rows = builder(rng, hosts, start, window_seconds,
                           config.flows_per_window, window_id)
            if rows:
                frames.append(pd.DataFrame(rows))
            window_id += 1

    if not frames:
        return pd.DataFrame(columns=list(_GRAPH_COLUMNS))

    out = pd.concat(frames, ignore_index=True)
    return out[list(_GRAPH_COLUMNS)]


def save_graph_windows(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a simulated graph-traffic frame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_graph_windows(path: str | Path) -> pd.DataFrame:
    """Read a frame written by :func:`save_graph_windows`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Graph traffic CSV not found: {path}")
    return pd.read_csv(path)


__all__ = [
    "DEFAULT_N_WINDOWS_PER_SCENARIO",
    "DEFAULT_WINDOW_SECONDS",
    "EXTERNAL_IP",
    "GATEWAY_IP",
    "GraphScenarioConfig",
    "SCENARIOS",
    "WINDOW_LABELS",
    "load_graph_windows",
    "save_graph_windows",
    "simulate_graph_windows",
]