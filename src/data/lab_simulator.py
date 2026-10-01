"""
src/data/lab_simulator.py — Home-lab capture simulator and PCAP writer (Phase 2,
section 2.1).

A real home-lab testbed (VirtualBox, Kali attacker VM, victim VMs, Zeek on the
host bridge) is not available in every environment, but the Phase 2 pipeline
still has to be exercised against genuine capture artefacts rather than against
in-memory arrays.  This module therefore produces **real** artefacts:

* standards-compliant libpcap ``.pcap`` files (Ethernet / IPv4 / TCP / UDP),
  parsed back by :func:`src.data.feature_extractor.pcap_to_flows`;
* Zeek ``conn.log`` files in JSON-lines form, parsed back by
  :func:`src.data.feature_extractor.zeek_conn_log_to_flows`.

Packets are drawn from :mod:`src.data.traffic_model`, the same behavioural model
that backs the CIC-IDS2017 and UNSW-NB15 stand-ins, so the home-lab domain is
behaviourally consistent with the public datasets while remaining layout- and
fingerprint-different.

.. warning::
   Output is **simulated traffic**, not traffic from a physical lab.  Every
   generated directory receives a ``PROVENANCE.md`` marker recording exactly
   that, and the experiment runner copies the marker into its reports.  Replace
   the contents of ``data/lab_captures`` with real captures to make any claim
   about self-captured traffic.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.data import traffic_model as tm
from src.data.feature_extractor import (
    extract_lab_features,
    load_session_labels,
    write_zeek_conn_log,
)

_ETH_TYPE_IPV4 = 0x0800

PROVENANCE_MARKER = "PROVENANCE.md"
PROVENANCE_TEXT = """# Provenance of this directory

**These captures were SIMULATED, not recorded from a physical lab.**

They were produced by `src/data/lab_simulator.py` from the behavioural model in
`src/data/traffic_model.py`, using the time windows documented in
`lab/capture_logs/session_labels.csv`.

The PCAP files are valid libpcap files and the `conn.log` files are valid Zeek
JSON-lines logs, so the extraction code path is exercised for real. The *packets
themselves* are synthetic.

To replace them with real captures:

1. Run the playbook in `lab/attack_scripts/playbook.md` against your lab.
2. Capture with `zeek -i <iface> LogAscii::use_json=T` (or save a pcap with
   `tcpdump -i <iface> -w capture.pcap`).
3. Drop the resulting `*.pcap` / `conn.log` files in this directory.
4. Re-run `python scripts/generate_lab_capture.py --extract-only`.
5. Delete this file.
"""


# ---------------------------------------------------------------------------
# PCAP writing
# ---------------------------------------------------------------------------
def _checksum(data: bytes) -> int:
    """Standard Internet checksum (RFC 1071)."""
    if len(data) % 2:
        data += b"\x00"
    total = 0
    for index in range(0, len(data), 2):
        total += (data[index] << 8) | data[index + 1]
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _ipv4_to_bytes(src_ip: str, dst_ip: str, protocol: int, payload_len: int,
                   ident: int) -> bytes:
    total_length = 20 + payload_len
    header = struct.pack(
        ">BBHHHBBH4s4s",
        0x45, 0x00, total_length & 0xFFFF, ident & 0xFFFF,
        0x4000, 64, protocol, 0,
        bytes(int(part) for part in src_ip.split(".")),
        bytes(int(part) for part in dst_ip.split(".")),
    )
    checksum = _checksum(header)
    return header[:10] + struct.pack(">H", checksum) + header[12:]


def _tcp_segment(src_port: int, dst_port: int, flags: int, payload: bytes,
                 seq: int) -> bytes:
    header = struct.pack(
        ">HHIIBBHHH",
        src_port, dst_port, seq & 0xFFFFFFFF, 0, 0x50, flags, 8192, 0, 0,
    )
    return header + payload


def _udp_datagram(src_port: int, dst_port: int, payload: bytes) -> bytes:
    length = 8 + len(payload)
    return struct.pack(">HHHH", src_port, dst_port, length, 0) + payload


def _ethernet_frame(src_mac: bytes, dst_mac: bytes, ip_packet: bytes) -> bytes:
    return dst_mac + src_mac + struct.pack(">H", _ETH_TYPE_IPV4) + ip_packet


def write_pcap(path: Path, packets: Iterable[tuple[float, bytes]]) -> Path:
    """
    Write ``(timestamp_seconds, frame_bytes)`` pairs to a libpcap file.

    Args:
        path: Output ``.pcap`` path (parent directories are created).
        packets: Iterable of ``(epoch_seconds, ethernet_frame)``.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as handle:
        # magic, ver_major, ver_minor, thiszone, sigfigs, snaplen, network=1(EN10MB)
        handle.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 262144, 1))

        for timestamp, frame in packets:
            seconds = int(timestamp)
            microseconds = int(round((timestamp - seconds) * 1_000_000))
            if microseconds >= 1_000_000:
                seconds += 1
                microseconds -= 1_000_000
            handle.write(struct.pack(
                "<IIII", seconds, microseconds, len(frame), len(frame),
            ))
            handle.write(frame)

    return path


def _mac_from_ip(ip: str) -> bytes:
    return bytes([0x02, 0x00]) + bytes(int(part) for part in ip.split("."))


# ---------------------------------------------------------------------------
# Session simulation
# ---------------------------------------------------------------------------
def _flow_to_packets(flow: dict, src_ip: str, dst_ip: str, src_port: int,
                     start_epoch: float, ident_base: int):
    """Turn one sampled flow into ``(timestamp, ethernet_frame)`` pairs."""
    protocol = flow["protocol"]
    dst_port = flow["dst_port"]
    start = float(flow["fwd_times"][0]) if flow["fwd_times"] else 0.0

    src_mac = _mac_from_ip(src_ip)
    dst_mac = _mac_from_ip(dst_ip)

    def _emit(offset_us: float, src: str, dst: str, sport: int, dport: int,
              flags: int, payload_len: int) -> tuple[float, bytes]:
        payload = bytes((index * 7 + 13) & 0xFF for index in range(payload_len))
        if protocol == 6:
            l4 = _tcp_segment(sport, dport, flags, payload, sport + dport)
        else:
            l4 = _udp_datagram(sport, dport, payload)
        ip = _ipv4_to_bytes(src, dst, protocol, len(l4), ident_base)
        return start_epoch + offset_us / 1_000_000.0, _ethernet_frame(src_mac, dst_mac, ip)

    packets = []
    for index, length in enumerate(flow["fwd_lengths"]):
        flags = flow["flags"][index] if index < len(flow["flags"]) else 0x10
        offset = flow["fwd_times"][index] if index < len(flow["fwd_times"]) else 0.0
        packets.append(_emit(offset, src_ip, dst_ip, src_port, dst_port, flags, int(length)))

    for index, length in enumerate(flow["bwd_lengths"]):
        offset = flow["bwd_times"][index] if index < len(flow["bwd_times"]) else 0.0
        flags = 0x10 if flow["flags"] else 0x04
        packets.append(_emit(offset, dst_ip, src_ip, dst_port, src_port, flags, int(length)))

    packets.sort(key=lambda item: item[0])
    return packets


def simulate_session_pcap(session: pd.Series, out_path: Path, flows: int,
                          rng: np.random.Generator,
                          src_port_base: int = 40000) -> dict:
    """
    Simulate one capture session and write its packets to a PCAP file.

    Flows are packed into the session's time window at ``flows_per_second``
    (derived from the window length) so that the resulting timestamps stay
    inside the window and can be labelled by
    :func:`src.data.feature_extractor.label_flows_by_session`.

    Returns:
        A dict of aggregate statistics for logging.
    """
    label = str(session["attack_type"])
    src_ip = str(session["source_ip"])
    dst_ip = str(session["target_ip"])
    window_start = session["timestamp_start"].timestamp()
    window_end = session["timestamp_end"].timestamp()
    window = max(window_end - window_start, 1.0)

    flows_per_second = max(1.0, flows / window)
    packets: list[tuple[float, bytes]] = []
    ident = int(rng.integers(1, 60_000))

    for _ in range(flows):
        flow = tm.sample_flow(rng, label)
        src_port = int(rng.integers(src_port_base, src_port_base + 20000))
        offset = float(rng.uniform(0.0, window * 0.98))
        ident += 1
        packets.extend(_flow_to_packets(
            flow, src_ip, dst_ip, src_port, window_start + offset, ident,
        ))

    write_pcap(out_path, packets)
    return {
        "session_id": session.get("session_id", ""),
        "attack_type": label,
        "flows": flows,
        "packets": len(packets),
        "path": str(out_path),
    }


def simulate_session_conn_log(session: pd.Series, out_path: Path, flows: int,
                              rng: np.random.Generator,
                              src_port_base: int = 40000) -> dict:
    """
    Simulate one capture session as a Zeek ``conn.log`` instead of a PCAP.

    Used to exercise :func:`zeek_conn_log_to_flows` when no pcap writer is
    wanted, and to verify both extraction paths agree.

    Returns:
        A dict of aggregate statistics for logging.
    """
    from src.data.feature_extractor import CANONICAL_FEATURES

    label = str(session["attack_type"])
    src_ip = str(session["source_ip"])
    dst_ip = str(session["target_ip"])
    window_start = session["timestamp_start"].timestamp()
    window_end = session["timestamp_end"].timestamp()
    window = max(window_end - window_start, 1.0)

    records = []
    for _ in range(flows):
        flow = tm.sample_flow(rng, label)
        aggregated = tm.aggregate_flow(flow)
        row = {name: aggregated[name] for name in CANONICAL_FEATURES}
        offset = float(rng.uniform(0.0, window * 0.98))
        row.update({
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": int(rng.integers(src_port_base, src_port_base + 20000)),
            "dst_port": aggregated["dst_port"],
            "protocol": aggregated["protocol"],
            "timestamp_start": window_start + offset,
            "timestamp_end": window_start + offset + aggregated["flow_duration"] / 1e6,
        })
        records.append(row)

    frame = pd.DataFrame.from_records(records)
    write_zeek_conn_log(frame, out_path)
    return {
        "session_id": session.get("session_id", ""),
        "attack_type": label,
        "flows": len(frame),
        "path": str(out_path),
    }


def generate_lab_capture(output_dir: Path, flows_per_session: int = 900,
                         seed: int = 42, session_labels: Optional[Path] = None,
                         mode: str = "pcap") -> dict:
    """
    Generate the full set of simulated home-lab capture artefacts.

    Args:
        output_dir: Target directory, normally ``data/lab_captures``.
        flows_per_session: Flows emitted for each row of the session-label CSV.
        seed: Base RNG seed; individual sessions derive independent streams.
        session_labels: Optional override for the session-label CSV path.
        mode: ``"pcap"`` to write capture files, ``"zeek"`` to write ``conn.log``
              files, or ``"both"``.

    Returns:
        Summary dict with per-session statistics and written paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sessions = load_session_labels(session_labels)

    summaries = []
    for position, (_, session) in enumerate(sessions.iterrows()):
        # Independent, reproducible stream per session.
        rng = np.random.default_rng([seed, position])
        day = pd.Timestamp(session["timestamp_start"]).strftime("%Y%m%d")
        session_id = str(session.get("session_id", f"sess_{position + 1:03d}"))
        stem = f"{day}_{session_id}_{session['attack_type']}"

        if mode in ("pcap", "both"):
            summaries.append(simulate_session_pcap(
                session, output_dir / f"{stem}.pcap", flows_per_session, rng,
            ))
        if mode in ("zeek", "both"):
            summaries.append(simulate_session_conn_log(
                session, output_dir / f"conn_{stem}.log", flows_per_session, rng,
            ))

    (output_dir / PROVENANCE_MARKER).write_text(PROVENANCE_TEXT, encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "sessions": len(sessions),
        "flows_per_session": flows_per_session,
        "mode": mode,
        "artifacts": summaries,
    }


def build_lab_dataset(output_dir: Path, flows_per_session: int = 900, seed: int = 42,
                      csv_name: str = "lab_flows.csv",
                      session_labels: Optional[Path] = None,
                      mode: str = "pcap") -> tuple[dict, pd.DataFrame]:
    """
    Generate the captures and immediately extract the labelled flow dataset.

    Returns:
        ``(generation_summary, lab_dataframe)``.
    """
    output_dir = Path(output_dir)
    summary = generate_lab_capture(
        output_dir, flows_per_session=flows_per_session, seed=seed,
        session_labels=session_labels, mode=mode,
    )
    frame = extract_lab_features(output_dir, save_to=output_dir / csv_name)
    summary["csv"] = str(output_dir / csv_name)
    summary["rows"] = int(len(frame))
    summary["labels"] = frame["label"].value_counts().to_dict()
    return summary, frame
