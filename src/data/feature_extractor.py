"""
src/data/feature_extractor.py — Cross-dataset feature alignment and lab-traffic
flow extraction (Phase 2, sections 2.2 and 2.3.4).

Three responsibilities:

1. **Schema alignment.**  Map whatever column names a dataset happens to use onto
   a single canonical, dataset-independent feature schema declared in
   ``configs/feature_schema.yaml``.  Features a dataset cannot supply become
   ``NaN`` and are reported through :func:`feature_coverage` instead of being
   silently invented.

2. **Flow extraction from lab captures.**  Convert raw capture artefacts into
   flow-level records carrying the canonical schema:

   * :func:`pcap_to_flows` — a self-contained libpcap reader (Ethernet / IPv4 /
     TCP / UDP).  Implemented with :mod:`struct` only so the pipeline has no
     scapy/dpkt dependency.
   * :func:`zeek_conn_log_to_flows` — Zeek ``conn.log`` parser (JSON lines or
     classic tab-separated headers).
   * :func:`extract_lab_features` — orchestration + time-window labelling from
     ``lab/capture_logs/session_labels.csv``.

3. **Integrity helpers.**  :func:`is_identity_feature` and
   :func:`assert_behavior_only` guarantee that a "domain-invariant" feature set
   never contains address-derived columns.

The canonical schema deliberately contains **no** IP, port or protocol fields:
that is the whole point of the Tier 2 claim.
"""
from __future__ import annotations

import json
import struct
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from src.utils.pandas_compat import apply_pandas_compat

apply_pandas_compat()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"

CANONICAL_FEATURES: tuple[str, ...] = (
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_length_of_fwd_packets",
    "total_length_of_bwd_packets",
    "total_length_of_packets",
    "fwd_packet_length_mean",
    "fwd_packet_length_max",
    "fwd_packet_length_min",
    "fwd_packet_length_std",
    "bwd_packet_length_mean",
    "bwd_packet_length_max",
    "bwd_packet_length_min",
    "bwd_packet_length_std",
    "packet_length_mean",
    "packet_length_std",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "down_up_ratio",
    "syn_flag_count",
    "ack_flag_count",
    "rst_flag_count",
    "fin_flag_count",
    "psh_flag_count",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
)

_TCP_FLAG_LETTERS = {
    "syn_flag_count": "S",
    "ack_flag_count": "A",
    "rst_flag_count": "R",
    "fin_flag_count": "F",
    "psh_flag_count": "P",
}

_PCAP_MAGIC_LITTLE_ENDIAN = 0xA1B2C3D4
_PCAP_MAGIC_BIG_ENDIAN = 0xD4C3B2A1
_PCAP_MAGIC_NS_LITTLE_ENDIAN = 0xA1B23C4D
_PCAP_MAGIC_NS_BIG_ENDIAN = 0x4D3CB2A1


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def load_feature_schema() -> dict:
    """Load ``configs/feature_schema.yaml``."""
    path = _CONFIGS_DIR / "feature_schema.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def behavior_features(mode: str = "behavior_core") -> list[str]:
    """
    Return the canonical feature list for a named behavior set.

    Args:
        mode: ``"behavior_core"`` (the 16-feature reference set) or
              ``"behavior_extended"``.

    Raises:
        ValueError: if ``mode`` is not a declared behavior set.
    """
    schema = load_feature_schema()
    if mode not in schema:
        raise ValueError(
            f"Unknown behavior feature set '{mode}'. "
            f"Available: {sorted(k for k in schema if k.startswith('behavior_'))}"
        )
    return list(schema[mode])


def identity_patterns() -> list[str]:
    """Column-name fragments that disqualify a feature from a behavior set."""
    return list(load_feature_schema()["identity_patterns"])


def is_identity_feature(name: str, patterns: Optional[Iterable[str]] = None) -> bool:
    """
    Return True when ``name`` looks address/transport derived.

    Substring matching is intentional and conservative: ``source_ip``,
    ``dstport`` and ``Protocol`` are all rejected, and so is anything whose name
    merely contains one of the fragments.
    """
    if patterns is None:
        patterns = identity_patterns()
    lowered = str(name).strip().lower()
    return any(fragment in lowered for fragment in patterns)


def assert_behavior_only(features: Iterable[str], patterns: Optional[Iterable[str]] = None) -> None:
    """
    Raise ``ValueError`` if any feature name carries an identity fragment.

    Used by the Phase 2 tests and by :class:`src.data.preprocessor.Preprocessor`
    whenever it builds a behavior feature set.
    """
    if patterns is None:
        patterns = identity_patterns()
    offenders = [
        name for name in features
        if any(fragment in str(name).lower() for fragment in patterns)
    ]
    if offenders:
        raise ValueError(
            "Behavior feature set contains identity-derived features: "
            f"{sorted(offenders)}"
        )


# ---------------------------------------------------------------------------
# Column resolution + alignment
# ---------------------------------------------------------------------------
def _column_map(dataset: str) -> dict:
    maps = load_feature_schema().get("column_maps", {})
    if dataset not in maps:
        raise ValueError(
            f"No canonical column map declared for dataset '{dataset}'. "
            f"Declared datasets: {sorted(maps)}"
        )
    return maps[dataset]


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a column to numeric, mapping unparseable values to NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(np.float64)
    cleaned = series.astype("string").str.strip().str.replace(",", "", regex=False)
    cleaned = cleaned.replace({"": None, "-": None, "None": None, "nan": None})
    return pd.to_numeric(cleaned, errors="coerce").astype(np.float64)


def _resolve(df: pd.DataFrame, columns: Iterable[str]) -> Optional[str]:
    """Return the first column in ``columns`` actually present in ``df``."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in columns:
        hit = lookup.get(str(candidate).strip().lower())
        if hit is not None:
            return hit
    return None


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division yielding 0.0 instead of inf/NaN on a zero divisor."""
    num = numerator.to_numpy(dtype=np.float64)
    den = denominator.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    out[~np.isfinite(out)] = 0.0
    return pd.Series(out, index=numerator.index)


def extract_canonical_feature(df: pd.DataFrame, dataset: str, feature: str) -> pd.Series:
    """
    Materialise one canonical feature from ``df``.

    Returns an all-``NaN`` float Series when the dataset does not provide the
    feature, so that missing coverage stays observable.
    """
    spec = _column_map(dataset).get(feature)
    if spec is None:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)

    kind = spec.get("kind", "direct")
    columns = spec.get("columns", [])

    if kind == "direct":
        name = _resolve(df, columns)
        if name is None:
            return pd.Series(np.nan, index=df.index, dtype=np.float64)
        return _to_numeric(df[name])

    if kind == "count_letters":
        name = _resolve(df, columns)
        if name is None:
            return pd.Series(np.nan, index=df.index, dtype=np.float64)
        letters = [str(ch) for ch in spec.get("letters", [])]
        text = df[name].astype("string").str.upper()
        total = pd.Series(0.0, index=df.index, dtype=np.float64)
        for letter in letters:
            total = total + text.str.count(letter, regex=False).astype(np.float64)
        return total

    if kind == "ratio":
        resolved = [_resolve(df, [c]) for c in columns]
        if len(resolved) != 2 or any(r is None for r in resolved):
            return pd.Series(np.nan, index=df.index, dtype=np.float64)
        return _safe_divide(_to_numeric(df[resolved[0]]), _to_numeric(df[resolved[1]]))

    if kind == "per_second":
        resolved = [_resolve(df, [c]) for c in columns]
        if len(resolved) != 2 or any(r is None for r in resolved):
            return pd.Series(np.nan, index=df.index, dtype=np.float64)
        first = _to_numeric(df[resolved[0]])
        second = _to_numeric(df[resolved[1]])
        total = first + second if spec.get("total", False) else first
        seconds = _to_numeric(df["flow_duration"]) / 1_000_000.0
        return _safe_divide(total, seconds)

    raise ValueError(f"Unknown column-map kind '{kind}' for feature '{feature}'")


def align_to_canonical(df: pd.DataFrame, dataset: str,
                       features: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Project ``df`` onto the canonical schema.

    Args:
        df: Raw dataset frame (as returned by :func:`src.data.loader.load_dataset`
            or by the lab extractor).
        dataset: Registry key used to look up the column map.
        features: Canonical features to emit.  Defaults to every canonical
                  feature the column map declares.

    Returns:
        DataFrame with exactly ``features`` columns, all float64.  Unsupported
        features are ``NaN``.
    """
    requested = list(features) if features is not None else sorted(_column_map(dataset))
    canonical = {name: extract_canonical_feature(df, dataset, name) for name in requested}
    out = pd.DataFrame(canonical, index=df.index, dtype=np.float64)
    return out


def feature_coverage(df: pd.DataFrame, dataset: str,
                     features: Optional[Iterable[str]] = None) -> dict:
    """
    Report which canonical features ``df`` can actually supply.

    Returns a dict with the per-feature boolean map plus
    ``coverage`` (fraction available) and ``missing``.
    """
    if features is None:
        features = sorted(_column_map(dataset))
    available = {
        name: bool(extract_canonical_feature(df, dataset, name).notna().any())
        for name in features
    }
    covered = sum(available.values())
    return {
        "available": available,
        "coverage": covered / len(features) if features else 0.0,
        "covered": covered,
        "total": len(features),
        "missing": sorted(name for name, ok in available.items() if not ok),
    }


# ---------------------------------------------------------------------------
# PCAP parsing (self-contained, no scapy/dpkt required)
# ---------------------------------------------------------------------------
def iter_pcap_packets(path: Path) -> Iterable[tuple[float, bytes]]:
    """
    Yield ``(timestamp_seconds, link_layer_frame)`` from a libpcap file.

    Supports the classic microsecond-resolution magic numbers in both byte
    orders plus their nanosecond-resolution variants, and skips
    LINKTYPE_NULL (0) / DLT_RAW (12) / DLT_NULL variants by prepending an
    artificial Ethernet header so downstream parsing stays uniform.
    """
    with open(path, "rb") as handle:
        header = handle.read(24)
        if len(header) < 24:
            raise ValueError(f"{path}: truncated pcap global header")

        magic = struct.unpack("<I", header[:4])[0]
        if magic == _PCAP_MAGIC_LITTLE_ENDIAN:
            endian, nano = "<", False
        elif magic == _PCAP_MAGIC_BIG_ENDIAN:
            endian, nano = ">", False
        elif magic == _PCAP_MAGIC_NS_LITTLE_ENDIAN:
            endian, nano = "<", True
        elif magic == _PCAP_MAGIC_NS_BIG_ENDIAN:
            endian, nano = ">", True
        else:
            raise ValueError(
                f"{path}: not a libpcap file (magic=0x{magic:08X}). "
                "pcapng files are not supported; convert with "
                "`editcap -F pcap in.pcapng out.pcap`."
            )

        _, _, _, _, _, linktype = struct.unpack(endian + "IHHiIII", header)
        if linktype not in (1, 0, 12, 101):
            raise ValueError(
                f"{path}: unsupported link type {linktype}; only Ethernet (1), "
                "raw IP (101) and null/loopback (0, 12) captures are handled."
            )

        rec_header = struct.Struct(endian + "IIII")
        min_eth = 14
        while True:
            raw = handle.read(rec_header.size)
            if len(raw) < rec_header.size:
                break
            ts_sec, ts_frac, incl_len, _orig_len = rec_header.unpack(raw)
            frame = handle.read(incl_len)
            if len(frame) < incl_len:
                break

            if linktype != 1:
                frame = (b"\x00" * 6 + b"\x00" * 6 + struct.pack(">H", 0x0800)) + frame

            if len(frame) < min_eth + 20:
                continue
            if struct.unpack(">H", frame[12:14])[0] != 0x0800:
                continue

            timestamp = ts_sec + (ts_frac / 1e9 if nano else ts_frac / 1e6)
            yield timestamp, frame


def _parse_ipv4_tcp_udp(frame: bytes) -> Optional[dict]:
    """Parse an Ethernet/IPv4 frame into the fields needed for flow features."""
    ip_offset = 14
    version_ihl = frame[ip_offset]
    if (version_ihl >> 4) != 4:
        return None
    ihl = (version_ihl & 0x0F) * 4
    if ihl < 20 or len(frame) < ip_offset + ihl:
        return None

    total_length = struct.unpack(">H", frame[ip_offset + 2:ip_offset + 4])[0]
    fragment_field = struct.unpack(">H", frame[ip_offset + 6:ip_offset + 8])[0]
    if fragment_field & 0x1FFF:
        return None

    protocol = frame[ip_offset + 9]
    src_ip = ".".join(str(b) for b in frame[ip_offset + 12:ip_offset + 16])
    dst_ip = ".".join(str(b) for b in frame[ip_offset + 16:ip_offset + 20])

    l4_offset = ip_offset + ihl
    if protocol == 6:
        if len(frame) < l4_offset + 20:
            return None
        src_port, dst_port = struct.unpack(">HH", frame[l4_offset:l4_offset + 4])
        flags = frame[l4_offset + 13]
        l4_len = 20
    elif protocol == 17:
        if len(frame) < l4_offset + 8:
            return None
        src_port, dst_port = struct.unpack(">HH", frame[l4_offset:l4_offset + 4])
        flags = 0
        l4_len = 8
    else:
        return None

    payload_len = max(0, total_length - ihl - l4_len)
    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": int(src_port),
        "dst_port": int(dst_port),
        "protocol": protocol,
        "flags": flags,
        "length": int(payload_len),
    }


def _flow_key(packet: dict) -> tuple:
    """Direction-normalised 5-tuple so both directions share one flow record."""
    a = (packet["src_ip"], packet["src_port"])
    b = (packet["dst_ip"], packet["dst_port"])
    first = a if a <= b else b
    return (first[0], first[1], b[0], b[1], packet["protocol"])


def pcap_to_flows(paths: Iterable[Path], src_port: Optional[int] = None,
                  dst_port: Optional[int] = None,
                  src_ip: Optional[str] = None,
                  dst_ip: Optional[str] = None) -> pd.DataFrame:
    """
    Convert PCAP capture(s) into canonical flow-level features.

    Each flow is aggregated per direction, and the resulting frame carries the
    canonical schema plus the TCP/UDP flag counts.

    Args:
        paths: One path or an iterable of ``.pcap`` paths.
        src_port / dst_port / src_ip / dst_ip: Optional filters, applied to the
            *initiating* direction of each flow (i.e. the first packet seen).

    Returns:
        DataFrame with the canonical feature columns and
        ``src_ip`` / ``dst_ip`` / ``src_port`` / ``dst_port`` / ``protocol``
        / ``timestamp_start`` / ``timestamp_end`` for auditing and for
        session-window labelling.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    flows: "OrderedDict[tuple, dict]" = OrderedDict()

    for path in paths:
        path = Path(path)
        for timestamp, frame in iter_pcap_packets(path):
            packet = _parse_ipv4_tcp_udp(frame)
            if packet is None:
                continue

            key = _flow_key(packet)
            flow = flows.get(key)
            if flow is None:
                forward = (packet["src_ip"], packet["src_port"]) <= (
                    packet["dst_ip"], packet["dst_port"]
                )
                initiator = packet if forward else {
                    "src_ip": packet["dst_ip"], "src_port": packet["dst_port"],
                    "dst_ip": packet["src_ip"], "dst_port": packet["src_port"],
                }
                if src_port is not None and initiator["src_port"] != src_port:
                    continue
                if dst_port is not None and initiator["dst_port"] != dst_port:
                    continue
                if src_ip is not None and initiator["src_ip"] != src_ip:
                    continue
                if dst_ip is not None and initiator["dst_ip"] != dst_ip:
                    continue
                flow = {
                    "src_ip": initiator["src_ip"],
                    "src_port": int(initiator["src_port"]),
                    "dst_ip": initiator["dst_ip"],
                    "dst_port": int(initiator["dst_port"]),
                    "protocol": initiator["protocol"],
                    "fwd_times": [],
                    "fwd_lengths": [],
                    "bwd_times": [],
                    "bwd_lengths": [],
                    "syn": 0,
                    "ack": 0,
                    "rst": 0,
                    "fin": 0,
                    "psh": 0,
                }
                flows[key] = flow

            forward = (packet["src_ip"], packet["src_port"]) <= (
                packet["dst_ip"], packet["dst_port"]
            )
            if forward:
                flow["fwd_times"].append(timestamp)
                flow["fwd_lengths"].append(packet["length"])
            else:
                flow["bwd_times"].append(timestamp)
                flow["bwd_lengths"].append(packet["length"])

            flags = packet["flags"]
            flow["syn"] += (flags >> 1) & 1
            flow["ack"] += (flags >> 4) & 1
            flow["rst"] += (flags >> 2) & 1
            flow["fin"] += (flags >> 0) & 1
            flow["psh"] += (flags >> 3) & 1

    if not flows:
        return pd.DataFrame(columns=list(CANONICAL_FEATURES) + [
            "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
            "timestamp_start", "timestamp_end",
        ])

    records = [_flow_to_record(flow) for flow in flows.values()]
    df = pd.DataFrame.from_records(records)

    ordered = [c for c in CANONICAL_FEATURES if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    return df[ordered + extra]


def _safe_stat(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64)))


def _flow_to_record(flow: dict) -> dict:
    """Aggregate one flow's packet list into canonical features."""
    fwd_times = flow["fwd_times"]
    bwd_times = flow["bwd_times"]
    fwd_lengths = flow["fwd_lengths"]
    bwd_lengths = flow["bwd_lengths"]
    all_times = sorted(fwd_times + bwd_times)
    all_lengths = fwd_lengths + bwd_lengths

    start = all_times[0] if all_times else 0.0
    end = all_times[-1] if all_times else 0.0
    duration_us = max(0.0, (end - start) * 1_000_000.0)

    fwd_bytes = float(sum(fwd_lengths))
    bwd_bytes = float(sum(bwd_lengths))
    total_packets = len(all_times)

    def _iat_std(times: list[float]) -> float:
        if len(times) < 3:
            return 0.0
        ordered = sorted(times)
        deltas = np.diff(np.asarray(ordered, dtype=np.float64))
        return float(np.std(deltas) * 1_000_000.0)

    flow_iat = _iat_std(all_times)
    fwd_iat = _iat_std(fwd_times)
    bwd_iat = _iat_std(bwd_times)

    return {
        "flow_duration": duration_us,
        "total_fwd_packets": float(len(fwd_times)),
        "total_bwd_packets": float(len(bwd_times)),
        "total_length_of_fwd_packets": fwd_bytes,
        "total_length_of_bwd_packets": bwd_bytes,
        "total_length_of_packets": fwd_bytes + bwd_bytes,
        "fwd_packet_length_mean": float(np.mean(fwd_lengths)) if fwd_lengths else 0.0,
        "fwd_packet_length_max": float(max(fwd_lengths)) if fwd_lengths else 0.0,
        "fwd_packet_length_min": float(min(fwd_lengths)) if fwd_lengths else 0.0,
        "fwd_packet_length_std": _safe_stat(fwd_lengths),
        "bwd_packet_length_mean": float(np.mean(bwd_lengths)) if bwd_lengths else 0.0,
        "bwd_packet_length_max": float(max(bwd_lengths)) if bwd_lengths else 0.0,
        "bwd_packet_length_min": float(min(bwd_lengths)) if bwd_lengths else 0.0,
        "bwd_packet_length_std": _safe_stat(bwd_lengths),
        "packet_length_mean": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "packet_length_std": _safe_stat(all_lengths),
        "flow_iat_mean": flow_iat,
        "flow_iat_std": flow_iat,
        "fwd_iat_mean": fwd_iat,
        "bwd_iat_mean": bwd_iat,
        "down_up_ratio": (bwd_bytes / fwd_bytes) if fwd_bytes else 0.0,
        "syn_flag_count": float(flow["syn"]),
        "ack_flag_count": float(flow["ack"]),
        "rst_flag_count": float(flow["rst"]),
        "fin_flag_count": float(flow["fin"]),
        "psh_flag_count": float(flow["psh"]),
        "flow_bytes_per_sec": ((fwd_bytes + bwd_bytes) / (duration_us / 1e6)) if duration_us else 0.0,
        "flow_packets_per_sec": (total_packets / (duration_us / 1e6)) if duration_us else 0.0,
        "src_ip": flow["src_ip"],
        "dst_ip": flow["dst_ip"],
        "src_port": flow["src_port"],
        "dst_port": flow["dst_port"],
        "protocol": flow["protocol"],
        "timestamp_start": start,
        "timestamp_end": end,
    }


# ---------------------------------------------------------------------------
# Zeek conn.log
# ---------------------------------------------------------------------------
_ZEEK_EMPTY = {"-", "(empty)", ""}


def _zeek_value(raw: str):
    """Normalise one Zeek field value (Zeek uses '-' and '(empty)' for null)."""
    raw = raw.strip()
    if raw in _ZEEK_EMPTY:
        return None
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1]
        if not inner:
            return None
        return [part.strip().strip('"') for part in inner.split(",")]
    return raw.strip('"')


def _zeek_number(raw: str) -> float:
    value = _zeek_value(raw)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _zeek_timestamp(raw: str) -> float:
    value = _zeek_value(raw)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("nan")


def zeek_conn_log_to_flows(paths: Iterable[Path]) -> pd.DataFrame:
    """
    Parse Zeek ``conn.log`` files into canonical flow features.

    Accepts both the JSON-lines writer (``LogAscii::use_json=T``) and the
    classic tab-separated format with a ``#fields`` header.

    Args:
        paths: One path or an iterable of ``conn.log`` paths.

    Returns:
        DataFrame with the canonical schema plus endpoint/timestamp columns.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    records = []
    for path in paths:
        path = Path(path)
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            fields: Optional[list[str]] = None
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("#"):
                    if line.startswith("#fields"):
                        fields = line.split("\t")[1:]
                    continue
                if fields is None:
                    raise ValueError(
                        f"{path}: TSV conn.log encountered before a '#fields' header. "
                        "Write Zeek logs with LogAscii::use_json=T."
                    )
                values = line.split("\t")
                values += ["-"] * (len(fields) - len(values))
                records.append(dict(zip(fields, values[:len(fields)])))

    if not records:
        return pd.DataFrame(columns=list(CANONICAL_FEATURES))

    raw = pd.DataFrame.from_records(records)

    start = raw["ts"].map(_zeek_timestamp) if "ts" in raw else pd.Series(np.nan, index=raw.index)
    duration = raw["duration"].map(_zeek_number) if "duration" in raw else pd.Series(np.nan, index=raw.index)
    orig_bytes = raw["orig_bytes"].map(_zeek_number) if "orig_bytes" in raw else pd.Series(np.nan, index=raw.index)
    resp_bytes = raw["resp_bytes"].map(_zeek_number) if "resp_bytes" in raw else pd.Series(np.nan, index=raw.index)
    orig_pkts = raw["orig_pkts"].map(_zeek_number) if "orig_pkts" in raw else pd.Series(np.nan, index=raw.index)
    resp_pkts = raw["resp_pkts"].map(_zeek_number) if "resp_pkts" in raw else pd.Series(np.nan, index=raw.index)

    def _flag_count(column: str, letter: str) -> pd.Series:
        if column not in raw:
            return pd.Series(0.0, index=raw.index, dtype=np.float64)
        counts = []
        for value in raw[column].map(_zeek_value):
            if isinstance(value, list):
                joined = "".join(str(v) for v in value)
            elif value is None:
                joined = ""
            else:
                joined = str(value)
            counts.append(float(joined.upper().count(letter)))
        return pd.Series(counts, index=raw.index, dtype=np.float64)

    total_bytes = (orig_bytes.fillna(0) + resp_bytes.fillna(0)).astype(np.float64)
    total_pkts = (orig_pkts.fillna(0) + resp_pkts.fillna(0)).astype(np.float64)
    seconds = duration.fillna(0) / 1_000_000.0
    orig_bytes = orig_bytes.fillna(0).astype(np.float64)
    resp_bytes = resp_bytes.fillna(0).astype(np.float64)
    orig_pkts = orig_pkts.fillna(0).astype(np.float64)
    resp_pkts = resp_pkts.fillna(0).astype(np.float64)
    duration = duration.fillna(0).astype(np.float64)

    out = pd.DataFrame({
        "flow_duration": duration,
        "total_fwd_packets": orig_pkts,
        "total_bwd_packets": resp_pkts,
        "total_length_of_fwd_packets": orig_bytes,
        "total_length_of_bwd_packets": resp_bytes,
        "total_length_of_packets": total_bytes,
        "fwd_packet_length_mean": _safe_divide(orig_bytes, orig_pkts),
        "fwd_packet_length_max": _zeek_vector_max(raw, "orig_len"),
        "fwd_packet_length_min": _zeek_vector_min(raw, "orig_len"),
        "fwd_packet_length_std": np.nan,
        "bwd_packet_length_mean": _safe_divide(resp_bytes, resp_pkts),
        "bwd_packet_length_max": _zeek_vector_max(raw, "resp_len"),
        "bwd_packet_length_min": _zeek_vector_min(raw, "resp_len"),
        "bwd_packet_length_std": np.nan,
        "packet_length_mean": _safe_divide(total_bytes, total_pkts),
        "packet_length_std": np.nan,
        "flow_iat_mean": np.nan,
        "flow_iat_std": np.nan,
        "fwd_iat_mean": np.nan,
        "bwd_iat_mean": np.nan,
        "down_up_ratio": _safe_divide(resp_bytes, orig_bytes),
        "syn_flag_count": _flag_count("orig_flags", "S"),
        "ack_flag_count": _flag_count("orig_flags", "A"),
        "rst_flag_count": _flag_count("orig_flags", "R"),
        "fin_flag_count": _flag_count("orig_flags", "F"),
        "psh_flag_count": _flag_count("orig_flags", "P"),
        "flow_bytes_per_sec": _safe_divide(total_bytes, seconds),
        "flow_packets_per_sec": _safe_divide(total_pkts, seconds),
    })

    for target, column in (
        ("src_ip", "id.orig_h"), ("dst_ip", "id.resp_h"),
        ("src_port", "id.orig_p"), ("dst_port", "id.resp_p"),
        ("protocol", "proto"),
    ):
        out[target] = raw[column].map(_zeek_value) if column in raw else None

    out["timestamp_start"] = start
    out["timestamp_end"] = start + seconds

    ordered = [c for c in CANONICAL_FEATURES if c in out.columns]
    extra = [c for c in out.columns if c not in ordered]
    return out[ordered + extra]


def _zeek_vector_max(raw: pd.DataFrame, column: str) -> pd.Series:
    """Maximum value of a Zeek vector column (e.g. ``orig_len``)."""
    return _zeek_vector_agg(raw, column, np.max)


def _zeek_vector_min(raw: pd.DataFrame, column: str) -> pd.Series:
    """Minimum value of a Zeek vector column (e.g. ``orig_len``)."""
    return _zeek_vector_agg(raw, column, np.min)


def _zeek_vector_agg(raw: pd.DataFrame, column: str, reducer) -> pd.Series:
    if column not in raw:
        return pd.Series(np.nan, index=raw.index, dtype=np.float64)
    out = []
    for value in raw[column].map(_zeek_value):
        if isinstance(value, list):
            numbers = []
            for item in value:
                try:
                    numbers.append(float(item))
                except (TypeError, ValueError):
                    continue
            out.append(float(reducer(numbers)) if numbers else np.nan)
        else:
            out.append(np.nan)
    return pd.Series(out, index=raw.index, dtype=np.float64)


# ---------------------------------------------------------------------------
# Session-window labelling
# ---------------------------------------------------------------------------
SESSION_LABELS_PATH = _PROJECT_ROOT / "lab" / "capture_logs" / "session_labels.csv"


def load_session_labels(path: Optional[Path] = None) -> pd.DataFrame:
    """Load ``lab/capture_logs/session_labels.csv`` and parse its timestamps."""
    path = Path(path) if path is not None else SESSION_LABELS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Session label file not found: {path}")

    labels = pd.read_csv(path)
    for column in ("timestamp_start", "timestamp_end"):
        if column not in labels.columns:
            raise ValueError(f"'{column}' missing from {path}")
        labels[column] = pd.to_datetime(labels[column], utc=True, format="mixed")

    if "attack_type" not in labels.columns:
        raise ValueError(f"'attack_type' missing from {path}")
    if "target_ip" not in labels.columns:
        labels["target_ip"] = None

    return labels


def label_flows_by_session(flows: pd.DataFrame, sessions: pd.DataFrame,
                           tolerance_sec: float = 0.0) -> pd.DataFrame:
    """
    Assign a unified label to each flow using capture-session time windows.

    A flow is labelled when its ``timestamp_start`` falls inside a session window
    and its destination matches the session's ``target_ip`` (when that column is
    populated).  Flows in no window are labelled ``"Other"``.

    Args:
        flows: Flow frame with at least ``timestamp_start``.
        sessions: Frame from :func:`load_session_labels`.
        tolerance_sec: Optional slack on both window edges.

    Returns:
        Copy of ``flows`` with an added ``label`` column.
    """
    if flows.empty:
        out = flows.copy()
        out["label"] = pd.Series(dtype=object)
        return out

    if "timestamp_start" not in flows.columns:
        raise ValueError("Flow frame must contain a 'timestamp_start' column")

    starts = pd.to_datetime(
        pd.Series(
            np.asarray(flows["timestamp_start"].to_numpy(), dtype=np.float64),
            index=flows.index,
        ),
        unit="s",
        utc=True,
    )

    labels = np.full(len(flows), "Other", dtype=object)
    session_ids = np.full(len(flows), "", dtype=object)

    destination = flows["dst_ip"].to_numpy() if "dst_ip" in flows.columns else None

    for _, session in sessions.iterrows():
        window_start = session["timestamp_start"] - pd.Timedelta(seconds=tolerance_sec)
        window_end = session["timestamp_end"] + pd.Timedelta(seconds=tolerance_sec)
        mask = (starts >= window_start) & (starts <= window_end)

        target_ip = session.get("target_ip")
        if destination is not None and isinstance(target_ip, str) and target_ip:
            mask &= pd.Series(destination == target_ip, index=flows.index)

        labels[mask.to_numpy()] = session["attack_type"]
        session_ids[mask.to_numpy()] = str(session.get("session_id", ""))

    out = flows.copy()
    out["label"] = labels
    out["session_id"] = session_ids
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def extract_lab_features(lab_dir: Path, session_labels: Optional[Path] = None,
                         drop_identity: bool = True,
                         save_to: Optional[Path] = None) -> pd.DataFrame:
    """
    Build the lab-captured flow dataset from raw capture artefacts.

    Preference order:

    1. ``*.pcap`` files -> :func:`pcap_to_flows`
    2. ``conn.log`` files -> :func:`zeek_conn_log_to_flows`

    Flows are then labelled by capture-session window and projected onto the
    canonical schema.

    Args:
        lab_dir: Directory holding the captures (default ``data/lab_captures``).
        session_labels: Optional override for the session-label CSV.
        drop_identity: Remove src/dst IP and port columns (default True).
        save_to: Optional CSV output path.

    Returns:
        Lab flow DataFrame with canonical features and a ``label`` column.
    """
    lab_dir = Path(lab_dir)
    if not lab_dir.is_dir():
        raise FileNotFoundError(f"Lab capture directory not found: {lab_dir}")

    pcaps = sorted(lab_dir.rglob("*.pcap"))
    if pcaps:
        source = "pcap"
        flows = pcap_to_flows(pcaps)
    else:
        conn_logs = sorted(lab_dir.rglob("conn.log")) + sorted(lab_dir.rglob("*.conn.log"))
        if not conn_logs:
            raise FileNotFoundError(
                f"No .pcap or conn.log captures found under {lab_dir}. "
                "Run scripts/generate_lab_capture.py first."
            )
        source = "zeek"
        flows = zeek_conn_log_to_flows(conn_logs)

    flows.attrs["source"] = source

    sessions = load_session_labels(session_labels)
    labelled = label_flows_by_session(flows, sessions)

    features = [c for c in CANONICAL_FEATURES if c in labelled.columns]
    extras = ["timestamp_start", "timestamp_end", "session_id"]
    if not drop_identity:
        extras = [c for c in ("src_ip", "dst_ip", "src_port", "dst_port", "protocol") if c in labelled.columns] + extras

    out = labelled[features + extras + ["label"]].copy()
    out.attrs["source"] = source

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_to, index=False)

    return out


def write_zeek_conn_log(flows: pd.DataFrame, path: Path,
                        fields: Optional[list[str]] = None) -> Path:
    """
    Serialise flows as a Zeek-style JSON-lines ``conn.log``.

    Provided so the parser and the simulator agree on one interchange format and
    so tests can round-trip through a realistic artefact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fields is None:
        fields = [
            "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
            "proto", "duration", "orig_bytes", "resp_bytes", "orig_pkts",
            "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "orig_len",
            "resp_len", "orig_flags", "resp_flags", "service",
        ]

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for position, (_, row) in enumerate(flows.iterrows()):
            protocol = row.get("protocol", 6)
            protocol = 6 if protocol is None else int(protocol)
            record = {
                "ts": round(float(row["timestamp_start"]), 6),
                "uid": f"C{position:08X}",
                "id.orig_h": str(row.get("src_ip", "10.0.0.1")),
                "id.orig_p": int(row.get("src_port", 40000 + position % 20000)),
                "id.resp_h": str(row.get("dst_ip", "10.0.0.10")),
                "id.resp_p": int(row.get("dst_port", 80)),
                "proto": "tcp" if protocol == 6 else "udp",
                "duration": float(row["flow_duration"]),
                "orig_bytes": float(row["total_length_of_fwd_packets"]),
                "resp_bytes": float(row["total_length_of_bwd_packets"]),
                "orig_pkts": float(row["total_fwd_packets"]),
                "resp_pkts": float(row["total_bwd_packets"]),
                "orig_ip_bytes": float(row["total_length_of_fwd_packets"]),
                "resp_ip_bytes": float(row["total_length_of_bwd_packets"]),
                "orig_len": [round(float(row["fwd_packet_length_mean"]), 2)],
                "resp_len": [round(float(row["bwd_packet_length_mean"]), 2)],
                "orig_flags": _flags_to_letters(
                    row.get("syn_flag_count", 0), row.get("ack_flag_count", 0),
                    row.get("fin_flag_count", 0), row.get("psh_flag_count", 0),
                    row.get("rst_flag_count", 0),
                ),
                "resp_flags": [],
                "service": "http",
            }
            ordered = {field: record.get(field, "-") for field in fields}
            handle.write(json.dumps(ordered) + "\n")

    return path


def _flags_to_letters(syn, ack, fin, psh, rst) -> str:
    letters = []
    if syn:
        letters.append("S")
    if rst:
        letters.append("R")
    if fin:
        letters.append("F")
    if ack:
        letters.append("A")
    if psh:
        letters.append("P")
    return "".join(letters) or "-"


__all__ = [
    "CANONICAL_FEATURES",
    "align_to_canonical",
    "assert_behavior_only",
    "behavior_features",
    "extract_canonical_feature",
    "extract_lab_features",
    "feature_coverage",
    "identity_patterns",
    "is_identity_feature",
    "iter_pcap_packets",
    "label_flows_by_session",
    "load_feature_schema",
    "load_session_labels",
    "pcap_to_flows",
    "write_zeek_conn_log",
    "zeek_conn_log_to_flows",
]
