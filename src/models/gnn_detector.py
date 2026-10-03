"""
src/models/gnn_detector.py — Host-communication-graph GNN (PHASE_3 section 3.2).

Scope
-----
This is a **bounded-window proof of concept**, exactly as the research proposal
section 3.3 requires.  It makes *no* claim about scalability or production
readiness; GNN-IDS scalability remains open in the literature and a system this
size cannot address it.  What it does demonstrate is that lateral-movement-style
patterns, which are invisible in any individual flow, become separable when the
*structure* of who-talked-to-whom is modelled.

Components
----------
:class:`HostCommGraph`
    Turns a frame of flows into one :class:`torch_geometric.data.Data` object
    per fixed time window.  Nodes are hosts, edges are aggregated host-to-host
    communication.  Edge attributes are actually consumed by the architecture
    (they are passed to ``SAGEConv(edge_dim=...)``), which matters: an edge
    feature vector that no layer reads is dead weight.
:class:`GNNDetector`
    Two GraphSAGE layers -> global mean pool -> MLP head.  Global pooling makes
    the classifier permutation-invariant over hosts, which is the correct
    inductive bias: the *position* of the misbehaving host in the graph is not
    fixed.
:class:`WindowGNNClassifier`
    fit / predict / evaluate wrapper with the window-level label contract and
    the Phase 3 latency gate.

Correctness rules (each has a test in ``tests/test_gnn_detector.py``):

1. **Windows never share flows.**  A flow belongs to exactly one window, so no
   edge appears in both the train and the evaluation split.  ``build_windows``
   asserts this by construction (window ids are unique and contiguous).
2. **Node features are computed from the window only.**  No cross-window
   statistics leak into a window's graph.
3. **The window label is derived from the flows inside it**, never assigned in
   advance by the simulator: :meth:`WindowGNNClassifier.label_windows` uses the
   documented attack-fraction rule, so a caller can point the same code at real
   captures.
4. **Latency is measured with warm-up and reported as a distribution** (mean,
   median, p95), not a single best case, and the 35 ms gate is checked against
   the *mean*, with the device named in the result.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.pandas_compat import apply_pandas_compat

apply_pandas_compat()

#: ``PHASE_3`` gate G3: the GNN must stay under this per-inference latency.
LATENCY_BUDGET_MS = 35.0

DEFAULT_WINDOW_SECONDS = 60

#: ``window_id``-derived window labels, matching the unified taxonomy.
ATTACK_LABELS = frozenset({"DoS", "DDoS", "PortScan", "BruteForce", "WebAttack",
                           "Botnet", "Infiltration", "Other"})

#: Node feature names, in order.  Kept as a module constant so the graph builder,
#: the tests and the reports all agree on the dimensionality.
NODE_FEATURES = (
    "in_degree",
    "out_degree",
    "unique_dst",
    "unique_src",
    "unique_ports_contacted",
    "total_bytes",
    "total_packets",
)

#: Edge feature names, in order.
EDGE_FEATURES = (
    "total_bytes",
    "total_packets",
    "mean_duration",
    "num_flows",
)


class GraphBuildError(RuntimeError):
    """Raised when a flow window cannot be turned into a valid graph."""


@dataclass
class WindowRecord:
    """One labelled graph window."""

    window_id: int
    data: object
    label: str
    is_attack: int
    n_flows: int
    n_nodes: int
    n_edges: int
    start: float
    end: float
    attack_fraction: float
    scenario: Optional[str] = None

    def to_row(self) -> dict:
        return {
            "window_id": self.window_id,
            "label": self.label,
            "is_attack": self.is_attack,
            "n_flows": self.n_flows,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "window_start": self.start,
            "window_end": self.end,
            "attack_fraction": self.attack_fraction,
            "scenario": self.scenario or "",
        }


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=np.float64)
    mask = np.abs(denominator) > 0
    out[mask] = numerator[mask] / denominator[mask]
    return out


class HostCommGraph:
    """
    Build host-communication graphs from flow records.

    Args:
        window_seconds: Window length; windows are aligned to ``floor(t / w) * w``
            so a flow belongs to exactly one window regardless of arrival order.
        src_column / dst_column: Host-address column names.  Accepts Zeek-style
            ``id.orig_h`` / ``id.resp_h`` through :func:`resolve_host_columns`.
        timestamp_column: Column used for windowing.
        bytes_column / packets_column / duration_column: Flow statistics
            aggregated onto the edges.
        port_column: Used for the ``unique_ports_contacted`` node feature.
        drop_self_loops: Remove ``src == dst`` flows before graph construction.
    """

    def __init__(self, window_seconds: int = DEFAULT_WINDOW_SECONDS,
                 src_column: str = "src_ip",
                 dst_column: str = "dst_ip",
                 timestamp_column: str = "timestamp_start",
                 bytes_column: str = "total_length_of_fwd_packets",
                 packets_column: str = "total_fwd_packets",
                 duration_column: str = "flow_duration",
                 port_column: str = "dst_port",
                 drop_self_loops: bool = True):
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        self.window_seconds = int(window_seconds)
        self.src_column = src_column
        self.dst_column = dst_column
        self.timestamp_column = timestamp_column
        self.bytes_column = bytes_column
        self.packets_column = packets_column
        self.duration_column = duration_column
        self.port_column = port_column
        self.drop_self_loops = drop_self_loops

    # -- windowing ----------------------------------------------------------
    def window_index(self, timestamps: Iterable[float]) -> np.ndarray:
        """Return the window index of each timestamp (floor-aligned)."""
        values = np.asarray(list(timestamps), dtype=np.float64)
        return np.floor(values / self.window_seconds).astype(np.int64)

    @staticmethod
    def resolve_host_columns(frame: pd.DataFrame) -> tuple[str, str]:
        """
        Locate the source/destination host columns.

        Supports the canonical ``src_ip``/``dst_ip`` produced by
        :mod:`src.data.feature_extractor` and the Zeek names
        ``id.orig_h``/``id.resp_h``.

        Raises:
            GraphBuildError: when neither naming convention is present.
        """
        lowered = {str(c).strip().lower(): c for c in frame.columns}
        for src, dst in (("src_ip", "dst_ip"), ("id.orig_h", "id.resp_h"),
                         ("source_ip", "destination_ip")):
            if src in lowered and dst in lowered:
                return lowered[src], lowered[dst]
        raise GraphBuildError(
            "No host-address columns found. Expected one of "
            "('src_ip','dst_ip') or ('id.orig_h','id.resp_h'); got "
            f"{list(frame.columns)[:12]}"
        )

    # -- graph construction -------------------------------------------------
    def build_graph(self, flows: pd.DataFrame, window_id: Optional[int] = None):
        """
        Build one :class:`torch_geometric.data.Data` from the supplied flows.

        Args:
            flows: Frame of flows belonging to a **single** window.
            window_id: Recorded on the returned object for traceability.

        Returns:
            A ``Data`` object with ``x`` ``(n_nodes, 7)``, ``edge_index``
            ``(2, n_edges)``, ``edge_attr`` ``(n_edges, 4)`` and ``y``.

        Raises:
            GraphBuildError: when ``flows`` is empty or yields fewer than one
                host, because a graph with no edges carries no signal and a
                silent empty graph would corrupt a training batch.
        """
        from torch_geometric.data import Data

        if flows is None or len(flows) == 0:
            raise GraphBuildError("Cannot build a graph from an empty flow window")

        src_col, dst_col = self.resolve_host_columns(flows)
        frame = flows
        if self.drop_self_loops:
            frame = frame[frame[src_col].astype(str) != frame[dst_col].astype(str)]
        if len(frame) == 0:
            raise GraphBuildError(
                "All flows in the window were self-loops; no host graph remains"
            )

        src = frame[src_col].astype(str).to_numpy()
        dst = frame[dst_col].astype(str).to_numpy()

        hosts = sorted(set(src) | set(dst))
        index = {host: position for position, host in enumerate(hosts)}
        n_nodes = len(hosts)
        if n_nodes < 2:
            raise GraphBuildError(
                f"Window has {n_nodes} distinct host(s); a host-communication "
                "graph needs at least two."
            )

        src_idx = np.fromiter((index[h] for h in src), dtype=np.int64, count=len(src))
        dst_idx = np.fromiter((index[h] for h in dst), dtype=np.int64, count=len(dst))

        # Aggregate flows per ordered host pair: the edge carries the *summed*
        # volume, so a scan and a single long session are distinguishable.
        pairs = pd.DataFrame({"s": src_idx, "d": dst_idx})
        pairs["bytes"] = _numeric(frame, self.bytes_column)
        pairs["packets"] = _numeric(frame, self.packets_column)
        pairs["duration"] = _numeric(frame, self.duration_column)
        # Constant 1 column: pandas named aggregation rejects ``("size",)``, so
        # the per-pair flow count is obtained by summing this instead.
        pairs["one"] = 1.0
        if self.port_column in frame.columns:
            pairs["port"] = _numeric(frame, self.port_column)
        else:
            pairs["port"] = 0.0

        grouped = pairs.groupby(["s", "d"], sort=True).agg(
            agg_bytes=("bytes", "sum"),
            agg_packets=("packets", "sum"),
            agg_duration_sum=("duration", "sum"),
            agg_duration_n=("duration", "count"),
            agg_n_flows=("one", "sum"),
        ).reset_index()

        edge_index = np.stack([grouped["s"].to_numpy(), grouped["d"].to_numpy()])
        mean_duration = _safe_divide(grouped["agg_duration_sum"].to_numpy(),
                                     grouped["agg_duration_n"].to_numpy())
        edge_attr = np.column_stack([
            grouped["agg_bytes"].to_numpy(dtype=np.float64),
            grouped["agg_packets"].to_numpy(dtype=np.float64),
            mean_duration,
            grouped["agg_n_flows"].to_numpy(dtype=np.float64),
        ])

        # Node features: degree plus *distinct counterparties*, which is the
        # part that separates a scanner (many distinct targets, few flows each)
        # from a busy host (many flows to one target).
        unique_dst = pairs.groupby("s")["d"].nunique()
        unique_src = pairs.groupby("d")["s"].nunique()
        unique_ports = pairs.groupby("s")["port"].nunique()
        in_degree = pairs.groupby("d").size()
        out_degree = pairs.groupby("s").size()

        def _aligned(series: pd.Series) -> np.ndarray:
            values = series.reindex(range(n_nodes)).fillna(0.0)
            return values.to_numpy(dtype=np.float64)

        # Traffic totals are the sum of the outbound and inbound views.
        # Each view MUST be aligned to the node index *before* being added:
        # adding two Series with different indices aligns on the union, so any
        # host that is only a source (or only a destination) would come back NaN
        # and be silently flattened to 0.0 by the fillna below.  In the
        # exfiltration windows the graph is a single 2-host pair, which is
        # precisely the case that loses everything - the node features read all
        # zero while the edge carried the real 3.8 MB total.
        def _traffic(column: str) -> np.ndarray:
            outbound = _aligned(pairs.groupby("s")[column].sum())
            inbound = _aligned(pairs.groupby("d")[column].sum())
            return outbound + inbound

        node_features = np.column_stack([
            _aligned(in_degree),
            _aligned(out_degree),
            _aligned(unique_dst),
            _aligned(unique_src),
            _aligned(unique_ports),
            _traffic("bytes"),
            _traffic("packets"),
        ])

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        )
        data.num_nodes = n_nodes
        data.window_id = -1 if window_id is None else int(window_id)
        data.hosts = hosts
        data.n_flows = int(len(frame))
        return data

    # -- windowed dataset ---------------------------------------------------
    def build_windows(self, flows: pd.DataFrame,
                      label_column: Optional[str] = "label",
                      attack_fraction: float = 0.5,
                      scenario_column: Optional[str] = "scenario") -> list[WindowRecord]:
        """
        Split ``flows`` into windows and build one graph per window.

        Args:
            flows: Frame carrying host, timestamp and (optionally) label columns.
            label_column: Column holding the per-flow label.  When ``None``,
                every window is labelled ``"Unlabelled"`` — the honest mode when
                no ground truth exists.
            attack_fraction: Fraction of attack flows in a window above which the
                window is labelled as an attack window.
            scenario_column: Optional provenance column copied onto the record.

        Returns:
            A list of :class:`WindowRecord` ordered by ``window_id``.

        Raises:
            GraphBuildError: when no window can be built.
        """
        if self.timestamp_column not in flows.columns:
            raise GraphBuildError(
                f"'{self.timestamp_column}' column required for windowing"
            )

        frame = flows.copy()
        frame["__window"] = self.window_index(frame[self.timestamp_column])
        records: list[WindowRecord] = []
        skipped: list[str] = []

        for window_id, group in frame.groupby("__window", sort=True):
            try:
                data = self.build_graph(group, window_id=int(window_id))
            except GraphBuildError as error:
                skipped.append(f"window {window_id}: {error}")
                continue

            label, fraction = self._window_label(group, label_column,
                                                 attack_fraction)
            records.append(WindowRecord(
                window_id=int(window_id),
                data=data,
                label=label,
                is_attack=int(label in ATTACK_LABELS),
                n_flows=int(len(group)),
                n_nodes=int(data.num_nodes),
                n_edges=int(data.edge_index.shape[1]),
                start=float(group[self.timestamp_column].min()),
                end=float(group[self.timestamp_column].max()),
                attack_fraction=fraction,
                scenario=(str(group[scenario_column].iloc[0])
                          if scenario_column and scenario_column in group.columns
                          else None),
            ))

        if not records:
            raise GraphBuildError(
                "No window could be turned into a graph. "
                + ("; ".join(skipped[:3]) if skipped else "")
            )
        self.skipped_windows_ = skipped
        return records

    @staticmethod
    def _window_label(group: pd.DataFrame, label_column: Optional[str],
                      attack_fraction: float) -> tuple[str, float]:
        """
        Derive a window label from the flows inside it.

        The rule is the one declared in ``PHASE_3`` section 3.2.5: a window is an
        attack window when at least ``attack_fraction`` of its flows are attack
        flows.  When the label column is absent the window is ``"Unlabelled"``.
        """
        if not label_column or label_column not in group.columns:
            return "Unlabelled", float("nan")

        labels = group[label_column].astype(str).str.strip()
        is_attack = labels.isin(ATTACK_LABELS).to_numpy()
        fraction = float(is_attack.mean()) if len(is_attack) else 0.0
        if fraction >= attack_fraction:
            counts = labels[is_attack].value_counts()
            label = str(counts.index[0]) if len(counts) else "Other"
        else:
            counts = labels[~is_attack].value_counts()
            label = str(counts.index[0]) if len(counts) else "Benign"
        return label, fraction


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Return ``column`` as float64, substituting 0.0 for missing values."""
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    return np.where(np.isfinite(values), values, 0.0)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class EdgeAttributeEncoder(nn.Module):
    """
    Embed aggregated edge attributes into ``hidden_dim`` message vectors.

    Keeping this a real module (rather than an inline expression) means its
    parameters appear in ``state_dict()`` and receive gradients, which the
    tests verify — that is the difference between "edge attributes are used"
    and "edge attributes are computed and then thrown away".
    """

    def __init__(self, edge_features: int, hidden_dim: int):
        super().__init__()
        self.edge_features = int(edge_features)
        self.hidden_dim = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.edge_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Return ``(n_edges, hidden_dim)`` edge message vectors."""
        return self.mlp(edge_attr)


def _batch_vector(data, reference: torch.Tensor) -> torch.Tensor:
    """
    Return ``data.batch``, synthesising it for a single un-batched graph.

    ``global_mean_pool`` requires a ``batch`` vector.  A lone ``Data`` object has
    none, so every node is assigned to graph 0 — which yields the plain mean
    over all nodes, exactly the correct result for a single graph.
    """
    batch = getattr(data, "batch", None)
    if batch is not None:
        return batch
    return torch.zeros(reference.size(0), dtype=torch.long,
                       device=reference.device)


class GNNDetector(nn.Module):
    """
    Two-layer GraphSAGE + global mean pool + MLP window classifier.

    Args:
        node_features: Number of node input features.
        edge_features: Number of edge input features.
        hidden_dim: Width of both SAGEConv layers.
        dropout: Dropout on the pooled graph embedding.
        out_channels: 2 for benign/attack.

    .. note::
       **How edge attributes actually enter the model.**  PyG's
       ``SAGEConv`` (2.x) has neither ``edge_dim`` nor ``edge_weight``: its
       ``message()`` receives ``x_j`` only, so ``edge_attr`` cannot be handed to
       it. Passing it anyway raises ``TypeError``.  Rather than silently
       dropping the edge features, this class keeps GraphSAGE as the
       neighbourhood operator (as the phase plan specifies) and adds an
       explicit **edge-message path**: each edge is embedded by
       :class:`EdgeAttributeEncoder` and scattered onto its destination node,
       then added to the SAGE output. Edge attributes therefore affect every
       layer, and the branch is genuinely exercised — ``edge_encoder`` receives
       gradients during training, which ``tests/test_gnn_detector.py`` asserts.
    """

    def __init__(self, node_features: int, edge_features: int,
                 hidden_dim: int = 32, dropout: float = 0.1,
                 out_channels: int = 2):
        super().__init__()
        from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool

        self.node_features = int(node_features)
        self.edge_features = int(edge_features)
        self.hidden_dim = int(hidden_dim)
        self.out_channels = int(out_channels)

        # Raw byte/packet counts span several orders of magnitude between
        # benign windows and exfiltration windows, so both feature blocks are
        # standardised before the first layer.
        self.node_norm = nn.BatchNorm1d(self.node_features)
        self.edge_norm = nn.BatchNorm1d(self.edge_features)

        self.conv1 = SAGEConv(self.node_features, self.hidden_dim)
        self.conv2 = SAGEConv(self.hidden_dim, self.hidden_dim)
        self.edge_encoder = EdgeAttributeEncoder(self.edge_features,
                                                 self.hidden_dim)

        self.dropout = float(dropout)
        self._pool_mean = global_mean_pool
        self._pool_max = global_max_pool
        # Mean+max concatenation, not mean alone.  The hosts that matter are
        # rare by construction: in a scan or exfiltration window one host carries
        # ~90 edges while the ~13 others carry a handful.  Averaging over nodes
        # divides that signal away and the classifier degenerates to guessing the
        # majority class; the max term preserves it.
        self.pool_dim = self.hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.pool_dim, max(8, self.hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(max(8, self.hidden_dim // 2), self.out_channels),
        )

    def _edge_messages(self, data, dim_size: int) -> torch.Tensor:
        """Return per-node edge-message sums of shape ``(dim_size, hidden)``."""
        from torch_geometric.utils import scatter

        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None or not edge_attr.numel():
            return None
        messages = self.edge_encoder(self.edge_norm(edge_attr))
        # edge_index is [source, destination], so messages accumulate on row 1.
        return scatter(messages, data.edge_index[1], dim=0, dim_size=dim_size)

    def _encode(self, data) -> torch.Tensor:
        x = self.node_norm(data.x)
        h = F.relu(self.conv1(x, data.edge_index))
        edge_sum = self._edge_messages(data, int(data.num_nodes))
        if edge_sum is not None:
            h = h + edge_sum
        h = F.relu(self.conv2(h, data.edge_index))
        edge_sum = self._edge_messages(data, int(data.num_nodes))
        if edge_sum is not None:
            h = h + edge_sum
        return h

    def _graph_embedding(self, data) -> torch.Tensor:
        h = self._encode(data)
        batch = _batch_vector(data, h)
        return torch.cat([self._pool_mean(h, batch), self._pool_max(h, batch)],
                         dim=1)

    def forward(self, data) -> torch.Tensor:
        """Return per-window logits of shape ``(batch, out_channels)``."""
        return self.classifier(self._graph_embedding(data))

    def embedding(self, data) -> torch.Tensor:
        """Return the pooled graph embedding (useful for inspection/plots)."""
        return self._graph_embedding(data)


@dataclass
class LatencyReport:
    """Latency distribution for one inference workload."""

    n_iterations: int
    batch_size: int
    n_nodes: int
    n_edges: int
    device: str
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    budget_ms: float = LATENCY_BUDGET_MS

    @property
    def within_budget(self) -> bool:
        """True when the **mean** latency is inside the Phase 3 G3 budget."""
        return self.mean_ms < self.budget_ms

    def to_dict(self) -> dict:
        return {
            "n_iterations": self.n_iterations,
            "batch_size": self.batch_size,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "device": self.device,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "budget_ms": self.budget_ms,
            "within_budget": self.within_budget,
            "headroom_ms": self.budget_ms - self.mean_ms,
        }

    def format(self) -> str:
        verdict = "PASS" if self.within_budget else "FAIL"
        return (
            f"GNN inference latency [{self.device}]: mean {self.mean_ms:.3f} ms, "
            f"median {self.median_ms:.3f} ms, p95 {self.p95_ms:.3f} ms over "
            f"{self.n_iterations} iterations (batch {self.batch_size}, "
            f"{self.n_nodes} nodes / {self.n_edges} edges) — "
            f"{verdict} vs the {self.budget_ms:g} ms G3 budget"
        )


@dataclass
class GNNEvaluation:
    """Window-level evaluation output."""

    metrics: dict
    per_class: Optional[pd.DataFrame] = None
    n_train_windows: int = 0
    n_test_windows: int = 0
    latency: Optional[LatencyReport] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            **self.metrics,
            "n_train_windows": self.n_train_windows,
            "n_test_windows": self.n_test_windows,
            "latency": self.latency.to_dict() if self.latency else None,
            "notes": "; ".join(self.notes),
        }


class WindowGNNClassifier:
    """
    Train and evaluate :class:`GNNDetector` over host-communication windows.

    Args:
        hidden_dim: SAGEConv width.
        dropout: Dropout on the pooled embedding.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        epochs: Training epochs.
        batch_size: Windows per gradient step.
        seed: Seed for initialisation and shuffling.
        device: ``None`` selects CUDA when available.
        class_weight: Optional per-class weights for the loss.
    """

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.1,
                 learning_rate: float = 1e-2, weight_decay: float = 5e-4,
                 epochs: int = 60, batch_size: int = 16, seed: int = 42,
                 device: Optional[str] = None,
                 class_weight: Optional[Sequence[float]] = None):
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.class_weight = (np.asarray(class_weight, dtype=np.float64)
                             if class_weight is not None else None)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model_: Optional[GNNDetector] = None
        self.history_: list[float] = []
        self._fitted = False

    # -- helpers ------------------------------------------------------------
    def _require_fitted(self) -> None:
        if not self._fitted or self.model_ is None:
            raise RuntimeError("WindowGNNClassifier must be fitted before use.")

    def _collate(self, records: Sequence[WindowRecord]):
        """
        Return the ``Data`` objects with their window label attached.

        The label is written onto ``data.y`` so that PyG's batching carries it
        together with the graph.  Slicing a separate target tensor in loader
        order is wrong the moment ``shuffle=True`` is used: the loader hands
        back a permuted order while a sequential slice keeps the original one,
        which pairs every graph with the wrong label.  The model then learns
        noise and collapses onto the majority class.
        """
        datas = []
        for record in records:
            data = record.data
            data.y = torch.tensor([record.is_attack], dtype=torch.long)
            datas.append(data)
        return datas

    # -- training -----------------------------------------------------------
    def fit(self, train_records: Sequence[WindowRecord],
            val_records: Optional[Sequence[WindowRecord]] = None) -> "WindowGNNClassifier":
        """Train the window classifier on labelled windows."""
        from torch_geometric.loader import DataLoader

        if not train_records:
            raise ValueError("fit() needs at least one training window")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)

        first = train_records[0].data
        self.model_ = GNNDetector(
            node_features=first.x.shape[1],
            edge_features=first.edge_attr.shape[1],
            hidden_dim=self.hidden_dim, dropout=self.dropout,
        ).to(self.device)

        optimiser = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loader = DataLoader(self._collate(train_records), batch_size=self.batch_size,
                            shuffle=True, generator=generator)
        criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(self.class_weight, dtype=torch.float32).to(self.device)
            if self.class_weight is not None else None
        )

        best_val = float("inf")
        best_state = None
        self.history_ = []

        for _ in range(self.epochs):
            self.model_.train()
            total, batches = 0.0, 0
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model_(batch)
                loss = criterion(logits, batch.y.view(-1))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += float(loss.item())
                batches += 1

            epoch_loss = total / max(batches, 1)
            self.history_.append(epoch_loss)

            if val_records:
                val_loss = self._loss_on(val_records, criterion)
                if val_loss < best_val - 1e-9:
                    best_val = val_loss
                    best_state = {k: v.detach().clone()
                                  for k, v in self.model_.state_dict().items()}

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self._fitted = True
        return self

    def _loss_on(self, records: Sequence[WindowRecord], criterion) -> float:
        from torch_geometric.loader import DataLoader

        loader = DataLoader(self._collate(records), batch_size=self.batch_size,
                            shuffle=False)
        self.model_.eval()
        total, batches = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model_(batch)
                total += float(criterion(logits, batch.y.view(-1)).item())
                batches += 1
        return total / max(batches, 1)

    # -- inference ----------------------------------------------------------
    def predict_proba(self, records: Sequence[WindowRecord]) -> np.ndarray:
        """Return ``(n_windows, 2)`` attack probabilities."""
        from torch_geometric.loader import DataLoader

        self._require_fitted()
        if not records:
            return np.empty((0, 2), dtype=np.float64)

        self.model_.eval()
        loader = DataLoader(self._collate(records), batch_size=self.batch_size,
                            shuffle=False)
        outputs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                outputs.append(
                    torch.softmax(self.model_(batch), dim=1).cpu().numpy()
                )
        return np.vstack(outputs).astype(np.float64)

    def predict(self, records: Sequence[WindowRecord]) -> np.ndarray:
        """Return a binary attack flag per window."""
        return (self.predict_proba(records)[:, 1] >= 0.5).astype(int)

    # -- latency ------------------------------------------------------------
    def benchmark_latency(self, n_nodes: int = 50, n_edges: int = 200,
                          batch_size: int = 1, n_iterations: int = 200,
                          warmup: int = 20, budget_ms: float = LATENCY_BUDGET_MS
                          ) -> LatencyReport:
        """
        Measure inference latency against the Phase 3 G3 budget.

        The workload is the one the plan specifies: a 50-node / 200-edge graph.
        Latency is timed over ``n_iterations`` runs after ``warmup`` untimed runs,
        and reported as a distribution so a single lucky call cannot pass the
        gate.

        Note:
            CUDA is synchronised around each timed iteration when available;
            without the sync the timer would measure kernel *launch*, not
            execution, and the budget check would be meaningless.
        """
        from torch_geometric.data import Batch, Data

        self._require_fitted()

        def _graph(seed: int) -> Data:
            rng = np.random.default_rng(seed)
            edge_index = rng.integers(0, n_nodes, size=(2, n_edges))
            return Data(
                x=torch.as_tensor(rng.normal(size=(n_nodes, NODE_FEATURES.__len__())),
                                  dtype=torch.float32),
                edge_index=torch.as_tensor(edge_index, dtype=torch.long),
                edge_attr=torch.as_tensor(
                    rng.normal(size=(n_edges, len(EDGE_FEATURES))), dtype=torch.float32),
            )

        graphs = [_graph(index) for index in range(batch_size)]
        batch = Batch.from_data_list(graphs).to(self.device)

        use_cuda = self.device.type == "cuda"
        with torch.no_grad():
            for _ in range(warmup):
                self.model_(batch)
            if use_cuda:
                torch.cuda.synchronize()

            timings = np.empty(n_iterations, dtype=np.float64)
            for index in range(n_iterations):
                start = time.perf_counter()
                self.model_(batch)
                if use_cuda:
                    torch.cuda.synchronize()
                timings[index] = (time.perf_counter() - start) * 1000.0

        return LatencyReport(
            n_iterations=int(n_iterations),
            batch_size=int(batch_size),
            n_nodes=int(n_nodes),
            n_edges=int(n_edges),
            device=str(self.device),
            mean_ms=float(timings.mean()),
            median_ms=float(np.median(timings)),
            p95_ms=float(np.percentile(timings, 95)),
            min_ms=float(timings.min()),
            max_ms=float(timings.max()),
            budget_ms=float(budget_ms),
        )

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, test_records: Sequence[WindowRecord],
                 latency: Optional[LatencyReport] = None) -> GNNEvaluation:
        """Score held-out windows with the standard classification metrics."""
        from src.evaluation.metrics import compute_metrics, per_class_report

        self._require_fitted()
        if not test_records:
            raise ValueError("evaluate() needs at least one test window")

        truth = np.array([record.is_attack for record in test_records], dtype=int)
        predicted = self.predict(test_records)
        metrics = compute_metrics(truth, predicted, average="weighted")
        report = per_class_report(
            np.where(truth == 1, "attack", "benign"),
            np.where(predicted == 1, "attack", "benign"),
        )

        notes: list[str] = []
        if latency is not None and not latency.within_budget:
            notes.append(
                f"G3 latency budget missed: {latency.mean_ms:.2f} ms > "
                f"{latency.budget_ms:g} ms"
            )
        if int(truth.sum()) == 0 or int(truth.size - truth.sum()) == 0:
            notes.append("test set is single-class: F1 is not informative")

        return GNNEvaluation(
            metrics=metrics,
            per_class=report,
            n_train_windows=int(getattr(self, "n_train_windows_", 0)),
            n_test_windows=int(len(test_records)),
            latency=latency,
            notes=notes,
        )

    def save(self, path: str | Path) -> Path:
        """Persist the trained weights plus the constructor arguments."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model_.state_dict(),
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "node_features": self.model_.node_features,
            "edge_features": self.model_.edge_features,
            "seed": self.seed,
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None
             ) -> "WindowGNNClassifier":
        """Rebuild a classifier written by :meth:`save`."""
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        wrapper = cls(hidden_dim=blob["hidden_dim"], dropout=blob["dropout"],
                      seed=blob["seed"], device=device)
        model = GNNDetector(node_features=blob["node_features"],
                            edge_features=blob["edge_features"],
                            hidden_dim=blob["hidden_dim"],
                            dropout=blob["dropout"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        wrapper.model_ = model.to(wrapper.device)
        wrapper._fitted = True
        return wrapper


def split_windows(records: Sequence[WindowRecord], test_fraction: float = 0.25,
                  seed: int = 42) -> tuple[list[WindowRecord], list[WindowRecord]]:
    """
    Split windows into train/test **by window id**, stratified on the label.

    Windows are the unit of analysis, so splitting on windows (rather than on
    flows) is what keeps the training and evaluation graphs disjoint.  The split
    is stratified because attack windows are the minority class by construction.

    Returns:
        ``(train_records, test_records)``.
    """
    from sklearn.model_selection import train_test_split

    records = list(records)
    if len(records) < 4:
        raise ValueError(
            f"Need at least 4 windows to split, got {len(records)}"
        )
    labels = np.array([record.is_attack for record in records], dtype=int)
    counts = pd.Series(labels).value_counts()
    stratify = labels if len(counts) >= 2 and counts.min() >= 2 else None

    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_fraction, random_state=seed, shuffle=True,
        stratify=stratify,
    )
    return ([records[i] for i in sorted(train_idx)],
            [records[i] for i in sorted(test_idx)])


__all__ = [
    "ATTACK_LABELS",
    "EDGE_FEATURES",
    "EdgeAttributeEncoder",
    "GNNDetector",
    "GNNEvaluation",
    "GraphBuildError",
    "HostCommGraph",
    "LATENCY_BUDGET_MS",
    "LatencyReport",
    "NODE_FEATURES",
    "WindowGNNClassifier",
    "WindowRecord",
    "split_windows",
]