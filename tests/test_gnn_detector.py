"""
tests/test_gnn_detector.py — Tests for the host-graph GNN (PHASE_3 section 3.2).

Three of these tests are regression tests for bugs found during development.
Each would pass silently without the assertion, because the failure mode is
"the model trains and reports a plausible number" rather than a crash:

1. :meth:`TestHostCommGraph.test_node_traffic_features_survive_directed_graphs`
   — pandas index alignment silently zeroed the traffic features of any host that
   was only a source or only a destination, which is every host in a
   two-host exfiltration window.
2. :meth:`TestWindowGNNClassifier.test_labels_stay_aligned_with_graphs` — the
   training loop sliced a separate target tensor in unshuffled order while the
   loader shuffled the data, pairing every graph with the wrong label.
3. :meth:`TestGNNDetector.test_edge_attributes_receive_gradients` — an earlier
   draft handed ``edge_attr`` to ``SAGEConv``, which PyG silently ignores
   because its ``message()`` only receives ``x_j``.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture(scope="module")
def simulated():
    from src.data.graph_simulator import GraphScenarioConfig, simulate_graph_windows
    return simulate_graph_windows(
        GraphScenarioConfig(seed=0), n_windows_per_scenario=6)


@pytest.fixture(scope="module")
def records(simulated):
    from src.models.gnn_detector import HostCommGraph
    return HostCommGraph(window_seconds=60).build_windows(
        simulated, label_column="label")


class TestHostCommGraph:
    """Tests for src/models/gnn_detector.HostCommGraph."""

    def test_builds_expected_tensors(self, simulated):
        from src.models.gnn_detector import HostCommGraph, EDGE_FEATURES, NODE_FEATURES
        builder = HostCommGraph(window_seconds=60)
        window = simulated[simulated["window_id"] == 0]
        data = builder.build_graph(window, window_id=0)

        assert data.x.shape == (data.num_nodes, len(NODE_FEATURES))
        assert data.edge_attr.shape == (data.edge_index.shape[1], len(EDGE_FEATURES))
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.max() < data.num_nodes
        assert data.edge_index.min() >= 0

    def test_rejects_empty_window(self):
        from src.models.gnn_detector import GraphBuildError, HostCommGraph
        with pytest.raises(GraphBuildError):
            HostCommGraph(60).build_graph(pd.DataFrame())

    def test_rejects_single_host_window(self):
        from src.models.gnn_detector import GraphBuildError, HostCommGraph
        flows = pd.DataFrame({
            "src_ip": ["10.0.0.1", "10.0.0.1"],
            "dst_ip": ["10.0.0.1", "10.0.0.1"],
            "timestamp_start": [0.0, 1.0],
        })
        with pytest.raises(GraphBuildError):
            HostCommGraph(60).build_graph(flows)

    def test_accepts_zeek_column_names(self, simulated):
        from src.models.gnn_detector import HostCommGraph
        renamed = simulated.rename(columns={"src_ip": "id.orig_h",
                                            "dst_ip": "id.resp_h"})
        window = renamed[renamed["window_id"] == 0]
        data = HostCommGraph(60).build_graph(window)
        assert data.num_nodes >= 2

    def test_missing_host_columns_raise(self):
        from src.models.gnn_detector import GraphBuildError, HostCommGraph
        with pytest.raises(GraphBuildError):
            HostCommGraph(60).build_graph(pd.DataFrame({"a": [1], "b": [2]}))

    def test_rejects_non_positive_window(self):
        from src.models.gnn_detector import HostCommGraph
        with pytest.raises(ValueError):
            HostCommGraph(window_seconds=0)

    def test_window_index_is_floor_aligned(self):
        from src.models.gnn_detector import HostCommGraph
        indices = HostCommGraph(60).window_index([0, 59.9, 60, 119, 121])
        assert list(indices) == [0, 0, 1, 1, 2]

    def test_every_flow_lands_in_exactly_one_window(self, simulated):
        from src.models.gnn_detector import HostCommGraph
        builder = HostCommGraph(window_seconds=60)
        indices = builder.window_index(simulated["timestamp_start"])
        assert len(indices) == len(simulated)
        assert indices.min() >= 0

    def test_node_traffic_features_survive_directed_graphs(self):
        """
        Regression: a two-host window must report its traffic on both nodes.

        The previous implementation added ``groupby(src)`` to ``groupby(dst)``
        directly.  Pandas aligns two Series on the *union* of their indices, so
        every host that was only ever a source came back NaN and was then
        flattened to 0.0 — exactly the exfiltration topology, whose traffic
        features were all zero while the edge carried 3.8 MB.
        """
        from src.models.gnn_detector import HostCommGraph, NODE_FEATURES
        flows = pd.DataFrame({
            "src_ip": ["10.0.0.1"] * 5,
            "dst_ip": ["10.0.0.2"] * 5,
            "timestamp_start": [0.0, 1, 2, 3, 4],
            "total_length_of_fwd_packets": [1000.0] * 5,
            "total_fwd_packets": [10] * 5,
            "flow_duration": [100.0] * 5,
            "dst_port": [80] * 5,
        })
        data = HostCommGraph(60).build_graph(flows)
        x = data.x.numpy()
        byte_index = NODE_FEATURES.index("total_bytes")

        assert x[:, byte_index].min() > 0, (
            "traffic must be attributed to both endpoints of a one-way pair"
        )
        assert x[:, byte_index].max() == pytest.approx(5000.0, rel=1e-5)


class TestWindowLabelling:
    """Tests for the window-label derivation rule."""

    def test_attack_fraction_rule(self):
        from src.models.gnn_detector import HostCommGraph
        builder = HostCommGraph(60)
        frame = pd.DataFrame({
            "src_ip": ["1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4"],
            "dst_ip": ["2.2.2.2", "3.3.3.3", "4.4.4.4", "1.1.1.1"],
            "timestamp_start": [0.0, 1, 2, 3],
            "label": ["Benign", "Benign", "PortScan", "PortScan"],
        })
        label, fraction = builder._window_label(frame, "label", 0.5)
        assert label == "PortScan" and fraction == pytest.approx(0.5)

        label, fraction = builder._window_label(frame, "label", 0.75)
        assert label == "Benign"

    def test_unlabelled_window_is_marked_honestly(self, simulated):
        from src.models.gnn_detector import HostCommGraph
        records = HostCommGraph(60).build_windows(simulated, label_column=None)
        assert all(record.label == "Unlabelled" for record in records)
        assert all(record.is_attack == 0 for record in records)


class TestGNNDetector:
    """Tests for src/models/gnn_detector.GNNDetector."""

    def test_forward_returns_per_window_logits(self, records):
        from src.models.gnn_detector import GNNDetector
        from torch_geometric.loader import DataLoader

        model = GNNDetector(node_features=7, edge_features=4, hidden_dim=16)
        loader = DataLoader([record.data for record in records[:4]], batch_size=4)
        logits = model(next(iter(loader)))
        assert logits.shape == (4, 2)

    def test_edge_attributes_receive_gradients(self, records):
        """
        Regression: the edge branch must actually participate in learning.

        PyG's ``SAGEConv`` takes neither ``edge_dim`` nor ``edge_weight`` and its
        ``message()`` only receives ``x_j``, so an earlier draft that passed
        ``edge_attr`` to it either raised or was quietly ignored. The edge
        encoder is now a separate module; this asserts it receives a
        non-zero gradient.
        """
        from src.models.gnn_detector import GNNDetector
        from torch_geometric.loader import DataLoader

        model = GNNDetector(node_features=7, edge_features=4, hidden_dim=16)
        loader = DataLoader([record.data for record in records[:4]], batch_size=4)
        batch = next(iter(loader))
        loss = model(batch).sum()
        loss.backward()

        grad = model.edge_encoder.mlp[0].weight.grad
        assert grad is not None
        assert float(grad.abs().sum()) > 0

    def test_sage_conv_rejects_edge_dim(self):
        """Pins the API limitation that motivated the explicit edge branch."""
        from torch_geometric.nn import SAGEConv
        with pytest.raises(TypeError):
            SAGEConv(4, 8, edge_dim=4)


class TestWindowGNNClassifier:
    """Tests for src/models/gnn_detector.WindowGNNClassifier."""

    def test_labels_stay_aligned_with_graphs(self, records):
        """
        Regression: shuffling must not desynchronise labels from graphs.

        The training loop used to slice a separate target tensor in record
        order while the ``DataLoader`` shuffled the graphs, so each epoch paired
        every window with the wrong label. The model converged to a constant
        prediction and reported the majority-class rate. This asserts the
        model can actually fit a small labelled set.
        """
        from src.models.gnn_detector import WindowGNNClassifier
        benign = [r for r in records if r.is_attack == 0][:4]
        attack = [r for r in records if r.is_attack == 1][:4]
        assert len(benign) == 4 and len(attack) == 4

        clf = WindowGNNClassifier(epochs=60, hidden_dim=16, learning_rate=0.05,
                                  batch_size=4, dropout=0.0, seed=0)
        clf.fit(benign + attack)
        accuracy = (clf.predict(benign + attack)
                    == np.array([r.is_attack for r in benign + attack])).mean()
        assert accuracy > 0.9, (
            "the window classifier failed to fit a trivially separable set, "
            "which indicates labels are being scrambled during training"
        )

    def test_requires_fitting_before_predict(self, records):
        from src.models.gnn_detector import WindowGNNClassifier
        with pytest.raises(RuntimeError):
            WindowGNNClassifier().predict(records[:2])

    def test_rejects_empty_training_set(self):
        from src.models.gnn_detector import WindowGNNClassifier
        with pytest.raises(ValueError):
            WindowGNNClassifier().fit([])

    def test_probabilities_sum_to_one(self, records):
        from src.models.gnn_detector import WindowGNNClassifier
        clf = WindowGNNClassifier(epochs=5, seed=0).fit(records[:8])
        probabilities = clf.predict_proba(records[8:12])
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_latency_gate_is_reported_as_a_distribution(self, records):
        from src.models.gnn_detector import LATENCY_BUDGET_MS, WindowGNNClassifier
        clf = WindowGNNClassifier(epochs=3, seed=0).fit(records[:8])
        report = clf.benchmark_latency(n_iterations=30, warmup=5)
        assert report.n_iterations == 30
        assert report.min_ms <= report.median_ms <= report.max_ms
        assert report.p95_ms >= report.median_ms
        assert report.budget_ms == LATENCY_BUDGET_MS
        assert report.mean_ms < LATENCY_BUDGET_MS
        assert report.within_budget is True
        assert "G3" in report.format()

    def test_round_trip_save_and_load(self, records, tmp_path):
        from src.models.gnn_detector import WindowGNNClassifier
        clf = WindowGNNClassifier(epochs=5, hidden_dim=16, seed=0).fit(records[:8])
        path = clf.save(tmp_path / "gnn.pt")
        restored = WindowGNNClassifier.load(path)
        assert np.allclose(clf.predict_proba(records[8:10]),
                           restored.predict_proba(records[8:10]), atol=1e-5)


class TestSplitWindows:
    """Tests for the window-level train/test split."""

    def test_split_has_no_window_overlap(self, records):
        from src.models.gnn_detector import split_windows
        train, test = split_windows(records, test_fraction=0.25, seed=0)
        assert not (set(r.window_id for r in train)
                    & set(r.window_id for r in test))
        assert len(train) + len(test) == len(records)

    def test_split_requires_enough_windows(self, records):
        from src.models.gnn_detector import split_windows
        with pytest.raises(ValueError):
            split_windows(records[:3])

    def test_split_is_deterministic_for_a_seed(self, records):
        from src.models.gnn_detector import split_windows
        a, _ = split_windows(records, seed=7)
        b, _ = split_windows(records, seed=7)
        assert [r.window_id for r in a] == [r.window_id for r in b]