"""
src/models/baseline_dnn.py — PyTorch Deep Neural Network baseline classifier for NIDS.

Provides an sklearn-compatible API (fit, predict, predict_proba) wrapping a PyTorch
multi-layer perceptron. Supports standard clean training and adversarial training (FGSM/PGD).
"""
from __future__ import annotations

import copy
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder


class _TabularMLP(nn.Module):
    """Feedforward network for tabular NIDS flow classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        n_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DNNBaseline:
    """
    Sklearn-compatible Deep Neural Network classifier for tabular flow features.

    Parameters
    ----------
    hidden_dims : Sequence[int]
        Hidden layer sizes (default: (64, 32)).
    learning_rate : float
        Adam optimizer learning rate (default: 0.001).
    weight_decay : float
        L2 regularization weight decay (default: 1e-4).
    epochs : int
        Number of training epochs (default: 20).
    batch_size : int
        Mini-batch size (default: 64).
    dropout : float
        Dropout probability (default: 0.1).
    device : Optional[str]
        Device to run on ('cpu', 'cuda', or None for auto-detect).
    random_state : int
        Seed for reproducibility (default: 42).
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = (64, 32),
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 64,
        dropout: float = 0.1,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.hidden_dims = tuple(hidden_dims)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.label_encoder_ = LabelEncoder()
        self.model_: Optional[_TabularMLP] = None
        self._fitted = False
        self.input_dim_: Optional[int] = None
        self.n_classes_: Optional[int] = None

    @property
    def torch_model_(self) -> nn.Module:
        """Access underlying PyTorch nn.Module in eval mode."""
        if not self._fitted or self.model_ is None:
            raise RuntimeError("Model must be fitted before accessing torch_model_.")
        self.model_.eval()
        return self.model_

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor, Sequence],
        adversarial_attacker: Optional[object] = None,
        adv_epsilon: float = 0.1,
        adv_ratio: float = 0.5,
    ) -> "DNNBaseline":
        """
        Fit model on training data.

        If `adversarial_attacker` is provided and `adv_ratio > 0`, performs adversarial
        training by perturbing `adv_ratio` fraction of each mini-batch using the attacker.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Handle X
        if isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().numpy().astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        # Handle y
        if isinstance(y, torch.Tensor):
            y_arr = y.detach().cpu().numpy()
        else:
            y_arr = np.asarray(y)

        y_encoded = self.label_encoder_.fit_transform(y_arr)
        self.n_classes_ = len(self.label_encoder_.classes_)
        self.input_dim_ = X_arr.shape[1]

        self.model_ = _TabularMLP(
            input_dim=self.input_dim_,
            hidden_dims=self.hidden_dims,
            n_classes=self.n_classes_,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        n_samples = len(X_arr)
        indices = np.arange(n_samples)

        for epoch in range(self.epochs):
            self.model_.train()
            np.random.shuffle(indices)

            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                if len(batch_idx) < 2:
                    continue

                X_b = torch.tensor(X_arr[batch_idx], dtype=torch.float32, device=self.device)
                y_b = torch.tensor(y_encoded[batch_idx], dtype=torch.long, device=self.device)

                if adversarial_attacker is not None and adv_ratio > 0.0:
                    n_adv = int(len(X_b) * adv_ratio)
                    if n_adv > 0:
                        self.model_.eval()
                        X_sub = X_b[:n_adv]
                        y_sub = y_b[:n_adv]
                        if hasattr(adversarial_attacker, "fgsm"):
                            X_adv = adversarial_attacker.fgsm(
                                self.model_, X_sub, y_sub, epsilon=adv_epsilon
                            )
                        else:
                            X_adv = X_sub
                        X_b = torch.cat([X_adv, X_b[n_adv:]], dim=0)
                        self.model_.train()

                optimizer.zero_grad()
                outputs = self.model_(X_b)
                loss = criterion(outputs, y_b)
                loss.backward()
                optimizer.step()

        self._fitted = True
        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(X)
        pred_idx = np.argmax(probas, axis=1)
        return self.label_encoder_.inverse_transform(pred_idx)

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted or self.model_ is None:
            raise RuntimeError("Model must be fitted before prediction.")

        if isinstance(X, torch.Tensor):
            X_tensor = X.to(self.device, dtype=torch.float32)
        else:
            X_tensor = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)

        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            probas = torch.softmax(outputs, dim=-1).cpu().numpy()
        return probas

    def clone(self) -> "DNNBaseline":
        """Create an un-fitted copy with the same hyperparameters."""
        return DNNBaseline(
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            batch_size=self.batch_size,
            dropout=self.dropout,
            device=self.device,
            random_state=self.random_state,
        )
