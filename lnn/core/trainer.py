import inspect
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Generic trainer for sequence models (LNN, LSTM, GRU, etc.).

    Handles the full training loop with validation, early stopping,
    and metric tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr: float = 1e-3,
        device: str = "auto",
        patience: int = 15,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = criterion or nn.MSELoss()
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
        self.patience = patience

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def _to_device(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._to_device(item) for item in value)
        return value

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if isinstance(batch, dict):
            x = batch.get("x", batch.get("inputs"))
            y = batch.get("y", batch.get("target", batch.get("targets")))
            if x is None or y is None:
                raise ValueError("Batch dict must contain x/inputs and y/target/targets")
            target_keys = {"x", "inputs", "y", "target", "targets"}
            metadata = {key: value for key, value in batch.items() if key not in target_keys}
            return x, y, metadata

        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
                return x, y, {}
            if len(batch) == 3:
                x, y, metadata = batch
                return x, y, metadata or {}

        raise ValueError("Batch must be (x, y), (x, y, metadata), or a dict with x and y")

    def _model_forward(self, x: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
        candidate_kwargs = {
            "dt": metadata.get("dt", metadata.get("delta_t")),
            "mask": metadata.get("mask"),
        }
        candidate_kwargs = {key: value for key, value in candidate_kwargs.items() if value is not None}
        if not candidate_kwargs:
            return self.model(x)

        signature = inspect.signature(self.model.forward)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if accepts_kwargs or key in signature.parameters
        }
        return self.model(x, **kwargs) if kwargs else self.model(x)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            x, y, metadata = self._unpack_batch(batch)
            x = x.to(self.device)
            y = y.to(self.device)
            metadata = self._to_device(metadata)
            self.optimizer.zero_grad()
            pred = self._model_forward(x, metadata)
            if pred.dim() > y.dim():
                pred = pred[:, -1, :]
            loss = self.criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            x, y, metadata = self._unpack_batch(batch)
            x = x.to(self.device)
            y = y.to(self.device)
            metadata = self._to_device(metadata)
            pred = self._model_forward(x, metadata)
            if pred.dim() > y.dim():
                pred = pred[:, -1, :]
            loss = self.criterion(pred, y)
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(num_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_epochs: int = 100,
        verbose: bool = True,
    ) -> dict:
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch % 10 == 0 or epoch == 1):
                    print(f"Epoch {epoch:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            else:
                if verbose and (epoch % 10 == 0 or epoch == 1):
                    print(f"Epoch {epoch:4d} | Train: {train_loss:.6f}")

        elapsed = time.time() - start_time
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss if val_loader else None,
            "total_epochs": epoch,
            "elapsed_seconds": elapsed,
        }

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        all_preds = []
        all_targets = []
        for batch in dataloader:
            x, y, metadata = self._unpack_batch(batch)
            x = x.to(self.device)
            metadata = self._to_device(metadata)
            pred = self._model_forward(x, metadata)
            if pred.dim() > y.dim():
                pred = pred[:, -1, :]
            all_preds.append(pred.cpu())
            all_targets.append(y)
        return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)
