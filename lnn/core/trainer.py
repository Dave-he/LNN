import time

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

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
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
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)
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
        for x, y in dataloader:
            x = x.to(self.device)
            pred = self.model(x)
            if pred.dim() > y.dim():
                pred = pred[:, -1, :]
            all_preds.append(pred.cpu())
            all_targets.append(y)
        return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)
