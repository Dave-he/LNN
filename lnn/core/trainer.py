import time
import os
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Enhanced trainer for sequence models (LNN, LSTM, GRU, etc.).

    Handles full training loop with validation, early stopping,
    learning rate scheduling, checkpointing, and metric tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: Any = None,
        lr: float = 1e-3,
        device: str = "auto",
        patience: int = 15,
        gradient_clip: float = 1.0,
        checkpoint_dir: str | None = None,
        use_amp: bool = False,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = criterion or nn.MSELoss()
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.lrs: list[float] = []
        self.epoch_times: list[float] = []
        self.best_val_loss = float("inf")
        self.best_state = None
        self.best_epoch = 0

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()

            if self.use_amp and self.scaler:
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
                    if pred.dim() > y.dim():
                        pred = pred[:, -1, :]
                    loss = self.criterion(pred, y)
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x)
                if pred.dim() > y.dim():
                    pred = pred[:, -1, :]
                loss = self.criterion(pred, y)
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
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

    def save_checkpoint(self, epoch: int, val_loss: float, path: str | None = None) -> str:
        if path is None:
            if self.checkpoint_dir:
                path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            else:
                return ""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, path)
        return path

    def load_checkpoint(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("val_loss", float("inf"))
        return checkpoint

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_epochs: int = 100,
        verbose: bool = True,
        save_best_only: bool = True,
        save_freq: int = 10,
    ) -> dict:
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    self.best_epoch = epoch
                    patience_counter = 0
                    if save_best_only and self.checkpoint_dir:
                        best_path = os.path.join(self.checkpoint_dir, "model_best.pt")
                        self.save_checkpoint(epoch, val_loss, best_path)
                else:
                    patience_counter += 1
            else:
                val_loss = None

            if self.lr_scheduler:
                if hasattr(self.lr_scheduler, 'step'):
                    if val_loader and hasattr(self.lr_scheduler, 'monitor'):
                        self.lr_scheduler.step(val_loss)
                    else:
                        self.lr_scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.lrs.append(current_lr)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                log_str = f"Epoch {epoch:4d} | Train: {train_loss:.6f}"
                if val_loss is not None:
                    log_str += f" | Val: {val_loss:.6f}"
                log_str += f" | LR: {current_lr:.2e} | {epoch_time:.2f}s"
                print(log_str)

            if not save_best_only and self.checkpoint_dir and epoch % save_freq == 0:
                self.save_checkpoint(epoch, val_loss)

            if patience_counter >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - start_time
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "lrs": self.lrs,
            "epoch_times": self.epoch_times,
            "best_val_loss": self.best_val_loss if val_loader else None,
            "best_epoch": self.best_epoch,
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
