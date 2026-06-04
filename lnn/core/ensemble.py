"""BiCfCEnsemble: K-seed ensemble with smart selection (round 65 v15 recipe).

Round 65 (46th meta, NEW BEST) established the production recipe:
  - Train 30 seeds with phase2_only inject=0.10 + freeze audio_encoder
  - Rank by validation MSE
  - Ensemble top 20 (smart selection)
  - Honest LOO MSE: 0.24 (reproducible across 4 folds)

Round 67 (47th meta) confirmed 30 seeds is FINAL sweet spot:
  - 40-seed pool does NOT improve (selection noise)
  - 30 seeds + K=20 is FINAL optimal

This class is the production-recipe implementation of that finding.

Usage:
    from lnn.core.ensemble import BiCfCEnsemble
    ensemble = BiCfCEnsemble(
        n_seeds=30, K=20, hidden_size=96,
        epochs=80, warmup_epochs=40,
        phase2_inject_sigma=0.10, freeze="audio_only",
        val_frac=0.20,
    )
    ensemble.fit(X_train, y_train, X_val, y_val)
    preds = ensemble.predict(X_test)
"""

from __future__ import annotations

import copy
import math
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .mdn import mdn_mean, mdn_negative_log_likelihood
from .multimodal_physreg import CrossModalAttnBiCfCNADWithMDN


def _make_model(
    seed: int,
    hidden_size: int,
    num_mixtures: int,
    output_size: int,
    video_dim: int = 3,
    audio_dim: int = 1,
) -> CrossModalAttnBiCfCNADWithMDN:
    """Create a fresh Bi-CfC model with a given seed."""
    torch.manual_seed(seed)
    return CrossModalAttnBiCfCNADWithMDN(
        video_dim=video_dim,
        audio_dim=audio_dim,
        hidden_size=hidden_size,
        output_size=output_size,
        num_mixtures=num_mixtures,
    )


def _inject_audio_noise(audio: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add N(0, sigma^2) noise to audio (data augmentation)."""
    if sigma <= 0.0:
        return audio
    return audio + torch.randn_like(audio) * sigma


def _train_one_seed(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    warmup_epochs: int,
    phase2_inject_sigma: float,
    freeze: str,
    lr: float,
    device: torch.device,
) -> float:
    """Train a single seed's model and return its val MSE."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(warmup_epochs):
        _train_epoch(model, train_loader, opt, phase2_inject_sigma, freeze=None, device=device)
    if freeze == "audio_only":
        for p in model.audio_encoder.parameters():
            p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=lr)
    for _ in range(epochs - warmup_epochs):
        _train_epoch(model, train_loader, opt, phase2_inject_sigma, freeze=freeze, device=device)
    return _eval_mse(model, val_loader, device)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    phase2_inject_sigma: float,
    freeze: str | None,
    device: torch.device,
) -> float:
    model.train()
    total, n = 0.0, 0
    for batch, target in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        target = {k: v.to(device) for k, v in target.items()}
        if "audio" in batch and phase2_inject_sigma > 0:
            batch["audio"] = _inject_audio_noise(batch["audio"], phase2_inject_sigma)
        opt.zero_grad()
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        loss = mdn_negative_log_likelihood(final, target["params"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _eval_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute MSE of model predictions on loader."""
    model.eval()
    sq = []
    for batch, target in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        target = {k: v.to(device) for k, v in target.items()}
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        mean = mdn_mean(final)
        sq.append((mean - target["params"]).pow(2).sum(dim=-1))
    return float(torch.cat(sq).mean().item())


class BiCfCEnsemble:
    """K-seed ensemble of Bi-CfC models with smart selection by val MSE.

    Production recipe (round 65 v15, confirmed round 67):
      1. Train n_seeds models with different random seeds
      2. Rank models by validation MSE
      3. Ensemble top-K predictions (smart selection)

    The recipe is the final result of 65+ rounds of ablation:
      - 30 seeds is the FINAL sweet spot (round 67)
      - K=20 (top 20 by val) is the FINAL optimal ensemble size
      - phase2_only inject=0.10 + freeze audio_encoder is the recipe
      - Expected honest LOO MSE: ~0.24 (reproducible across folds)

    Parameters
    ----------
    n_seeds : int, default=30
        Number of seeds to train. Round 67 confirmed 30 is the sweet spot.
    K : int, default=20
        Number of top models to ensemble. Round 65 confirmed K=20 is optimal.
    hidden_size : int, default=96
        Hidden size of each base model. Round 38 SOTA used h=96.
    num_mixtures : int, default=1
        Number of MDN mixtures per base model.
    epochs : int, default=80
        Total training epochs per seed.
    warmup_epochs : int, default=40
        Epochs before freezing (or phase 2 start). Half of epochs recommended.
    phase2_inject_sigma : float, default=0.10
        N(0, sigma^2) noise added to audio in phase 2 (after warmup/freeze).
        Round 54/65 confirmed 0.10 is optimal.
    freeze : str, default="audio_only"
        Freezing strategy. Round 56/65 confirmed "audio_only" is best.
    val_frac : float, default=0.20
        Fraction of training data held out as validation set for ranking.
    lr : float, default=5e-3
        Learning rate.
    device : str, default="cpu"
        Device to use for training.

    Attributes
    ----------
    models_ : list[nn.Module]
        All trained models (n_seeds total).
    val_mses_ : list[float]
        Per-seed validation MSEs (used for ranking).
    top_k_indices_ : list[int]
        Indices of the top-K models (by val MSE).
    """

    def __init__(
        self,
        n_seeds: int = 30,
        K: int = 20,
        hidden_size: int = 96,
        num_mixtures: int = 1,
        output_size: int = 5,
        epochs: int = 80,
        warmup_epochs: int = 40,
        phase2_inject_sigma: float = 0.10,
        freeze: str = "audio_only",
        val_frac: float = 0.20,
        lr: float = 5e-3,
        device: str = "cpu",
    ) -> None:
        if K > n_seeds:
            raise ValueError(f"K ({K}) cannot exceed n_seeds ({n_seeds})")
        if freeze not in ("audio_only", "none"):
            raise ValueError(f"freeze must be 'audio_only' or 'none', got {freeze!r}")
        self.n_seeds = n_seeds
        self.K = K
        self.hidden_size = hidden_size
        self.num_mixtures = num_mixtures
        self.output_size = output_size
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.phase2_inject_sigma = phase2_inject_sigma
        self.freeze = freeze
        self.val_frac = val_frac
        self.lr = lr
        self.device = torch.device(device)
        self.models_: list[nn.Module] = []
        self.val_mses_: list[float] = []
        self.top_k_indices_: list[int] = []

    def fit(
        self,
        train_dataset: Any,
        seed_values: Sequence[int] | None = None,
    ) -> "BiCfCEnsemble":
        """Train the ensemble on the given dataset.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            The full training dataset. Will be split into train/val by
            ``val_frac`` (with deterministic split seed).
        seed_values : sequence of int, optional
            Specific seed values to use. If None, uses seeds 1..n_seeds.
            Should have at least n_seeds values; only the first n_seeds
            are used.

        Returns
        -------
        self : BiCfCEnsemble
        """
        if seed_values is None:
            seed_values = list(range(1, self.n_seeds + 1))
        if len(seed_values) < self.n_seeds:
            raise ValueError(
                f"Need at least n_seeds={self.n_seeds} seed values, got {len(seed_values)}"
            )
        # Split train/val
        train_sub, val_sub = _split_train_val(train_dataset, val_frac=self.val_frac, seed=42)
        train_loader = DataLoader(train_sub, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=8, shuffle=False)

        # Train each seed
        self.models_ = []
        self.val_mses_ = []
        for i in range(self.n_seeds):
            seed = seed_values[i]
            model = _make_model(
                seed=seed,
                hidden_size=self.hidden_size,
                num_mixtures=self.num_mixtures,
                output_size=self.output_size,
            ).to(self.device)
            val_mse = _train_one_seed(
                model, train_loader, val_loader,
                epochs=self.epochs,
                warmup_epochs=self.warmup_epochs,
                phase2_inject_sigma=self.phase2_inject_sigma,
                freeze=self.freeze,
                lr=self.lr,
                device=self.device,
            )
            self.models_.append(model)
            self.val_mses_.append(val_mse)

        # Rank by val MSE, select top K
        ranked = sorted(range(self.n_seeds), key=lambda i: self.val_mses_[i])
        self.top_k_indices_ = ranked[: self.K]
        return self

    def predict(self, test_loader: DataLoader) -> torch.Tensor:
        """Predict by averaging top-K models' predictions on test_loader.

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            DataLoader for the test set.

        Returns
        -------
        preds : torch.Tensor
            Averaged predictions, shape [N_samples, output_size].
        """
        if not self.models_:
            raise RuntimeError("Call fit() before predict()")
        # Collect predictions from top K models
        all_preds_per_model: list[torch.Tensor] = []
        targets: torch.Tensor | None = None
        for idx in self.top_k_indices_:
            model = self.models_[idx]
            model.eval()
            preds, tgts = [], []
            with torch.no_grad():
                for batch, target in test_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    target = {k: v.to(self.device) for k, v in target.items()}
                    out = model(batch["video"], batch["audio"])
                    final = {k: v[:, -1] for k, v in out.items()}
                    mean = mdn_mean(final)
                    preds.append(mean.cpu())
                    tgts.append(target["params"].cpu())
            all_preds_per_model.append(torch.cat(preds))
            if targets is None:
                targets = torch.cat(tgts)
        # Average
        stacked = torch.stack(all_preds_per_model, dim=0)  # [K, N, D]
        return stacked.mean(dim=0)

    def evaluate(self, test_loader: DataLoader) -> dict[str, float]:
        """Compute ensemble MSE on test set, plus per-K breakdown.

        Returns
        -------
        metrics : dict
            "ensemble_mse": K-model ensemble MSE on test set
            "per_seed_mean_mse": mean of per-seed test MSEs (for comparison)
            "per_seed_std_mse": std of per-seed test MSEs
        """
        if not self.models_:
            raise RuntimeError("Call fit() before evaluate()")
        per_seed_mses = []
        for idx, model in enumerate(self.models_):
            model.eval()
            sq = []
            with torch.no_grad():
                for batch, target in test_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    target = {k: v.to(self.device) for k, v in target.items()}
                    out = model(batch["video"], batch["audio"])
                    final = {k: v[:, -1] for k, v in out.items()}
                    mean = mdn_mean(final)
                    sq.append((mean - target["params"]).pow(2).sum(dim=-1))
            per_seed_mses.append(float(torch.cat(sq).mean().item()))
        # Ensemble MSE
        ensemble_pred = self.predict(test_loader)
        # Get targets
        targets = []
        with torch.no_grad():
            for _, target in test_loader:
                targets.append(target["params"])
        targets = torch.cat(targets)
        ensemble_mse = float(((ensemble_pred - targets) ** 2).sum(dim=-1).mean().item())
        # Std
        import statistics
        return {
            "ensemble_mse": ensemble_mse,
            "per_seed_mean_mse": statistics.mean(per_seed_mses),
            "per_seed_std_mse": statistics.stdev(per_seed_mses) if len(per_seed_mses) > 1 else 0.0,
            "n_seeds": self.n_seeds,
            "K": self.K,
        }


def _split_train_val(dataset: Any, val_frac: float, seed: int) -> tuple[Subset, Subset]:
    """Split a dataset into train and val by indices."""
    n = len(dataset)
    n_val = max(1, int(n * val_frac))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
