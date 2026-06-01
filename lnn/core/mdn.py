from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNHead(nn.Module):
    """Mixture Density Network head for continuous multi-modal actions."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_mixtures: int = 5,
        min_log_scale: float = -7.0,
        max_log_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if num_mixtures < 1:
            raise ValueError("num_mixtures must be >= 1")
        self.input_size = input_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures
        self.min_log_scale = min_log_scale
        self.max_log_scale = max_log_scale

        self.logit_proj = nn.Linear(input_size, num_mixtures)
        self.loc_proj = nn.Linear(input_size, num_mixtures * output_size)
        self.log_scale_proj = nn.Linear(input_size, num_mixtures * output_size)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        prefix_shape = features.shape[:-1]
        logits = self.logit_proj(features)
        loc = self.loc_proj(features).view(*prefix_shape, self.num_mixtures, self.output_size)
        log_scale = self.log_scale_proj(features).view(*prefix_shape, self.num_mixtures, self.output_size)
        log_scale = log_scale.clamp(self.min_log_scale, self.max_log_scale)
        return {
            "logits": logits,
            "loc": loc,
            "log_scale": log_scale,
        }


def mdn_negative_log_likelihood(params: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Return mean negative log likelihood under a diagonal Gaussian mixture."""
    logits = params["logits"]
    loc = params["loc"]
    log_scale = params["log_scale"]
    target = target.unsqueeze(-2)

    inv_scale = torch.exp(-log_scale)
    centered = (target - loc) * inv_scale
    log_prob = -0.5 * centered.pow(2) - log_scale - 0.5 * math.log(2.0 * math.pi)
    component_log_prob = log_prob.sum(dim=-1) + F.log_softmax(logits, dim=-1)
    return -torch.logsumexp(component_log_prob, dim=-1).mean()


def mdn_mean(params: dict[str, torch.Tensor]) -> torch.Tensor:
    weights = F.softmax(params["logits"], dim=-1).unsqueeze(-1)
    return (weights * params["loc"]).sum(dim=-2)


def mdn_sample(params: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = params["logits"]
    loc = params["loc"]
    scale = torch.exp(params["log_scale"])
    mixture_index = torch.distributions.Categorical(logits=logits).sample()
    gather_index = mixture_index.unsqueeze(-1).unsqueeze(-1).expand(*mixture_index.shape, 1, loc.shape[-1])
    chosen_loc = loc.gather(dim=-2, index=gather_index).squeeze(-2)
    chosen_scale = scale.gather(dim=-2, index=gather_index).squeeze(-2)
    return chosen_loc + torch.randn_like(chosen_loc) * chosen_scale
