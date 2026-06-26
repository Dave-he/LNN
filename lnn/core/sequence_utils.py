from __future__ import annotations

from typing import Any

import torch


def select_step_delta(
    dt: Any,
    step: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float | torch.Tensor:
    """Return the delta-t value for one recurrent step.

    Accepted shapes:
    - scalar / 0-d tensor: shared by every sample and time step
    - [T]: shared per-step schedule
    - [B]: per-sample value shared across the sequence
    - [B, T] or [B, T, 1]: per-sample, per-step deltas
    """
    if dt is None:
        return 1.0
    if isinstance(dt, int | float):
        return float(max(dt, 0.0))

    if not torch.is_tensor(dt):
        dt = torch.as_tensor(dt)
    dt = dt.to(device=device, dtype=dtype)

    if dt.dim() == 0:
        return dt.clamp_min(0.0)

    if dt.dim() == 1:
        if dt.shape[0] == seq_len:
            step_dt = dt[step].reshape(1).expand(batch_size)
        elif dt.shape[0] == batch_size:
            step_dt = dt
        elif dt.numel() == 1:
            step_dt = dt.reshape(1).expand(batch_size)
        else:
            raise ValueError(
                "dt with shape [N] must match seq_len, batch_size, or contain one value; "
                f"got {tuple(dt.shape)} for batch_size={batch_size}, seq_len={seq_len}"
            )
    elif dt.dim() == 2:
        if dt.shape == (batch_size, seq_len):
            step_dt = dt[:, step]
        elif dt.shape[0] == batch_size and dt.shape[1] == 1:
            step_dt = dt[:, 0]
        elif dt.shape[0] == seq_len and dt.shape[1] == 1:
            step_dt = dt[step].expand(batch_size)
        else:
            raise ValueError(
                "dt with shape [N, M] must be [batch, time], [batch, 1], or [time, 1]; "
                f"got {tuple(dt.shape)} for batch_size={batch_size}, seq_len={seq_len}"
            )
    elif dt.dim() == 3:
        if dt.shape[0] != batch_size or dt.shape[1] != seq_len:
            raise ValueError(
                "dt with shape [B, T, D] must match batch and sequence dimensions; "
                f"got {tuple(dt.shape)} for batch_size={batch_size}, seq_len={seq_len}"
            )
        step_dt = dt[:, step, :]
    else:
        raise ValueError(f"dt must be scalar or have 1-3 dimensions, got {dt.dim()}")

    if step_dt.dim() == 0:
        return step_dt.clamp_min(0.0)
    if step_dt.dim() == 1:
        step_dt = step_dt.unsqueeze(-1)
    return step_dt.clamp_min(0.0)


def select_step_mask(
    mask: Any,
    step: int,
    batch_size: int,
    seq_len: int,
    input_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return an input feature mask and a hidden-state update mask.

    `input_mask` has shape [B, input_size] and zeros missing input features.
    `update_mask` has shape [B, 1] and freezes hidden state on fully masked
    or padded time steps.
    """
    if mask is None:
        return None, None

    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.to(device=device)

    if mask.dim() == 1:
        if mask.shape[0] == seq_len:
            input_mask = mask[step].reshape(1, 1).expand(batch_size, input_size)
        elif mask.shape[0] == batch_size:
            input_mask = mask.reshape(batch_size, 1).expand(batch_size, input_size)
        elif mask.numel() == 1:
            input_mask = mask.reshape(1, 1).expand(batch_size, input_size)
        else:
            raise ValueError(
                "mask with shape [N] must match seq_len, batch_size, or contain one value; "
                f"got {tuple(mask.shape)} for batch_size={batch_size}, seq_len={seq_len}"
            )
    elif mask.dim() == 2:
        if mask.shape == (batch_size, seq_len):
            input_mask = mask[:, step].unsqueeze(-1).expand(batch_size, input_size)
        elif mask.shape == (seq_len, input_size):
            input_mask = mask[step].unsqueeze(0).expand(batch_size, input_size)
        elif mask.shape == (batch_size, input_size):
            input_mask = mask
        elif mask.shape[0] == seq_len and mask.shape[1] == 1:
            input_mask = mask[step].reshape(1, 1).expand(batch_size, input_size)
        elif mask.shape[0] == batch_size and mask.shape[1] == 1:
            input_mask = mask.expand(batch_size, input_size)
        else:
            raise ValueError(
                "mask with shape [N, M] must be [batch, time], [time, features], "
                "[batch, features], [time, 1], or [batch, 1]; "
                f"got {tuple(mask.shape)} for batch_size={batch_size}, seq_len={seq_len}, input_size={input_size}"
            )
    elif mask.dim() == 3:
        if mask.shape[0] != batch_size or mask.shape[1] != seq_len:
            raise ValueError(
                "mask with shape [B, T, F] must match batch and sequence dimensions; "
                f"got {tuple(mask.shape)} for batch_size={batch_size}, seq_len={seq_len}"
            )
        input_mask = mask[:, step, :]
        if input_mask.shape[-1] == 1:
            input_mask = input_mask.expand(batch_size, input_size)
        elif input_mask.shape[-1] != input_size:
            raise ValueError(
                "mask feature dimension must be 1 or match input_size; "
                f"got {input_mask.shape[-1]} for input_size={input_size}"
            )
    else:
        raise ValueError(f"mask must have 1-3 dimensions, got {mask.dim()}")

    input_mask = (input_mask > 0).to(dtype=dtype, device=device)
    # `input_mask` is already a 0/1 tensor, so reuse it directly for the
    # update mask — saves one full boolean comparison per step.
    update_mask = input_mask.any(dim=-1, keepdim=True).to(dtype=dtype)
    return input_mask, update_mask
