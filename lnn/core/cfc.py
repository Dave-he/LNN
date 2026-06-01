import torch
import torch.nn as nn

from lnn.core.sequence_utils import select_step_delta, select_step_mask


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) cell.

    Implements the closed-form solution to the LTC ODE:
        x(t) = sigma(-f(x,I;theta_f) * t) * g(x,I;theta_g)
             + [1 - sigma(-f(x,I;theta_f) * t)] * h(x,I;theta_h)

    Key advantage: No ODE solver needed, making it much faster
    than LTC while preserving the continuous-time dynamics.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.f_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.g_branch = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.h_branch = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.time_scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt: float | torch.Tensor = 1.0) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        f = self.f_gate(combined)
        g = self.g_branch(combined)
        h_out = self.h_branch(combined)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        h_new = decay * g + (1.0 - decay) * h_out
        return h_new


class CfCNetwork(nn.Module):
    """
    Full CfC network for sequence processing.

    The CfC network replaces the ODE solver with a closed-form
    approximation, achieving orders of magnitude speedup over LTC
    while maintaining comparable performance.

    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state
        output_size: Dimension of output
        num_layers: Number of stacked CfC layers
        return_sequences: Whether to return full sequence or last step
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(CfCCell(in_dim, hidden_size))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Process a batch of sequences.

        Args:
            x: Input tensor with shape [batch, time, features].
            h0: Optional initial hidden state [layers, batch, hidden].
            dt: Optional per-step time deltas. Supports scalar, [T], [B],
                [B, T], or [B, T, 1] shapes.
            mask: Optional observed-feature or sequence mask. Supports [B, T],
                [B, T, features], [T], or [T, features]. Missing input values
                are zeroed and fully masked steps keep the previous hidden state.
        """
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, self.input_size, x.device, x.dtype
                )
                x_t = torch.nan_to_num(layer_input[:, t, :])
                if i == 0 and input_mask is not None:
                    x_t = x_t * input_mask
                h_candidate = cell(x_t, h_i, dt=dt_t)
                h_i = h_candidate if update_mask is None else update_mask * h_candidate + (1.0 - update_mask) * h_i
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat(
                [h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)],
                dim=0,
            )

        if self.return_sequences:
            return self.output_proj(layer_input)
        else:
            return self.output_proj(layer_input[:, -1, :])
