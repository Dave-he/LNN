import torch
import torch.nn as nn
from torchdiffeq import odeint


class LTCODEFunc(nn.Module):
    """
    ODE function for Liquid Time-Constant networks.

    Implements the LTC dynamics:
        dx/dt = -[1 / (tau + f(x, I, theta))] * x + f(x, I, theta) * A

    where:
        - tau is a base time constant
        - f(x, I, theta) is a learned nonlinear function
        - A is a learnable bias term
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.f_tau = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.f_bias = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.tau_base = nn.Parameter(torch.ones(hidden_size) * 1.0)
        self.A = nn.Parameter(torch.ones(hidden_size) * 0.5)

    def forward(self, t: torch.Tensor, state: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_t, state], dim=-1)
        f_out = self.f_tau(combined)
        bias = self.f_bias(combined)
        tau_eff = torch.abs(self.tau_base) + f_out + 0.01
        dxdt = -state / tau_eff + bias * self.A
        return dxdt


class LTCCell(nn.Module):
    """
    Single LTC cell that processes one time step using ODE integration.

    This is the fundamental building block of the LTC network.
    Each cell maintains a hidden state that evolves continuously
    according to the LTC ODE dynamics.
    """

    def __init__(self, input_size: int, hidden_size: int, ode_method: str = "rk4"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ode_method = ode_method
        self.ode_func = LTCODEFunc(input_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        t_span = torch.tensor([0.0, dt], device=x_t.device, dtype=x_t.dtype)
        h_new = odeint(
            lambda t, state: self.ode_func(t, state, x_t),
            h,
            t_span,
            method=self.ode_method,
        )[-1]
        return h_new


class LTCNetwork(nn.Module):
    """
    Full LTC network for sequence processing.

    Processes an entire sequence by iteratively applying the LTC cell
    at each time step, with ODE integration between steps.

    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state
        output_size: Dimension of output
        num_layers: Number of stacked LTC layers
        ode_method: ODE solver method ('euler', 'rk4', 'dopri5')
        return_sequences: Whether to return full sequence or last step
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        ode_method: str = "rk4",
        return_sequences: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.ode_method = ode_method
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(LTCCell(in_dim, hidden_size, ode_method=ode_method))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                h_i = cell(layer_input[:, t, :], h_i)
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
