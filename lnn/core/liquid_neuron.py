import torch
import torch.nn as nn
from torchdiffeq import odeint


class LiquidNeuron(nn.Module):
    """
    Liquid Neuron: single neuron with input-dependent dynamic time constant.

    The state evolves according to:
        dh/dt = -h / (tau + f_tau(x, h)) + f_input(x, h)

    where tau is a base time constant and f_tau, f_input are learned
    nonlinear maps that make the time constant input-dependent.
    """

    def __init__(self, input_size: int, hidden_size: int, tau_min: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_min = tau_min

        self.f_tau = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.f_input = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward_dynamics(self, t: torch.Tensor, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = state
        combined = torch.cat([x, h], dim=-1)
        tau = self.tau_min + self.f_tau(combined)
        input_term = self.f_input(combined)
        dhdt = -h / tau + input_term
        return dhdt

    def forward(
        self, x: torch.Tensor, h0: torch.Tensor | None = None, dt: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        outputs = []
        h = h0
        for t in range(seq_len):
            x_t = x[:, t, :]
            t_span = torch.tensor([0.0, dt], device=x.device, dtype=x.dtype)
            h = odeint(lambda t, state: self.forward_dynamics(t, state, x_t), h, t_span, method="euler")[-1]
            outputs.append(h)

        return torch.stack(outputs, dim=1), h


class LiquidLayer(nn.Module):
    """
    A layer of liquid neurons with ODE-based dynamics.

    Uses a single ODE solver call per time step for the entire layer,
    which is more efficient than solving each neuron independently.
    """

    def __init__(self, input_size: int, hidden_size: int, ode_method: str = "euler"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ode_method = ode_method

        self.W_input = nn.Linear(input_size, hidden_size)
        self.W_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau_log = nn.Parameter(torch.zeros(hidden_size))
        self.gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )

    def get_tau(self) -> torch.Tensor:
        return torch.exp(self.tau_log) + 0.1

    def forward_dynamics(self, t: torch.Tensor, h: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        tau = self.get_tau()
        input_contrib = self.W_input(x_t)
        hidden_contrib = self.W_hidden(h)
        combined = torch.cat([x_t, h], dim=-1)
        g = self.gate(combined)
        dhdt = (-h + torch.tanh(input_contrib + hidden_contrib)) * g / tau
        return dhdt

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        outputs = []
        h = h0
        for t in range(seq_len):
            x_t = x[:, t, :]
            t_span = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)
            h = odeint(lambda t, state: self.forward_dynamics(t, state, x_t), h, t_span, method=self.ode_method)[-1]
            outputs.append(h)

        return torch.stack(outputs, dim=1), h


class LiquidNN(nn.Module):
    """
    Complete Liquid Neural Network for sequence modeling.

    Architecture:
        Input -> LiquidLayer(s) -> Output projection

    The network stacks multiple liquid layers, each with its own
    ODE-based dynamics and learnable time constants.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        ode_method: str = "euler",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.layers.append(LiquidLayer(in_dim, hidden_size, ode_method=ode_method))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.size(0)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        h = h0
        for i, layer in enumerate(self.layers):
            x, h_i = layer(x, h0=h[i])
            h = torch.cat([h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)], dim=0)

        return self.output_proj(x)
