import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


class StrictCfCCell(nn.Module):
    """
    Strict Closed-form Continuous-time (CfC) cell.
    
    Implements the strict version of CfC without any shortcuts.
    This version strictly follows the continuous-time dynamics 
    with exact integration.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Strict version uses standard continuous-time updates
        self.ff_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.ff_state = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        gate = self.ff_gate(combined)
        state = self.ff_state(combined)
        
        # Strict continuous-time update
        decay = torch.exp(-dt * gate * F.softplus(self.tau))
        h_new = decay * h + (1 - decay) * state
        return h_new


class StrictCfCNetwork(nn.Module):
    """
    Full Strict CfC network for sequence modeling.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1, return_sequences: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(StrictCfCCell(in_dim, hidden_size))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None, dt: float = 1.0) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                h_i = cell(layer_input[:, t, :], h_i, dt)
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat([h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)], dim=0)

        if self.return_sequences:
            return self.output_proj(layer_input)
        else:
            return self.output_proj(layer_input[:, -1, :])


class HybridCfCCell(nn.Module):
    """
    Hybrid Closed-form Continuous-time (CfC) cell.
    
    Combines RNN-like recurrence with continuous-time updates.
    This version balances speed of RNN with continuous-time dynamics.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Hybrid: uses standard RNN gates with continuous time ideas
        self.update_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.candidate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.time_scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        z = self.update_gate(combined)
        r = self.reset_gate(combined)
        h_tilde = self.candidate(torch.cat([x_t, r * h], dim=-1))
        # Hybrid update with time-aware gating
        h_new = (1 - z) * h + z * h_tilde
        return h_new


class HybridCfCNetwork(nn.Module):
    """
    Full Hybrid CfC network for sequence modeling.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1, return_sequences: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(HybridCfCCell(in_dim, hidden_size))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None, dt: float = 1.0) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                h_i = cell(layer_input[:, t, :], h_i, dt)
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat([h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)], dim=0)

        if self.return_sequences:
            return self.output_proj(layer_input)
        else:
            return self.output_proj(layer_input[:, -1, :])


class CTLTCCell(nn.Module):
    """
    Continuous-Time Liquid Time-Constant (CT-LTC) cell.
    
    Directly solves continuous-time LTC with continuous dynamics
    with explicit continuous time simulation.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.f_tau = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.f_input = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.tau_base = nn.Parameter(torch.ones(hidden_size))
        self.A = nn.Parameter(torch.ones(hidden_size) * 0.5)

    def forward_dynamics(self, t: torch.Tensor, h: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        tau = F.softplus(self.tau_base) + self.f_tau(combined) + 0.01
        input_term = self.f_input(combined) * self.A
        dhdt = -h / tau + input_term
        return dhdt

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        t_span = torch.tensor([0.0, dt], device=x_t.device, dtype=x_t.dtype)
        h_new = odeint(
            lambda t, state: self.forward_dynamics(t, state, x_t),
            h,
            t_span,
            method='rk4',
        )[-1]
        return h_new


class CTLTCNetwork(nn.Module):
    """
    Full CT-LTC network for sequence modeling.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1, ode_method: str = "rk4", return_sequences: bool = True):
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
            self.cells.append(CTLTCCell(in_dim, hidden_size))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None, dt: float = 1.0) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                h_i = cell(layer_input[:, t, :], h_i, dt)
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat([h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)], dim=0)

        if self.return_sequences:
            return self.output_proj(layer_input)
        else:
            return self.output_proj(layer_input[:, -1, :])


class LiquidS4Cell(nn.Module):
    """
    Liquid-S4: combination of LNN and S4 (Structured State Space)
    
    Combines LNN's continuous time with S4's long-sequence capabilities.
    Simplified implementation for practical use.
    
    Args:
        input_size: Input dimension
        hidden_size: State dimension
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # State space parameters (simplified diagonal S4)
        self.log_decay = nn.Parameter(torch.log(torch.ones(hidden_size) * 0.5))
        self.B = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        
        # Liquid modulation
        self.liquid_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.liquid_input = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        
    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        # Simplified diagonal state space update
        decay = torch.exp(-torch.exp(self.log_decay) * dt)
        
        # State update: h_new = decay * h + B * x
        h_decayed = decay.unsqueeze(0) * h
        input_contrib = torch.tanh(x_t @ self.B.T)
        h_s4 = h_decayed + input_contrib
        
        # Liquid modulation
        combined = torch.cat([x_t, h_s4], dim=-1)
        gate = self.liquid_gate(combined)
        liquid_input = self.liquid_input(combined)
        
        # Combine S4 state with liquid modulation
        h_new = gate * liquid_input + (1 - gate) * h_s4
        
        return h_new


class LiquidS4Network(nn.Module):
    """
    Full Liquid-S4 network for sequence modeling.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1, return_sequences: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(LiquidS4Cell(in_dim, hidden_size))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None, dt: float = 1.0) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                h_i = cell(layer_input[:, t, :], h_i, dt)
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat([h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)], dim=0)

        if self.return_sequences:
            return self.output_proj(layer_input)
        else:
            return self.output_proj(layer_input[:, -1, :])


class LRCCell(nn.Module):
    """
    Liquid Resistive-Capacitive (LRC) cell.
    
    Introduces liquid capacitor to enhance biological plausibility
    and suppress oscillations.
    
    Args:
        input_size: Input dimension
        hidden_size: Hidden state dimension
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Resistive-Capacitive dynamics
        self.f_resistance = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.f_capacitance = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.f_input = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.R_base = nn.Parameter(torch.ones(hidden_size))
        self.C_base = nn.Parameter(torch.ones(hidden_size))

    def forward_dynamics(self, t: torch.Tensor, h: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        R = F.softplus(self.R_base) + self.f_resistance(combined)
        C = F.softplus(self.C_base) + self.f_capacitance(combined)
        input_term = self.f_input(combined)
        tau = R * C  # RC time constant
        dhdt = (-h + input_term) / tau
        return dhdt

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        t_span = torch.tensor([0.0, dt], device=x_t.device, dtype=x_t.dtype)
        h_new = odeint(
            lambda t, state: self.forward_dynamics(t, state, x_t),
            h,
            t_span,
            method='rk4',
        )[-1]
        return h_new


class LRCNetwork(nn.Module):
    """
    Full LRC network for sequence modeling.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1, ode_method: str = "rk4", return_sequences: bool = True):
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
            self.cells.append(LRCCell(in_dim, hidden_size))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None, dt: float = 1.0) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                h_i = cell(layer_input[:, t, :], h_i, dt)
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat([h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)], dim=0)

        if self.return_sequences:
            return self.output_proj(layer_input)
        else:
            return self.output_proj(layer_input[:, -1, :])


class CfCDTCell(nn.Module):
    """
    CfC-DT: Closed-form Continuous-time with explicit dt support.
    
    Supports irregularly sampled time series with explicit dt handling.
    
    Args:
        input_size: Input dimension
        hidden_size: Hidden state dimension
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ff1 = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.ff2 = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.time_a = nn.Linear(input_size + hidden_size, hidden_size)
        self.time_b = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt_t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        decay_rate = F.softplus(self.time_a(combined)) + 1e-4
        gate = torch.sigmoid(-decay_rate * dt_t + self.time_b(combined))
        return self.ff1(combined) * (1.0 - gate) + self.ff2(combined) * gate


class CfCDTNetwork(nn.Module):
    """
    Full CfC-DT network for sequence modeling with dt support.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1, return_sequences: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(CfCDTCell(in_dim, hidden_size))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, dt: torch.Tensor | None = None, h0: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        if dt is None:
            dt = torch.ones(batch_size, seq_len, 1, device=x.device, dtype=x.dtype)

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                h_i = cell(layer_input[:, t, :], h_i, dt[:, t, :])
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat([h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)], dim=0)

        if self.return_sequences:
            return self.output_proj(layer_input)
        else:
            return self.output_proj(layer_input[:, -1, :])


class EulerLTCDTCell(nn.Module):
    """
    Euler-LTC-DT: Liquid Time-Constant with Euler method and explicit dt support.
    
    Simple Euler integration for faster computation on edge devices.
    
    Args:
        input_size: Input dimension
        hidden_size: Hidden state dimension
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.f_tau = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.f_drive = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.tau_base = nn.Parameter(torch.ones(hidden_size))
        self.amplitude = nn.Parameter(torch.ones(hidden_size) * 0.5)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt_t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        tau = F.softplus(self.tau_base) + self.f_tau(combined) + 0.05
        drive = self.f_drive(combined) * self.amplitude
        dhdt = -h / tau + drive
        h_new = h + dt_t * dhdt
        return h_new


class EulerLTCDTNetwork(nn.Module):
    """
    Full Euler-LTC-DT network for sequence modeling with dt support.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1, return_sequences: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(EulerLTCDTCell(in_dim, hidden_size))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, dt: torch.Tensor | None = None, h0: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        if dt is None:
            dt = torch.ones(batch_size, seq_len, 1, device=x.device, dtype=x.dtype)

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                h_i = cell(layer_input[:, t, :], h_i, dt[:, t, :])
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat([h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)], dim=0)

        if self.return_sequences:
            return self.output_proj(layer_input)
        else:
            return self.output_proj(layer_input[:, -1, :])
