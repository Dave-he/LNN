import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LSTMModel(nn.Module):
    """
    Standard Single-layer LSTM baseline.
    Insensitive to temporal spacing or continuous-time dynamics.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 1)

        # Initialize parameters
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x: torch.Tensor, dt: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_size]
        out, _ = self.lstm(x)
        # Take the final hidden state h_L
        h_L = out[:, -1, :]
        return self.readout(h_L)


class StrictCfCModel(nn.Module):
    """
    Strict CfC cell with gated interpolation form:
        h_t = g_t * (1 - sigma_t) + sigma_t * f_t
    Using a shared backbone MLP.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Backbone dimension m = max(2 * hidden_size, 32)
        self.m = max(2 * hidden_size, 32)
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, self.m),
            nn.Tanh()
        )
        
        # Double candidate trajectory limit heads
        self.g_head = nn.Sequential(
            nn.Linear(self.m, hidden_size),
            nn.Tanh()
        )
        self.f_head = nn.Sequential(
            nn.Linear(self.m, hidden_size),
            nn.Tanh()
        )
        
        # Double linear interpolation gate heads
        self.sig_a = nn.Linear(self.m, hidden_size)
        self.sig_b = nn.Linear(self.m, hidden_size)
        
        self.readout = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, dt: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_size)
        
        # Strict CfC has constant unit step size dt_t = 1.0 by default
        step_dt = 1.0 if dt is None else dt
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            u_t = torch.cat([x_t, h], dim=-1)
            s_t = self.backbone(u_t)
            
            gt = self.g_head(s_t)
            ft = self.f_head(s_t)
            
            t_a = self.sig_a(s_t)
            t_b = self.sig_b(s_t)
            
            # If dt is provided per-step, we use it, otherwise unit step
            if isinstance(step_dt, torch.Tensor):
                dt_t = step_dt[:, t, :] if step_dt.dim() == 3 else step_dt[:, t].unsqueeze(-1)
            else:
                dt_t = step_dt
                
            sigma_t = torch.sigmoid(t_a * dt_t + t_b)
            h = gt * (1.0 - sigma_t) + sigma_t * ft
            
        return self.readout(h)


class LTCModel(nn.Module):
    """
    Liquid Time-Constant (LTC) network with fused semi-implicit Euler solver (L_ode = 6).
    """
    def __init__(self, input_size: int, hidden_size: int, l_ode: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.l_ode = l_ode

        # projection computed once per external step
        self.proj_in = nn.Linear(input_size, hidden_size)
        # recurrent weights recomputed per sub-step
        self.w_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        # Learnable parameters in log space for positivity
        self.theta_tau = nn.Parameter(torch.ones(hidden_size) * 0.0)  # tau = exp(theta_tau)
        self.A = nn.Parameter(torch.ones(hidden_size) * 0.5)         # attractor

        self.readout = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.w_rec.weight)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x: torch.Tensor, dt: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_size)
        
        # Step size
        tau = torch.exp(self.theta_tau)
        rec_delta = 1.0 / self.l_ode

        # Loop through sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            x_proj = self.proj_in(x_t) + self.bias
            
            # Semi-implicit Euler integration (L_ode steps)
            for _ in range(self.l_ode):
                f_ell = torch.sigmoid(self.w_rec(h) + x_proj)
                numerator = h + rec_delta * (f_ell * self.A)
                denominator = 1.0 + rec_delta * (1.0 / (tau + 1e-6) + f_ell)
                h = numerator / denominator

        return self.readout(h)


class HybridCfCModel(nn.Module):
    """
    Hybrid CfC model featuring input-conditioned timescale decay modulation.
    At = 1 / tau_t + |f_t|
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.ff = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_tau = nn.Linear(input_size + hidden_size, hidden_size)
        self.theta_tau = nn.Parameter(torch.zeros(hidden_size))

        self.readout = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ff.weight)
        nn.init.zeros_(self.ff.bias)
        nn.init.xavier_uniform_(self.w_tau.weight)
        nn.init.zeros_(self.w_tau.bias)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x: torch.Tensor, dt: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_size)
        
        step_dt = 1.0 if dt is None else dt

        for t in range(seq_len):
            x_t = x[:, t, :]
            u_t = torch.cat([x_t, h], dim=-1)
            
            f_t = torch.tanh(self.ff(u_t))
            tau_t = torch.exp(self.theta_tau) * torch.sigmoid(self.w_tau(u_t))
            tau_t = torch.clamp(tau_t, min=1e-3)
            
            A_t = 1.0 / tau_t + torch.abs(f_t)
            
            if isinstance(step_dt, torch.Tensor):
                dt_t = step_dt[:, t, :] if step_dt.dim() == 3 else step_dt[:, t].unsqueeze(-1)
            else:
                dt_t = step_dt
                
            g_t = torch.exp(-torch.clamp(A_t, min=1e-3) * dt_t)
            h = g_t * h + (1.0 - g_t) * (f_t / (A_t + 1e-6))

        return self.readout(h)


class CTLTCModel(nn.Module):
    """
    Calendar-gap Continuous-Time LTC (CT-LTC).
    Uses observed calendar gap delta_t directly in semi-implicit Euler integration steps.
    """
    def __init__(self, input_size: int, hidden_size: int, l_ode: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.l_ode = l_ode

        # projection computed once per external step
        self.proj_in = nn.Linear(input_size, hidden_size)
        # recurrent weights recomputed per sub-step
        self.w_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        self.theta_tau = nn.Parameter(torch.ones(hidden_size) * 0.0)
        self.A = nn.Parameter(torch.ones(hidden_size) * 0.5)

        self.readout = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.w_rec.weight)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        # dt shape: [batch_size, seq_len] or [batch_size, seq_len, 1]
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_size)
        
        tau = torch.exp(self.theta_tau)
        
        # Ensure calendar gap dt is 3D for matrix operations
        if dt.dim() == 2:
            dt = dt.unsqueeze(-1)

        # Loop through sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            x_proj = self.proj_in(x_t) + self.bias
            
            # Observe specific calendar gap delta_t for current step
            dt_t = dt[:, t, :]
            
            # Sub-step size dynamically scaled by delta_t
            delta_t_ell = dt_t / self.l_ode
            
            for _ in range(self.l_ode):
                f_ell = torch.sigmoid(self.w_rec(h) + x_proj)
                numerator = h + delta_t_ell * (f_ell * self.A)
                denominator = 1.0 + delta_t_ell * (1.0 / (tau + 1e-6) + f_ell)
                h = numerator / denominator

        return self.readout(h)


# =====================================================================
# 🚀 优化策略 1: 多尺度通道自适应液态网络 (Multi-Scale Hybrid CfC, MS-CfC)
# =====================================================================

class MSCfCModel(nn.Module):
    """
    Multi-Scale Gated Hybrid CfC (MS-CfC).
    Splits hidden units into fast, medium, and slow timescales,
    allowing different neural channels to track different market dynamic ranges.
    Additionally, learns an adaptive channel-attention weighting conditioned on inputs.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.ff = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_tau = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Initialize multi-scale base time constants
        # 1/3 fast (decay fast, instant reaction), 1/3 standard, 1/3 slow (long-term trend)
        theta_init = torch.zeros(hidden_size)
        part = max(1, hidden_size // 3)
        
        # Fast timescale channels: tau base ~ 0.05 (log-space: -3.0)
        theta_init[:part] = -3.0
        # Medium channels: tau base ~ 1.0 (log-space: 0.0)
        theta_init[part:2*part] = 0.0
        # Slow channels: tau base ~ 20.0 (log-space: 3.0)
        theta_init[2*part:] = 3.0
        
        self.theta_tau = nn.Parameter(theta_init)

        # Dynamic timescale scaling attention based on local volatility/inputs
        self.vol_attention = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )

        self.readout = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ff.weight)
        nn.init.zeros_(self.ff.bias)
        nn.init.xavier_uniform_(self.w_tau.weight)
        nn.init.zeros_(self.w_tau.bias)
        nn.init.xavier_uniform_(self.vol_attention[0].weight)
        nn.init.zeros_(self.vol_attention[0].bias)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x: torch.Tensor, dt: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_size)
        
        step_dt = 1.0 if dt is None else dt

        for t in range(seq_len):
            x_t = x[:, t, :]
            u_t = torch.cat([x_t, h], dim=-1)
            
            f_t = torch.tanh(self.ff(u_t))
            
            # Input-conditioned time constant modification
            tau_t = torch.exp(self.theta_tau) * torch.sigmoid(self.w_tau(u_t))
            tau_t = torch.clamp(tau_t, min=1e-3)
            
            # Channel Volatility Gated Attention (scale time constant dynamics)
            gate_attn = self.vol_attention(u_t)
            
            # If input is highly volatile, scale down tau (accelerate decay) for fast channels,
            # and hold slow channels stable
            adjusted_tau = tau_t * (2.0 * gate_attn + 0.1)
            
            A_t = 1.0 / adjusted_tau + torch.abs(f_t)
            
            if isinstance(step_dt, torch.Tensor):
                dt_t = step_dt[:, t, :] if step_dt.dim() == 3 else step_dt[:, t].unsqueeze(-1)
            else:
                dt_t = step_dt
                
            g_t = torch.exp(-torch.clamp(A_t, min=1e-3) * dt_t)
            h = g_t * h + (1.0 - g_t) * (f_t / (A_t + 1e-6))

        return self.readout(h)


# =====================================================================
# 🚀 优化策略 2: 波动率加权 MSE 损失函数 (Volatility-Weighted Loss)
# =====================================================================

class VolatilityWeightedMSELoss(nn.Module):
    """
    Volatility-Weighted MSE Loss.
    Weighs training errors by the rolling volatility.
    Forces the liquid network to prioritize learning state transitions during high-volatility regime shifts.
    """
    def __init__(self, gamma: float = 1.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor, rolling_vol: torch.Tensor) -> torch.Tensor:
        # pred, target: [batch_size, 1]
        # rolling_vol: [batch_size, 1]
        
        # Calculate weights
        weights = 1.0 + self.gamma * torch.abs(rolling_vol)
        
        # Weighted square error
        squared_errors = (pred - target) ** 2
        weighted_loss = squared_errors * weights
        
        return weighted_loss.mean()
