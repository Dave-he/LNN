import torch
import torch.nn as nn

try:
    from ncps.torch import LTC as NCPS_LTC
    from ncps.torch import CfC as NCPS_CfC
    from ncps.wirings import AutoNCP

    NCPS_AVAILABLE = True
except ImportError:
    NCPS_AVAILABLE = False


def _check_ncps():
    if not NCPS_AVAILABLE:
        raise ImportError(
            "ncps library is not installed. Install it with: pip install ncps torchdiffeq"
        )


class NCPSCfC(nn.Module):
    """
    CfC model using the ncps library with AutoNCP wiring.

    This wraps the official ncps CfC implementation, which provides
    production-quality LTC/CfC/NCP with proper ODE handling.

    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        output_size: Number of output features
        return_sequences: Return full sequence or last step
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        return_sequences: bool = True,
    ):
        super().__init__()
        _check_ncps()
        wiring = AutoNCP(hidden_size, output_size)
        self.rnn = NCPS_CfC(input_size, wiring, batch_first=True)
        self.return_sequences = return_sequences
        self.output_size = output_size

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        output, _ = self.rnn(x, h0)
        if self.return_sequences:
            return output
        return output[:, -1, :]


class NCPSLTC(nn.Module):
    """
    LTC model using the ncps library with AutoNCP wiring.

    The LTC model uses an ODE solver for exact dynamics simulation.
    Slower than CfC but provides the most faithful continuous-time dynamics.

    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        output_size: Number of output features
        ode_method: ODE solver method ('rk4', 'euler', 'dopri5')
        return_sequences: Return full sequence or last step
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        ode_method: str = "rk4",
        return_sequences: bool = True,
    ):
        super().__init__()
        _check_ncps()
        wiring = AutoNCP(hidden_size, output_size)
        self.rnn = NCPS_LTC(input_size, wiring, batch_first=True, odeint_backend=ode_method)
        self.return_sequences = return_sequences
        self.output_size = output_size

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        output, _ = self.rnn(x, h0)
        if self.return_sequences:
            return output
        return output[:, -1, :]


class NCPSAutoNCP(nn.Module):
    """
    AutoNCP model: automatically discovers neural circuit wiring.

    AutoNCP uses an evolutionary algorithm to find the optimal
    sparse connectivity pattern for the liquid network, inspired
    by the C. elegans neural circuit structure.

    Architecture: Sensory -> Inter -> Command -> Motor neurons

    Args:
        input_size: Number of input features (sensory neurons)
        hidden_size: Total number of internal neurons
        output_size: Number of output features (motor neurons)
        model_type: 'cfc' or 'ltc'
        return_sequences: Return full sequence or last step
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        model_type: str = "cfc",
        return_sequences: bool = True,
    ):
        super().__init__()
        _check_ncps()
        wiring = AutoNCP(hidden_size, output_size)
        if model_type == "cfc":
            self.rnn = NCPS_CfC(input_size, wiring, batch_first=True)
        elif model_type == "ltc":
            self.rnn = NCPS_LTC(input_size, wiring, batch_first=True)
        else:
            raise ValueError(f"model_type must be 'cfc' or 'ltc', got '{model_type}'")
        self.return_sequences = return_sequences
        self.output_size = output_size

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        output, _ = self.rnn(x, h0)
        if self.return_sequences:
            return output
        return output[:, -1, :]
