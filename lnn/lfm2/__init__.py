from lnn.lfm2.inference import AVAILABLE_MODELS, LFM2EdgeDeployer, LFM2Inference
from lnn.lfm2.parallel_integration import (
    RECURRENT_CLASSES,
    replace_lstm_with_parallel_cfc,
)

__all__ = [
    "LFM2Inference",
    "LFM2EdgeDeployer",
    "AVAILABLE_MODELS",
    "RECURRENT_CLASSES",
    "replace_lstm_with_parallel_cfc",
]
