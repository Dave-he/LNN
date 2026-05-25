"""
LFM2 (Liquid Foundation Model 2) inference and deployment module.

Provides utilities for loading, running inference, and deploying
Liquid AI's LFM2 models from HuggingFace.

Note: LFM2 models require the `transformers` library and may need
specific model configurations from Liquid AI's HuggingFace repository.

Usage:
    from lnn.lfm2.inference import LFM2Inference

    runner = LFM2Inference(model_name="LiquidAI/LFM2-350M")
    output = runner.generate("The future of AI is")
"""

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def _check_transformers():
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers library is not installed. Install with: pip install transformers accelerate sentencepiece"
        )


AVAILABLE_MODELS = {
    "LFM2-350M": "LiquidAI/LFM2-350M",
    "LFM2-700M": "LiquidAI/LFM2-700M",
    "LFM2-1.2B": "LiquidAI/LFM2-1.2B",
    "LFM2-2.6B-Exp": "LiquidAI/LFM2-2.6B-Exp",
    "LFM2-24B-A2B": "LiquidAI/LFM2-24B-A2B",
}


class LFM2Inference:
    """
    Inference runner for LFM2 models.

    Handles model loading, tokenization, and text generation
    with support for different precision modes and device placement.

    Args:
        model_name: HuggingFace model ID or shorthand (e.g., 'LFM2-350M')
        device: Device to run on ('auto', 'cuda', 'cpu', 'mps')
        dtype: Model precision ('float32', 'float16', 'bfloat16')
    """

    def __init__(
        self,
        model_name: str = "LFM2-350M",
        device: str = "auto",
        dtype: str = "float32",
    ):
        _check_transformers()

        if model_name in AVAILABLE_MODELS:
            model_name = AVAILABLE_MODELS[model_name]

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(dtype, torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            **kwargs,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.no_grad()
    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        return {
            "total_params": total_params,
            "model_size_mb": model_size_mb,
            "dtype": str(self.dtype),
            "device": str(self.device),
        }


class LFM2EdgeDeployer:
    """
    Edge deployment utilities for LFM2 models.

    Provides model quantization, export, and optimization
    for deployment on resource-constrained devices.

    Strategies:
        - Dynamic quantization (INT8)
        - TorchScript export
        - ONNX export (if onnx available)
    """

    def __init__(self, model_name: str = "LFM2-350M"):
        _check_transformers()
        if model_name in AVAILABLE_MODELS:
            model_name = AVAILABLE_MODELS[model_name]
        self.model_name = model_name

    def quantize_dynamic(self) -> torch.nn.Module:
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype=torch.float32
        )
        quantized = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized

    def export_torchscript(self, output_path: str, max_seq_len: int = 128):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype=torch.float32
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        dummy_input = tokenizer("Hello", return_tensors="pt")["input_ids"]
        traced = torch.jit.trace(model, dummy_input)
        traced.save(output_path)
        return output_path
