"""ONNX export and inference-latency benchmarking.

A corridor overwatch sensor is field hardware, not a data-center GPU --
export and benchmark accordingly: fp32 and dynamic int8 quantization, and
both CPU and (when available) GPU latency, rather than assuming one platform.
See docs/METHODOLOGY_AND_LIMITATIONS.md for why quantization's storage win is
close to guaranteed while its *latency* win is not -- it depends on model
scale and hardware kernel support, and this module reports what was actually
measured rather than the generic "quantization is faster" assumption.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ExportedModel:
    fp32_path: Path
    int8_path: Path
    fp32_size_kb: float
    int8_size_kb: float


def export_onnx(model: torch.nn.Module, input_shape: tuple[int, ...], out_dir: str | Path, name: str) -> ExportedModel:
    """Exports `model` to ONNX (fp32) and a dynamically-quantized int8
    sibling. `input_shape` excludes the batch dimension, e.g. `(1, 96, 96)`
    for a single-channel 96x96 chip."""
    import onnx  # noqa: F401  (import validates onnx is installed before we write files)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / f"{name}_fp32.onnx"
    int8_path = out_dir / f"{name}_int8.onnx"

    original_device = next(model.parameters()).device
    model.eval().cpu()  # torch.onnx.export traces on CPU
    dummy = torch.randn(1, *input_shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (dummy,),
            str(fp32_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,
        )
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    model.to(original_device)

    fp32_size = fp32_path.stat().st_size / 1024
    int8_size = int8_path.stat().st_size / 1024
    return ExportedModel(fp32_path, int8_path, fp32_size, int8_size)


def benchmark_onnx(
    path: str | Path, input_shape: tuple[int, ...], n: int = 300, provider: str = "CPUExecutionProvider"
) -> float:
    """Mean inference latency in milliseconds."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=[provider])
    x = np.random.randn(1, *input_shape).astype(np.float32)
    for _ in range(20):
        sess.run(None, {"input": x})
    t0 = time.time()
    for _ in range(n):
        sess.run(None, {"input": x})
    return (time.time() - t0) / n * 1000


def benchmark_torch(
    model: torch.nn.Module, input_shape: tuple[int, ...], n: int = 300, device: torch.device | None = None
) -> float:
    """Mean inference latency in milliseconds. Restores the model to its
    original device afterward -- moving a model in place is a real side
    effect callers shouldn't have to remember to undo."""
    device = device or torch.device("cpu")
    original_device = next(model.parameters()).device
    model = model.to(device).eval()
    x = torch.randn(1, *input_shape, device=device)
    with torch.no_grad():
        for _ in range(20):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n * 1000
    model.to(original_device)
    return elapsed
