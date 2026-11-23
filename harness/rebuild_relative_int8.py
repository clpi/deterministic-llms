"""Generate a separately labeled QInt8 control in an isolated asset directory."""
import sys
from pathlib import Path
from onnxruntime.quantization import QuantType, quantize_dynamic

root = Path(sys.argv[1])
output = root / "gpt2-lm-head-10-int8.onnx"
if output.exists():
    raise FileExistsError(output)
quantize_dynamic(root / "gpt2-lm-head-10.onnx", output, weight_type=QuantType.QInt8)
