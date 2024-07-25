import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(model_path, quantized_model_path):
    # Load the ONNX model
    onnx_model = onnx.load(model_path)
    
    # Quantize the model
    quantized_model = quantize_dynamic(
        model_input=model_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QUInt8
    )
    
    print(f"Model quantized and saved to {quantized_model_path}")

if __name__ == "__main__":
    original_model = "packet_prioritizer.onnx"
    quantized_model = "packet_prioritizer_quantized.onnx"
    
    quantize_onnx_model(original_model, quantized_model)