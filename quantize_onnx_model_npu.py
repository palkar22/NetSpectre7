import onnx
from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantType, QuantizationMode
import numpy as np

def generate_calibration_data(num_samples=100, input_shape=(1, 10, 10)):
    """
    Generate dummy calibration data.
    Adjust the input_shape based on your model's input requirements.
    """
    return [np.random.randn(*input_shape).astype(np.float32) for _ in range(num_samples)]

def representative_dataset():
    """
    Generator function to yield calibration data.
    """
    for data in generate_calibration_data():
        yield {'input': data}

def quantize_onnx_model_for_npu(model_path, quantized_model_path):
    # Load the ONNX model
    onnx_model = onnx.load(model_path)
    
    # Quantize the model
    quantized_model = quantize_static(
        model_input=model_path,
        model_output=quantized_model_path,
        calibration_data_reader=representative_dataset(),
        quant_format=QuantizationMode.QDQ,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax
    )
    
    print(f"Model quantized for NPU and saved to {quantized_model_path}")

if __name__ == "__main__":
    original_model = "packet_prioritizer.onnx"
    quantized_model = "packet_prioritizer_quantized_npu.onnx"
    
    quantize_onnx_model_for_npu(original_model, quantized_model)