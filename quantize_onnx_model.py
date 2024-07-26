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
    while True:
        original_model = input("Enter the previously trained model (.pth) file name:")
        if isfile(original_model) == True:
            break
        else:
            print("File doesnt exist. Retry.")

    quantized_model = "quantized_" + original_model
    quantize_onnx_model(original_model, quantized_model)