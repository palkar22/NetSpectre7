import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from sys import argv
from os.path import isfile

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



def main(original_model):
    if original_model == None:
        while True:
            original_model = input("Enter the previously trained model (.pth) file name:")
            if isfile(original_model) == True:
                break
            else:
                print("File doesnt exist. Retry.")

    # string manipulation to get filename for quantized model
    original_model = original_model[2:]
    quantized_model_list = original_model.split("_")
    quantized_model_list.insert(0,"quantized")
    quantized_model = "_".join(quantized_model_list)

    quantize_onnx_model(original_model, quantized_model)

if __name__ == "__main__":
    file = argv[1] if len(argv) > 1 else None
    main(file)