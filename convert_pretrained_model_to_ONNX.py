import torch
import torch.onnx
from train_prioritizer_model import PacketPrioritizer
import datetime
from sys import argv
from os.path import isfile

def main(PTH_FILENAME):
    if PTH_FILENAME is None:
        while True:
            PTH_FILENAME = input("Enter the pretrained model (.pth) file name: ")
            if not isfile(PTH_FILENAME):
                print(f"File doesn't exist: {PTH_FILENAME}. Retry.")
            else:
                break

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pre-trained model
    model = PacketPrioritizer()

    try:
        model.load_state_dict(torch.load(PTH_FILENAME, map_location=device))
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Convert to ONNX
    model.eval()  # Set the model to evaluation mode
    
    # Create a dummy input tensor on the same device as the model
    dummy_input = torch.randn(5, 15, 15, device=device)  # Adjust size based on input shape (batch_size, sequence_length, input_size)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create ONNX filename with timestamp
    onnx_filename = f"packet_prioritizer_{timestamp}.onnx"
    
    # Export the model
    try:
        torch.onnx.export(model,               # model being run
                          dummy_input,         # model input (or a tuple for multiple inputs)
                          onnx_filename,       # where to save the model
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=11,    # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],   # the model's input names
                          output_names=['output'], # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                        'output': {0: 'batch_size'}})
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        return   
    
    print(f"Model converted to ONNX format and saved as {onnx_filename}")

if __name__ == "__main__":
    PTH_FILENAME = argv[1] if len(argv) > 1 else None
    main(PTH_FILENAME)