import torch
import torch.onnx
from train_prioritizer_model import PacketPrioritizer


# ... (keep all your existing imports and class definitions)

def main():
    # Load the dataset
    X, y = torch.load('packet_dataset.pt')
    
    # Prepare data
    train_loader = prepare_data(X, y)
    
    # Initialize the model
    model = PacketPrioritizer()
    
    # Train the model
    train_model(model, train_loader)
    
    # Save the trained model
    torch.save(model.state_dict(), 'packet_prioritizer.pth')
    
    print("Model trained and saved successfully.")

    # Convert to ONNX
    model.eval()  # Set the model to evaluation mode
    
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 10, 10)  # Adjust size based on your input shape (batch_size, sequence_length, input_size)
    
    # Export the model
    torch.onnx.export(model,               # model being run
                      dummy_input,         # model input (or a tuple for multiple inputs)
                      "packet_prioritizer.onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    
    print("Model converted to ONNX format and saved as packet_prioritizer.onnx")

if __name__ == "__main__":
    main()