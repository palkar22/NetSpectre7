import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# globals
PTH_FILENAME = None


class PacketPrioritizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 6)  # 6 output classes for 6 priority levels
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.relu(self.fc1(lstm_out[:, -1, :]))
        return self.fc2(x)

def train_model(model, train_loader, device, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

def prepare_data(X, y):
    priority_to_class = {
        'Games': 0,
        'Real Time': 1,
        'Streaming': 2,
        'Normal': 3,
        'Web download': 4,
        'App download': 5
    }
    y_classes = torch.tensor([priority_to_class[p] for p in y])
    
    dataset = TensorDataset(X, y_classes)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def main():
    global PTH_FILENAME
    PTH_FILENAME = input("Enter the previously trained model .pt file name:")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pre-trained model
    model = PacketPrioritizer()
    model.load_state_dict(torch.load(PTH_FILENAME))
    model.to(device)
    print("Pre-trained model loaded successfully.")

    # Load the new dataset
    X, y = torch.load('packet_dataset.pt')
    
    # Prepare data
    train_loader = prepare_data(X, y)
    
    # Train the model again
    print("Starting additional training...")
    train_model(model, train_loader, device)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # updated PTH model's filename
    new_PTH_FILENAME = f"packet_prioritizer_{timestamp}.onnx"

    # Save the updated model
    torch.save(model.state_dict(), new_PTH_FILENAME)
    
    print(f"Model retrained and saved successfully as {new_PTH_FILENAME}")

if __name__ == "__main__":
    main()