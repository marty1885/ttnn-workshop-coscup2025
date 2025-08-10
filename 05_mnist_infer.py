import torch
import ttnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = ttnn.open_device(device_id=0)

class Model:
    def __init__(self, state):
        self.fc0_weight = ttnn.from_torch(state['fc0.weight'].T, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        self.fc0_bias = ttnn.from_torch(state['fc0.bias'], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        self.fc1_weight = ttnn.from_torch(state['fc1.weight'].T, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        self.fc1_bias = ttnn.from_torch(state['fc1.bias'], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        self.fc2_weight = ttnn.from_torch(state['fc2.weight'].T, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        self.fc2_bias = ttnn.from_torch(state['fc2.bias'], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    def forward(self, x):
        x = ttnn.matmul(x, self.fc0_weight)
        x = ttnn.add(x, self.fc0_bias)
        x = ttnn.relu(x)

        x = ttnn.matmul(x, self.fc1_weight)
        x = ttnn.add(x, self.fc1_bias)
        x = ttnn.relu(x)

        x = ttnn.matmul(x, self.fc2_weight)
        x = ttnn.add(x, self.fc2_bias)
        x = ttnn.softmax(x, dim=1)

        return x

state = torch.load("mnist_classifier.pt")
model = Model(state)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST("./dataset", train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)


correct = 0
for data, target in loader:
    # Flatten the input (as needed for MLP)
    data = data.view(data.size(0), -1)
    # Convert to TTNN format
    data = ttnn.from_torch(data, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    # Run inference
    output = model.forward(data)
    # Convert output back to Torch format
    output = ttnn.to_torch(output)

    # Calculate accuracy
    _, predicted = torch.max(output, axis=1)
    correct += (predicted == target).sum().item()

print(f'Accuracy: {correct / len(dataset) * 100:.2f}%')

