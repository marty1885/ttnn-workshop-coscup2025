import ttnn
import torch

# Open device
device = ttnn.open_device(device_id=0)

layer = torch.nn.Linear(64, 128)

# Convert torch tensor to ttnn tensor
# Note: Linear layer in PyTorch has the weight in shape (out_features, in_features),
#                               vvvvvvvvvv
ttnn_weight = ttnn.from_torch(layer.weight.T,
    device=device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT)
ttnn_bias = ttnn.from_torch(layer.bias,
    device=device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT)

# Same operation in TTNN
x = torch.rand(1, 64)
x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

y = ttnn.matmul(x, ttnn_weight) + ttnn_bias
print(y)


