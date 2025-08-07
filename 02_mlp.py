import ttnn

# Open device
device = ttnn.open_device(device_id=0)

# Create a tensor with all elements set to 1.0
a = ttnn.rand(
    shape=(32, 32),            # 32x32 matrix
    dtype=ttnn.bfloat16,       # data type
    layout=ttnn.TILE_LAYOUT,   # TTNN uses TILE_LAYOUT to enable performance on hardware
    device=device,             # specify the device
    low=-1.0,                  # lower bound for random values
    high=1.0                   # upper bound for random values
)

# Create a tensor with all elements set to 2.0
b = ttnn.rand(
    shape=(32, 32),            # 32x32 matrix
    dtype=ttnn.bfloat16,       # data type
    layout=ttnn.TILE_LAYOUT,   # TTNN uses TILE_LAYOUT to enable performance on hardware
    device=device,             # specify the device
    low=-1.0,                  # lower bound for random values
    high=1.0                   # upper bound for random values
)

# Perform matrix multiplication
c = ttnn.matmul(a, b)
d = ttnn.sigmoid(c) # sigmoid activation function

# Print the result
print("Result of matrix multiplication followed by sigmoid activation:")
print(d)



