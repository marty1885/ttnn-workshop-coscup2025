import ttnn

# Open device
device = ttnn.open_device(device_id=0)

# Create random tensor as input
x = ttnn.rand(
    shape=(32, 32),            # 32x32 matrix
    dtype=ttnn.bfloat16,       # data type
    layout=ttnn.TILE_LAYOUT,   # TTNN uses TILE_LAYOUT to enable performance on hardware
    device=device,             # specify the device
    low=-1.0,                  # lower bound for random values
    high=1.0                   # upper bound for random values
)

# Create random tensor as weights
w = ttnn.rand(
    shape=(32, 32),            # 32x32 matrix
    dtype=ttnn.bfloat16,       # data type
    layout=ttnn.TILE_LAYOUT,   # TTNN uses TILE_LAYOUT to enable performance on hardware
    device=device,             # specify the device
    low=-1.0,                  # lower bound for random values
    high=1.0                   # upper bound for random values
)

# Create random tensor as bias
b = ttnn.rand(
    shape=(1, 32),            # 32x32 matrix
    dtype=ttnn.bfloat16,       # data type
    layout=ttnn.TILE_LAYOUT,   # TTNN uses TILE_LAYOUT to enable performance on hardware
    device=device,             # specify the device
    low=-1.0,                  # lower bound for random values
    high=1.0                   # upper bound for random values
)

# Perform matrix multiplication
y = ttnn.matmul(x, w) # matrix multiplication of a and w
# Add bias
y = ttnn.add(y, b) # add bias b to the result of matrix multiplication
y = ttnn.sigmoid(y) # sigmoid activation function

# Print the result
print("Result of matrix multiplication followed by sigmoid activation:")
print(y)



