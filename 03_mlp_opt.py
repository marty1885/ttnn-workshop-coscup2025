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
    high=1.0,                  # upper bound for random values
)

# Create a tensor with all elements set to 2.0
b = ttnn.rand(
    shape=(32, 32),            # 32x32 matrix
    dtype=ttnn.bfloat16,       # data type
    layout=ttnn.TILE_LAYOUT,   # TTNN uses TILE_LAYOUT to enable performance on hardware
    device=device,             # specify the device
    low=-1.0,                  # lower bound for random values
    high=1.0,                  # upper bound for random values
)

c = ttnn.matmul(a, b, memory_config=ttnn.L1_MEMORY_CONFIG)
d = ttnn.sigmoid(c)
del c # SRAM is very limited, so we delete c to free up memory

# Print the result
print("Result of adding two tensors:")
print(d)



