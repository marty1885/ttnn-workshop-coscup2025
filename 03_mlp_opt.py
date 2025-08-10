import ttnn

# Open device
device = ttnn.open_device(device_id=0)

# Create a random tensor for input
x = ttnn.rand(
    shape=(32, 32),            # 32x32 matrix
    dtype=ttnn.bfloat16,       # data type
    layout=ttnn.TILE_LAYOUT,   # TTNN uses TILE_LAYOUT to enable performance on hardware
    device=device,             # specify the device
    low=-1.0,                  # lower bound for random values
    high=1.0,                  # upper bound for random values
)

# Create a random tensor for weights
w = ttnn.rand(
    shape=(32, 32),            # 32x32 matrix
    dtype=ttnn.bfloat16,       # data type
    layout=ttnn.TILE_LAYOUT,   # TTNN uses TILE_LAYOUT to enable performance on hardware
    device=device,             # specify the device
    low=-1.0,                  # lower bound for random values
    high=1.0,                  # upper bound for random values
)

# Create a random tensor for bias
b = ttnn.rand(
    shape=(1, 32),            # 32x32 matrix
    dtype=ttnn.bfloat16,       # data type
    layout=ttnn.TILE_LAYOUT,   # TTNN uses TILE_LAYOUT to enable performance on hardware
    device=device,             # specify the device
    low=-1.0,                  # lower bound for random values
    high=1.0,                  # upper bound for random values
)

# Put the tensors into SRAM to optimize memory access
y = ttnn.matmul(x, w, memory_config=ttnn.L1_MEMORY_CONFIG)
z = ttnn.add(y, b, memory_config=ttnn.L1_MEMORY_CONFIG)
# L1 is a very limited resource, free as soon as possible
del y
# no L1_MEMORY_CONFIG in activation - result on DRAM
res = ttnn.sigmoid(z)
del z

# Print the result
print("Result of adding two tensors:")
print(res)



