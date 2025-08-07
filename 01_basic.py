import ttnn

# Open device
device = ttnn.open_device(device_id=0)

# Create a tensor with all elements set to 1.0
a = ttnn.full(
    shape=(32, 32),            # 32x32 matrix
    fill_value=1.0,            # value
    dtype=ttnn.bfloat16,       # data type
    layout=ttnn.TILE_LAYOUT,   # TTNN uses TILE_LAYOUT to enable performance on hardware
    device=device,             # specify the device
)

# Create a tensor with all elements set to 2.0
b = ttnn.full(
    shape=(32, 32),            # 32x32 matrix
    fill_value=2.0,            # value
    dtype=ttnn.bfloat16,       # data type
    layout=ttnn.TILE_LAYOUT,   # TTNN uses TILE_LAYOUT to enable performance on hardware
    device=device,             # specify the device
)

# Add the two tensors
c = ttnn.add(a, b)

# Print the result
print("Result of adding two tensors:")
print(c)

