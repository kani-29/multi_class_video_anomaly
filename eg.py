import torch

# Create a tensor with 10 rows and 3 columns
tensor = torch.zeros(10, 3)

# Set the first column to 1
tensor[:, 0] = 1

print(tensor)
