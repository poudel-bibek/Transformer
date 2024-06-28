import torch

# Let's create a sample tensor to represent the transformer output
batch_size = 2
sequence_length = 130  # 10 timesteps * 13 tokens per timestep
d_model = 64

# Create a random tensor to simulate transformer output
x = torch.rand(batch_size, sequence_length, d_model)

print("Original shape:", x.shape)

# Select the last token
last_token = x[:, -1, :]

print("Shape after selecting last token:", last_token.shape)

# Let's look at the first sample in the batch
print("First sample, all tokens:", x[0].shape)
print("First sample, last token:", last_token[0].shape)