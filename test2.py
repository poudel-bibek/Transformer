import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_fake_data(sequence_length, feature_dim):
    return np.random.rand(sequence_length, feature_dim)

def self_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    output = np.dot(attention_weights, V)
    return output, attention_weights

# Parameters
sequence_length = 10
feature_dim = 38
d_model = 64

# Create fake data
X = create_fake_data(sequence_length, feature_dim)

# Create fake linear projections for Q, K, V
W_Q = np.random.rand(feature_dim, d_model)
W_K = np.random.rand(feature_dim, d_model)
W_V = np.random.rand(feature_dim, d_model)

# Compute Q, K, V
Q = np.dot(X, W_Q)
K = np.dot(X, W_K)
V = np.dot(X, W_V)

# Compute self-attention
output, attention_weights = self_attention(Q, K, V)

# Visualize attention weights
plt.figure(figsize=(10, 8))
sns.heatmap(attention_weights, annot=True, cmap='YlGnBu')
plt.title('Self-Attention Weights')
plt.xlabel('Key Positions')
plt.ylabel('Query Positions')
plt.show()

print("Attention weights shape:", attention_weights.shape)
print("\nSample interpretation:")
print(f"Element 0 attends to element 5 with weight: {attention_weights[0, 5]:.4f}")
print(f"Element 3 attends to element 7 with weight: {attention_weights[3, 7]:.4f}")