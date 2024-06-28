import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from tqdm import tqdm

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# CNN Encoder
class CNNEncoder(nn.Module):
    def __init__(self, input_channels):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)  # Flatten the output

# Transformer Encoder
class TransformerEncoder(nn.Module):
    """
    d_model: the number of expected features in the input (required). Input embedding size.
    input_dim

    """
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        x = self.embedding(x)
        return self.transformer_encoder(x)

# PPO Actor-Critic Network
class PPONetwork(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(PPONetwork, self).__init__()
        self.cnn = CNNEncoder(input_shape[0])
        # Calculate the output size of the CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            cnn_output = self.cnn(sample_input)
            cnn_output_size = cnn_output.shape[1]
        
        self.transformer = TransformerEncoder(cnn_output_size, d_model=32, nhead=2, num_layers=2)
        self.actor = nn.Linear(32, output_dim)
        self.critic = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.unsqueeze(1)  # Add sequence dimension for transformer
        x = self.transformer(x)
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

# PPO Agent
class PPOAgent:
    def __init__(self, input_shape, output_dim, learning_rate=3e-4, gamma=0.99, epsilon=0.2):
        self.policy = PPONetwork(input_shape, output_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, _ = self.policy(state)
        dist = Categorical(action_probs.squeeze())
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def update(self, states, actions, log_probs, rewards, next_states, dones):
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(log_probs).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Compute advantages
        with torch.no_grad():
            _, next_values = self.policy(next_states)
            _, values = self.policy(states)
            advantages = rewards + self.gamma * next_values.squeeze() * (1 - dones) - values.squeeze()
        
        # PPO update
        for epoch in range(5):  # Multiple epochs
            action_probs, values = self.policy(states)
            dist = Categorical(action_probs.squeeze())
            new_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), rewards + self.gamma * next_values.squeeze() * (1 - dones))
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            print(f"    Epoch {epoch + 1}/5 - Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Total Loss: {loss:.4f}")

# Preprocess function for Pong
def preprocess(observation):
    # Convert to grayscale and normalize
    return np.expand_dims(np.mean(observation, axis=2).astype(np.float32) / 255.0, axis=0)

# Training loop
def train(env_name, num_episodes=1000):
    env = gym.make(env_name)
    input_shape = (1, 210, 160)  # Grayscale image
    output_dim = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    agent = PPOAgent(input_shape, output_dim)
    
    episode_rewards = []
    
    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        state, _ = env.reset()
        state = preprocess(state)
        done = False
        total_reward = 0
        step = 0
        
        states, actions, log_probs, rewards, next_states, dones = [], [], [], [], [], []
        
        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess(next_state)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
            total_reward += reward
            step += 1
        
        episode_rewards.append(total_reward)
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"  Steps: {step}")
        print(f"  Total Reward: {total_reward}")
        print(f"  Average Reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
        
        print("Updating policy...")
        agent.update(states, actions, log_probs, rewards, next_states, dones)
        
        if (episode + 1) % 100 == 0:
            print(f"\nEpisode {episode + 1} Summary:")
            print(f"  Average Reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
            print(f"  Best Reward: {max(episode_rewards):.2f}")
            print(f"  Worst Reward: {min(episode_rewards):.2f}")
    
    env.close()
    
    print("\nTraining Completed!")
    print(f"Final Average Reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")

# Run the training
if __name__ == "__main__":
    train("ALE/Pong-v5")