import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import swanlab  
from tqdm import tqdm


# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class PolicyNetwork(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 36)
        self.fc2 = nn.Linear(36, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
    

class REINFORCEAgent:
    def __init__(self,state_dim,action_dim,lr=1e-3, gamma=0.99):
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs = self.policy_net(state_tensor)
        action = np.random.choice(len(action_probs[0]), p=action_probs[0].cpu().detach().numpy())
        return action
    
    def compute_returns(self, rewards):
        returns =[]
        R=0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = np.array(returns)
        # 标准化回报
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def learn(self, states, actions, rewards):
        returns = self.compute_returns(rewards)
        returns = torch.FloatTensor(returns).to(device)
        actions = torch.LongTensor(actions).to(device)
        states = np.array(states)
        states = torch.FloatTensor(states).to(device)

        action_probs = self.policy_net(states)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
        loss = -torch.mean(action_log_probs * returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
    

if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]  
    action_dim = env.action_space.n
    agent = REINFORCEAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)
    best_avg_reward = -np.inf
    swanlab.init(
        project="CartPole-v1",
        experiment_name="REINFORCE-CartPole-v1",
        config={
            "state_dim": state_dim,
            "action_dim": action_dim,
            "gamma": agent.gamma,
            "lr": agent.optimizer.defaults['lr'],
            "episode" : 2000,
        }
    )

    for episode in tqdm(range(swanlab.config["episode"])):
        state, _ = env.reset()
        done = False
        states, actions, rewards = [], [], []
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated ,_ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            done = terminated or truncated
        train_reward= sum(rewards)
        print(f"Episode: {episode + 1}, Total Reward: {train_reward}", flush=True)
        loss = agent.learn(states, actions, rewards)

        if (episode + 1) % 20 == 0:
            eval_env= gym.make(env_name)
            total_rewards = []

            for _ in range(10):
                state, _ = eval_env.reset()
                total_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated ,_ = eval_env.step(action)
                    total_reward += reward
                    state = next_state
                    done = terminated or truncated
                total_rewards.append(total_reward)
            avg_reward = np.mean(total_rewards)
            if avg_reward >=best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model('0509_REINFORCE_CartPole/best_model.pth')
            print(f"Evaluation - Episode: {episode + 1}, Avg Reward: {avg_reward:.2f}")
            eval_env.close()
        swanlab.log({"loss": loss,
                     "train_reward": train_reward,
                     "eval_best_avg_reward": best_avg_reward})
        
    env.close()