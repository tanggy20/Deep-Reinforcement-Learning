import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import swanlab
from tqdm import tqdm
import os

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, fc1_dim)
        self.relu= nn.ReLU()
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim,is_train=True):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.relu= nn.ReLU()
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, action_dim)
        self.noisy =torch.distributions.Normal(0, 0.2)
        self.is_train = is_train
    
    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = torch.tanh(self.fc3(x))
        return x
    
    def select_action(self, epsilon, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.forward(state).squeeze()
            if self.is_train:
                noise = epsilon*self.noisy.sample(action.size()).to(device)
                action = action + noise
        
        return 2*torch.clip(action, -1, 1)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, fc1_dim=64, fc2_dim=64, lr=1e-3, gamma=0.99, tau=0.005, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=1e-5, batch_size=128, memory_size=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon_start
        self.step_count = 0 
        self.update_freq = 10
        self.best_avg_reward = -np.inf

        # Initialize actor and critic networks
        self.actor_eval = Actor(state_dim, action_dim, fc1_dim, fc2_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, fc1_dim, fc2_dim, is_train=False).to(device)
        self.critic_eval = Critic(state_dim, action_dim, fc1_dim, fc2_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, fc1_dim, fc2_dim).to(device)

        # Copy weights from eval to target networks
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())

        # Initialize optimizers for actor and critic networks
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=lr)

    def update_network_parameters(self):
        for target_param, local_param in zip(self.actor_target.parameters(),self.actor_eval.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, local_param in zip(self.critic_target.parameters(),self.critic_eval.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def decrement_epsilon(self):   
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
    
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def select_action(self, state):
        action = self.actor_eval.select_action(self.epsilon, state)
        return action.cpu().numpy().reshape(self.action_dim)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.sample_batch()
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        # Update critic
        next_actions = self.actor_target(next_states)
        target_q = self.critic_target(next_states, next_actions)
        target_q = rewards + self.gamma * target_q * (1 - dones)
        current_q = self.critic_eval(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic_eval(states, self.actor_eval(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.step_count += 1

        if self.step_count % self.update_freq == 0:
            self.update_network_parameters()
        self.decrement_epsilon()

    def eval(self, env):
        original_epsilon = self.epsilon
        self.epsilon = 0.0
        total_rewards = []
        for _ in range(10):
            state, _ = env.reset()
            total_reward =0 
            while True:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _= env.step(action)
                total_reward += reward
                done = terminated or truncated
                state = next_state
                if done:
                    break
            total_rewards.append(total_reward)
        self.epsilon = original_epsilon
        return np.mean(total_rewards)
            



    def save_model(self, path):
        actor_path = os.path.join(path, "best_actor.pth")
        critic_path = os.path.join(path, "best_critic.pth")
        torch.save(self.actor_eval.state_dict(), actor_path)
        torch.save(self.critic_eval.state_dict(), critic_path)
    


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    fc1_dim = 64
    fc2_dim = 64
    lr = 1e-3
    gamma = 0.99
    tau = 0.005
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 1e-5
    batch_size = 32
    memory_size = 100000
    agent = DDPGAgent(state_dim, action_dim, fc1_dim, fc2_dim, lr, gamma, tau, epsilon_start, epsilon_end, 
                      epsilon_decay, batch_size, memory_size)
    
    swanlab.init(
        project="Pendulum-v1",
        experiment_name="DDPG-Pendulum-v1",
        config={
            "state_dim": state_dim,
            "action_dim": action_dim,
            "fc1_dim": fc1_dim,
            "fc2_dim": fc2_dim,
            "lr": lr,
            "gamma": gamma,
            "tau": tau,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "batch_size": batch_size,
            "memory_size": memory_size,
            "episode": 1000
        }
    )

    for episode in tqdm(range(swanlab.config['episode'])):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _= env.step(action)
            total_reward += reward
            done = terminated or truncated
            agent.store_transition((state, action, reward, next_state, done))
            agent.train()
            state = next_state
            steps += 1
            if done:
                break
        print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {steps}", flush=True)

        if (episode + 1) % 10 == 0:
            eval_env = gym.make("Pendulum-v1")
            avg_reward = agent.eval(eval_env)
            if avg_reward >= agent.best_avg_reward:
                agent.best_avg_reward = avg_reward
                agent.save_model("0513_DDPG_Pendulum")
                print(f"Evaluation: Episode: {episode}, Avg Reward: {avg_reward}", flush=True)
            eval_env.close()
        swanlab.log(
            {
                "step_train":steps,
                "train_reward": total_reward,
                "eval_best_avg_reward": agent.best_avg_reward,
                "epsilon": agent.epsilon,
            },
            step=episode+1,
        )
    env.close()    