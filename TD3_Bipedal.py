import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import os
import swanlab
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.relu= nn.ReLU()
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, action_dim)
    
    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = torch.tanh(self.fc3(x))
        return x
    


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

class TD3Agent:
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim, lr_actor, lr_critic,  gamma, tau, batch_size, memory_size, noise_clip):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.step_count = 0
        self.delay_time = 2
        self.noise_clip = noise_clip
        self.best_avg_reward = -np.inf

        # Initialize networks and optimizers
        self.actor= Actor(state_dim, action_dim, fc1_dim, fc2_dim).to(device)
        self.actor_target= Actor(state_dim, action_dim, fc1_dim, fc2_dim).to(device)
        self.critic1= Critic(state_dim, action_dim, fc1_dim, fc2_dim).to(device)
        self.critic1_target= Critic(state_dim, action_dim, fc1_dim, fc2_dim).to(device)
        self.critic2= Critic(state_dim, action_dim, fc1_dim, fc2_dim).to(device)
        self.critic2_target= Critic(state_dim, action_dim, fc1_dim, fc2_dim).to(device)

        # Copy weights from actor and critic to target networks
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(local_param.data)
        
        for target_param, local_param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(local_param.data)

        for target_param, local_param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(local_param.data)

        # Initialize optimizers
        self.actor_optimizer= optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer= optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer= optim.Adam(self.critic2.parameters(), lr=lr_critic)

    def update_network_parameters(self):
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, local_param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

        for target_param, local_param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state, is_train):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        noise = np.random.normal(0, 0.1, size=self.action_dim)
        noise = np.clip(noise, -self.noise_clip, self.noise_clip)
        if is_train:
            # Add exploration noise
            action = action + noise
        action = np.clip(action, -1, 1)
        return action
    
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def train(self):
        if len(self.memory) <self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.sample_batch()
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        # Train critic networks
        with torch.no_grad():
            noise = torch.FloatTensor(np.random.normal(0, 0.1, size=actions.size())).to(device)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -1, 1)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * target * (1 - dones)
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss1 = nn.MSELoss()(current_q1, target_q.detach())
        critic_loss2 = nn.MSELoss()(current_q2, target_q.detach())
        self.critic1_optimizer.zero_grad()
        critic_loss1.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()
        
        self.step_count += 1
        # Train actor network
        if self.step_count % self.delay_time != 0:
            return
        next_actions = self.actor(states)
        actor_loss = - self.critic1(states, next_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Update target networks
        self.update_network_parameters()
    
    def eval(self, env):
        total_rewards = []
        for _ in range(10):
            state, _ = env.reset()
            total_reward =0 
            while True:
                action = self.select_action(state,False)
                next_state, reward, terminated, truncated, _= env.step(action)
                total_reward += reward
                done = terminated or truncated
                state = next_state
                if done:
                    break
            total_rewards.append(total_reward)
        return np.mean(total_rewards)
    
    def save_model(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, "best_actor.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(path, "best_critic1.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(path, "best_critic2.pth"))

if __name__ == "__main__":
    # Hyperparameters
    env_name = "BipedalWalker-v3"
    fc1_dim = 400
    fc2_dim = 300
    lr_actor = 3e-3
    lr_critic = 2e-3
    gamma = 0.99
    tau = 0.005
    batch_size = 128
    memory_size = 1000000
    noise_clip = 0.5

    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize agent
    agent = TD3Agent(state_dim, action_dim, fc1_dim, fc2_dim, lr_actor, lr_critic, gamma, tau, batch_size, memory_size, noise_clip)

    swanlab.init(
        project="BipedalWalker-v3",
        experiment_name="TD3-BipedalWalker-v3",
        config=
       {
           "state_dim": state_dim,
           "action_dim": action_dim,
            "fc1_dim": fc1_dim,
            "fc2_dim": fc2_dim,
            "lr_actor": lr_actor,
            "lr_critic": lr_critic,
            "gamma": gamma,
            "tau": tau,
            "batch_size": batch_size,
            "memory_size": memory_size,
            "noise_clip": noise_clip,
            "episode": 1000,
       }
   )
    for episode in tqdm(range(swanlab.config['episode'])):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        while True:
            action = agent.select_action(state,True)
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
            eval_env = gym.make("BipedalWalker-v3")
            avg_reward = agent.eval(eval_env)
            if avg_reward >= agent.best_avg_reward:
                agent.best_avg_reward = avg_reward
                agent.save_model("0515_TD3_Bipedal-v3")
                print(f"Evaluation: Episode: {episode}, Avg Reward: {avg_reward}", flush=True)
            eval_env.close()
        swanlab.log(
            {
                "step_train":steps,
                "train_reward": total_reward,
                "eval_best_avg_reward": agent.best_avg_reward,
            },
            step=episode+1,
        )
    env.close()    