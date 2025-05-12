import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import swanlab


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DuelingDeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.V = nn.Linear(fc2_dim, 1)  # Value stream
        self.A = nn.Linear(fc2_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        V = self.V(x)
        A = self.A(x)
        return V, A
    
class DuelingDQN:
    def __init__(self,state_dim, action_dim, fc1_dim, fc2_dim, lr, gamma, tau, epsilon_start, epsilon_end, epsilon_decay, batch_size, memory_size):
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

        self.q_eval= DuelingDeepQNetwork(state_dim, action_dim, fc1_dim, fc2_dim).to(device)
        self.q_target= DuelingDeepQNetwork(state_dim, action_dim, fc1_dim, fc2_dim).to(device)
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=lr)
    
    def update_netword_parameters(self):
        for target_param, local_param in zip(self.q_target.parameters(), self.q_eval.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
    
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def sample_batch(self):
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            V, A = self.q_eval(state_tensor)
            Q = V + (A - A.mean(dim=1, keepdim=True))
            action = torch.argmax(Q).item()
            return action
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.sample_batch()
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_V, current_A = self.q_eval(states)
        current_Q = current_V + (current_A - current_A.mean(dim=1, keepdim=True))
        current_Q = current_Q.gather(1, actions)

        with torch.no_grad():
            next_V, next_A = self.q_target(next_states)
            next_Q = next_V + (next_A - next_A.mean(dim=1, keepdim=True))
            next_Q = next_Q.max(1)[0].unsqueeze(1)
            target_Q = rewards + self.gamma * next_Q * (1 - dones)
        
        loss = nn.MSELoss()(current_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_freq == 0:
            self.update_netword_parameters()
        self.decrement_epsilon()
    
    def save_model(self, path):
        torch.save(self.q_eval.state_dict(), path)
    
    def eval(self, env):
        original_epsilon = self.epsilon
        self.epsilon = 0.0
        total_rewards = []
        for _ in range(10):
            state, _ = env.reset()
            total_reward = 0
            while True:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
                state = next_state
            total_rewards.append(total_reward)
        self.epsilon = original_epsilon
        return np.mean(total_rewards)


if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n 
    fc_dim1=64
    fc_dim2=64
    lr=1e-3
    gamma=0.99
    tau=0.005
    epsilon_start=1.0
    epsilon_end=0.01
    epsilon_decay=1e-5
    batch_size=128
    memory_size=100000
    agent = DuelingDQN(state_dim, action_dim, fc_dim1, fc_dim2, lr, gamma, tau, epsilon_start, epsilon_end, epsilon_decay, batch_size, memory_size)
    swanlab.init(
        project="Acrobot-v1",
        experiment_name="Dueling-DQN-Acrobot-v1",
        config={
            "state_dim": state_dim,
            "action_dim": action_dim,
            "fc_dim1": fc_dim1,
            "fc_dim2": fc_dim2,
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

    for episode in tqdm(range(swanlab.config["episode"])):
        state, _ = env.reset()
        total_reward = 0
        step_train = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            agent.store_transition((state, action, reward, next_state, done))
            agent.train()
            state = next_state
            step_train += 1
            if done:
                break
        
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}, Steps: {step_train}", flush=True)
        if (episode+1) % 10 == 0:
            eval_env = gym.make("Acrobot-v1")
            avg_reward = agent.eval(eval_env)
            if avg_reward >= agent.best_avg_reward:
                agent.best_avg_reward = avg_reward
                agent.save_model("0512_Dueling_DQN_Acrobot/best_model.pth")
                print(f"Episode {episode}: New best model saved with avg reward: {avg_reward:.2f}")
            eval_env.close()
        
        swanlab.log(
            {
                "step_train":step_train,
                "train_reward": total_reward,
                "eval_best_avg_reward": agent.best_avg_reward,
                "epsilon": agent.epsilon,
            },
            step=episode+1,
        )
    env.close()

