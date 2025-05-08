import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import random
from collections import deque
import matplotlib.pyplot as plt
import swanlab
from tqdm import tqdm

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED=42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

class QNet(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(QNet,self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(state_dim, 36),
            nn.ReLU(),
            nn.Linear(36, 36),
            nn.ReLU(),
            nn.Linear(36, action_dim)
        )
    
    def forward(self,x):
        x=self.fc(x)
        return x
    

class DDQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, batch_size=128):
        self.Q_eval=QNet(state_dim, action_dim).to(device)
        self.Q_target=QNet(state_dim, action_dim).to(device)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.optimizer=optim.Adam(self.Q_eval.parameters(), lr=lr)
        self.gamma=gamma
        self.epsilon=epsilon
        self.batch_size=batch_size
        self.replay_buffer=deque(maxlen=100000)
        self.update_freq=10
        self.step_count=0
        self.eval_episode=5
        self.best_avg_reward=-np.inf
        self.epsilon_decay=0.99
    
    def select_action(self, state):
        if np.random.rand() <self.epsilon:
            return np.random.randint(0, 4)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.Q_eval(state_tensor)
            action = torch.argmax(q_values).item()
            return action
    
    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def sample_batch(self, batch_size):
        batch =np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[i] for i in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def update_epsilon(self):
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = 0.01
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 
        
        states, actions, rewards, next_states, dones= self.sample_batch(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_q_values = self.Q_eval(states).gather(1, actions)

        with torch.no_grad():
            next_actions =self.Q_eval(next_states).argmax(1, keepdim=True)
            next_q_values= self. Q_target(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_freq == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
        
    def save_model(self, path):
        torch.save(self.Q_eval.state_dict(), path)

    def eval(self, env):
        original_epsilon = self.epsilon
        self.epsilon = 0.0
        total_rewards =[]
        for _ in range(self.eval_episode):
            state, _ = env.reset()
            total_reward = 0
            while True:
                action = self.select_action(state)
                next_state, reward, terminated, truncated ,_ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
                state = next_state
            total_rewards.append(total_reward)
        self.epsilon = original_epsilon
        return np.mean(total_rewards)



if __name__ == "__main__":
    env=gym.make("LunarLander-v2")
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.n
    agent=DDQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.1, batch_size=128)
    swanlab.init(
        project="LunarLander-v2",
        experiment_name="DDQN-LunarLander-v2",
        config={
            "state_dim": state_dim,
            "action_dim": action_dim,
            "batch_size": agent.batch_size,
            "gamma": agent.gamma,
            "epsilon": agent.epsilon,
            "update_freq": agent.update_freq,
            "replay_buffer_size": len(agent.replay_buffer),
            "lr": agent.optimizer.defaults['lr'],
            "episode" : 600,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.99,
        }
    )
    agent.epsilon=swanlab.config["epsilon_start"]

    for episode in tqdm(range(swanlab.config["episode"])):
        state,_ =env.reset()
        total_reward=0
        step_train=0
        while True:
            action=agent.select_action(state)
            next_state, reward, terminated, truncated ,_ = env.step(action)
            total_reward += reward
            done= terminated or truncated
            agent.store_transition((state, action, reward, next_state, done))

            agent.train()
            state=next_state
            step_train+=1
            if done :
                break
        print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Steps: {step_train}",flush=True)
        agent.epsilon=max(agent.epsilon * swanlab.config["epsilon_decay"], swanlab.config["epsilon_end"])
    
        if episode % 10 == 0:
            eval_env=gym.make("LunarLander-v2")
            avg_reward=agent.eval(eval_env)
            eval_env.close()

            if avg_reward >= agent.best_avg_reward:
                agent.best_avg_reward = avg_reward
                agent.save_model("0507_DDQN_LunarLander/best_model.pth")
                print(f"Episode {episode}: New best model saved with avg reward: {avg_reward:.2f}")
        
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