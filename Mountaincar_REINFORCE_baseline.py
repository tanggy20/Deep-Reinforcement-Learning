import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import swanlab
from tqdm import tqdm
import os

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_height(pos_x):
    return np.sin(pos_x * 3) * 0.45 + 0.55  

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 24)
        self.fc2 = nn.Linear(24, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

class BaselineNetwork(nn.Module):
    def __init__(self, state_dim):
        super(BaselineNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 24)
        self.fc2 = nn.Linear(24, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.baseline_net = BaselineNetwork(state_dim).to(device)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_baseline = optim.Adam(self.baseline_net.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self,state):
        state =torch.FloatTensor(state).unsqueeze(0).to(device)
        probs= self.policy_net(state)
        action =np.random.choice(len(probs[0]),p=probs[0].cpu().detach().numpy())
        return action
    
    def compute_returns(self, rewards):
        returns= []
        R=0
        for r in rewards[::-1]:
            R=r+self.gamma*R
            returns.insert(0,R)
        returns = np.array(returns)
        returns =(returns-returns.mean())/(returns.std()+1e-8)
        return returns
    
    def learn(self, states, actions, rewards):
        returns=self.compute_returns(rewards)
        returns=torch.FloatTensor(returns).to(device)
        actions=torch.LongTensor(actions).to(device)
        states=np.array(states)
        states=torch.FloatTensor(states).to(device)
        action_probs=self.policy_net(states)
        action_log_probs =torch.log(action_probs.gather(1,actions.unsqueeze(1)).squeeze())
        baseline_values=self.baseline_net(states).squeeze()
        advantages=returns-baseline_values.detach()
        policy_loss= -torch.mean(action_log_probs*advantages)
        baseline_loss =torch.mean((returns-baseline_values)**2)
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()
        self.optimizer_baseline.zero_grad()
        baseline_loss.backward()
        self.optimizer_baseline.step()
        return policy_loss.item(), baseline_loss.item()
    
    def save_model(self, path):
        policy_path = os.path.join(path, "best_policy_net.pth")
        baseline_path = os.path.join(path, "best_baseline_net.pth")
        torch.save(self.policy_net.state_dict(),  policy_path)
        torch.save(self.baseline_net.state_dict(),  baseline_path)
    
if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env = env.unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCEAgent(state_dim, action_dim)
    best_avg_reward = -np.inf
    swanlab.init(
    project="MountainCar-v0",
    experiment_name="REINFORCE-baseline-MountainCar-v0",
    config={
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": agent.gamma,
        "lr": agent.optimizer_policy.defaults['lr'],
        "episode" : 1000,
    }
    )
    for episode in tqdm(range(swanlab.config["episode"])):
        state, _ =env.reset()
        states, actions, rewards = [], [], []
        x_min=10
        x_max=-10
        while True:
            action =agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            x_now= state[0]
            x_next= next_state[0]
            v_now= state[1]
            v_next= next_state[1]

            reward+= 9.8*(get_height(x_next) - get_height(x_now))+ (v_next**2-v_now**2)/2
            

            # if x<=-0.8:
            #     reward+=10
            # if next_state[0] >= x_max:
            #     reward+=10*(x-x_max)
            #     x_max = next_state[0]
            # elif next_state[0] <= x_min:
            #     reward+=5*(x_min-x)
            #     x_min = next_state[0]
            
            # reward+=abs(next_state[1])*5

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            done = terminated or truncated
            state = next_state
            if done:
                break
        train_reward = sum(rewards)
        print(f"Episode: {episode + 1}, Total Reward: {train_reward}", flush=True)
        loss = agent.learn(states, actions, rewards)

        if (episode + 1) % 20 == 0:
            eval_env = gym.make("MountainCar-v0")
            eval_env = eval_env.unwrapped
            total_rewards = []
            for _ in range(10):
                state, _ = eval_env.reset()
                total_reward=0
                done = False
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated, _ = eval_env.step(action)
                    total_reward += reward
                    state = next_state
                    done = terminated or truncated
                total_rewards.append(total_reward)
            avg_reward = np.mean(total_rewards)
            if avg_reward >= best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model("0510_REINFORCE_baseline_Mountaincar")
                print(f"Episode: {episode + 1}, Avg Reward: {avg_reward:.2f}")
            eval_env.close()
        swanlab.log(
            {"policy_loss": loss[0],
             "baseline_loss": loss[1],
             "best_avg_reward": best_avg_reward,
             "train_reward": train_reward,
             },
             step=episode+1
        )
    env.close()

    