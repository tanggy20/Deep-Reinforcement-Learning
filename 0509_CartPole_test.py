import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque
from CartPole_v1_REINFORCE import REINFORCEAgent



if __name__ == "__main__":
    env=gym.make("CartPole-v1",render_mode="human")
    # env=gym.make("CartPole-v1")
    state_reset=env.reset()

    agent= REINFORCEAgent(env.observation_space.shape[0],env.action_space.n,lr=1e-3,gamma=0.99)
    agent.policy_net.load_state_dict(torch.load("0509_REINFORCE_CartPole/best_model.pth"))
    total_rewards=[]
    for episode in range(1000):
        state,_ = env.reset()
        total_reward=0
        steps=0
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated ,_ = env.step(action)
            total_reward += reward
            done= terminated or truncated
            state=next_state
            steps+=1
            if done :
                break
        total_rewards.append(total_reward)
        print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {steps}",flush=True)
    print(f" Average Scores: {np.mean(total_rewards)}")
    plt.plot(total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE CartPole-v1")
    plt.show()

    
    env.close()