import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque
from Mountaincar_REINFORCE_baseline import REINFORCEAgent



if __name__ == "__main__":
    # env=gym.make("MountainCar-v0",render_mode="human")
    env=gym.make("MountainCar-v0")
    env=env.unwrapped
    state_reset=env.reset()

    agent= REINFORCEAgent(env.observation_space.shape[0],env.action_space.n,lr=1e-3,gamma=0.99)
    agent.policy_net.load_state_dict(torch.load("0510_REINFORCE_baseline_Mountaincar/best_policy_net.pth"))
    agent.baseline_net.load_state_dict(torch.load("0510_REINFORCE_baseline_Mountaincar/best_baseline_net.pth"))
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
    plt.title("REINFORCE_baseline MountainCar-v0")
    plt.show()
    env.close()