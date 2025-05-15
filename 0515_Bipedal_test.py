import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque
from TD3_Bipedal import TD3Agent



if __name__ == "__main__":
    env=gym.make("BipedalWalker-v3",render_mode="human")
    # env=gym.make("BipedalWalker-v3")
    state_reset=env.reset()

    agent= TD3Agent(env.observation_space.shape[0],env.action_space.shape[0],fc1_dim=400,fc2_dim=300,lr_actor=3e-3,lr_critic=2e-3,gamma=0.99,tau=0.005,batch_size=128,memory_size=100000,noise_clip=0.5)
    agent.actor.load_state_dict(torch.load("0515_TD3_Bipedal-v3/best_actor.pth"))
    agent.critic1.load_state_dict(torch.load("0515_TD3_Bipedal-v3/best_critic1.pth"))
    agent.critic2.load_state_dict(torch.load("0515_TD3_Bipedal-v3/best_critic2.pth"))
    total_rewards=[]
    for episode in range(1000):
        state,_ = env.reset()
        total_reward=0
        steps=0
        while True:
            action = agent.select_action(state, False)
            next_state, reward, terminated, truncated ,_ = env.step(action)
            total_reward += reward
            done= terminated or truncated
            state=next_state
            steps+=1
            if done :
                break
        print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {steps}",flush=True)
        total_rewards.append(total_reward)
    print(f"Average Reward: {np.mean(total_rewards)}")
    plt.plot(total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("TD3 BipedalWalker-v3")
    plt.show()

    
    
    env.close()