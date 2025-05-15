
# TD3 算法 - BipedalWalker-v3 环境

## 项目概述 📚

本项目实现了深度强化学习中的 **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** 算法，应用于 **BipedalWalker-v3** 环境中的行走控制任务。TD3 是一种改进的 DDPG 算法，通过引入双重 Q 网络和延迟更新机制来提高训练稳定性。

### 项目目标
训练一个智能体在 **BipedalWalker-v3** 环境中成功控制机器人进行行走。

---

## 方法原理 💡

### TD3 算法简介
TD3 是 **DDPG** 算法的改进版本，主要通过以下几点来提升训练的稳定性：

1. **双 Q 网络**（Twin Q Networks）：使用两个独立的 Q 网络来计算目标 Q 值，然后选择两个 Q 网络中的最小值。这可以缓解 **Q 值过高** 的问题，减少过估计偏差。
   
2. **延迟更新**（Delayed Updates）：TD3 在更新 Actor 网络时每隔一段时间才会进行一次。这有助于减少因过频繁更新造成的不稳定问题。

3. **目标网络的噪声**（Target Policy Smoothing）：在计算目标 Q 值时，为目标网络的动作添加噪声，避免过拟合并增加训练的随机性。

### 算法流程
1. **初始化**：初始化 Actor 和 Critic 网络及其目标网络。
2. **选择动作**：根据当前的状态选择动作，加入噪声来探索环境。
3. **存储过渡**：将当前的状态、动作、奖励、下一个状态和完成标志存入经验回放池。
4. **训练网络**：从经验回放池中随机采样一批数据，并利用 TD3 算法训练 Critic 和 Actor 网络。
5. **更新目标网络**：通过软更新的方式将目标网络的参数向当前网络的参数逼近。
![image](https://github.com/user-attachments/assets/6aa89e56-65d6-4cf2-8b2e-85416a6d0a83)
---



## 环境要求 ⚙️

在运行该项目之前，请确保已安装以下依赖项：

- Python 3.x
- `gym`：用于创建强化学习环境（安装命令：`pip install gym`）
- `numpy`：用于数值计算（安装命令：`pip install numpy`）
- `torch`：用于实现神经网络和训练（安装命令：`pip install torch`）
- `swanlab`：用于实验记录和追踪（安装命令：`pip install swanlab`）
- `tqdm`：用于显示进度条（安装命令：`pip install tqdm`）
---

## 项目结构 🗂️

```
.
├── TD3_Bipedal.py         # TD3智能体实现
├── 0515_Bipedal_test.py   # 测试验证脚本
├── best_actor.pth         # 最佳演员网络
├── best_critic1.pth       # 最佳评论家1网络
├── best_critic2.pth       # 最佳评论家2网络
└── README.md              # 项目说明
```

---

## 超参数设置 🛠️

以下是本项目使用的主要超参数：

* `fc1_dim`：第一个全连接层的维度（默认：400）
* `fc2_dim`：第二个全连接层的维度（默认：300）
* `lr_actor`：Actor 网络的学习率（默认：3e-3）
* `lr_critic`：Critic 网络的学习率（默认：2e-3）
* `gamma`：折扣因子（默认：0.99）
* `tau`：目标网络软更新的参数（默认：0.005）
* `batch_size`：每次更新的批大小（默认：128）
* `memory_size`：经验回放缓冲区的大小（默认：1000000）
* `noise_clip`：噪声裁剪的阈值（默认：0.5）

---

## 如何运行 🏃‍♂️

1. 克隆该项目到本地：

   ```bash
   git clone https://github.com/yourusername/TD3-BipedalWalker-v3.git
   cd TD3-BipedalWalker-v3
   ```

2. 运行训练脚本：

   ```bash
   python TD3_Bipedal.py
   ```

3. 训练过程中，程序会在每 10 轮评估一次智能体的表现，并保存最优模型。

---

## 实验追踪 📊

本项目使用 [Swanlab](https://github.com/swanlab) 进行实验追踪。在训练过程中，实验的每个步骤都将记录并上传到 Swanlab，以便于实验结果的跟踪与分析。

---

## 评估与模型保存 🏆

训练过程中，每当智能体在评估时获得更好的平均奖励时，模型将会被保存。保存的模型包括：

* `best_actor.pth`：最优 Actor 网络
* `best_critic1.pth`：最优 Critic1 网络
* `best_critic2.pth`：最优 Critic2 网络

这些模型文件将被保存在 `"0515_TD3_Bipedal-v3"` 目录中。

---

## 结果展示 📈

在训练过程中，智能体会在 **BipedalWalker-v3** 环境中执行以下任务：

- **🎯 训练奖励**：记录每个训练回合的奖励，展示智能体在环境中的学习过程。
- **🔍 评估奖励**：每 10 轮训练后，会评估当前智能体的平均奖励，并显示最优奖励。

### 🏋️‍♂️ 训练过程奖励曲线
下图展示了智能体在训练过程中的奖励变化趋势，横轴为训练回合数，纵轴为每回合获得的奖励：

![训练奖励曲线](https://github.com/user-attachments/assets/0f827aa9-90bf-40ba-a8b7-e66f16d71770)

### 🧑‍🏫 测试验证奖励曲线
以下图展示了经过 1000 回合训练后，智能体在测试环境中的表现。横轴为回合数，纵轴为每回合的测试奖励：

![测试验证奖励曲线](https://github.com/user-attachments/assets/787df323-2dba-4726-80be-357994f5725f)

### 🏆 训练结果
经过 1000 回合的训练与评估，智能体的 **平均奖励为：295.55** 🎉。这表明智能体已经学习到一定的行走策略，但仍有进一步提升的空间。

---

通过不断训练和评估，智能体的表现会逐步提升，最终能够在 **BipedalWalker-v3** 环境中达到稳定的行走状态 🚶‍♂️。

