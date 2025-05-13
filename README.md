# DDPG 算法实现 - Pendulum-v1 环境 🚀

## 项目概述 📚

本项目实现了 **深度确定性策略梯度**（DDPG）算法，用于解决 OpenAI Gym 中的 **Pendulum-v1** 环境。该环境的目标是使智能体通过控制的自由端的扭矩使倒立摆保持倒立平衡位置。

DDPG 是一种 **基于 Actor-Critic 的强化学习方法**，采用了深度神经网络来逼近策略函数和价值函数。该算法具有高效性和可扩展性，适用于**连续动作空间**的强化学习任务。

## 依赖项 📦

在运行该项目之前，请确保安装以下依赖项：

- Python 3.x
- `gym`：用于创建强化学习环境
- `torch`：用于神经网络和训练
- `numpy`：用于数值计算
- `matplotlib`：用于绘制图表
- `swanlab`：用于实验记录和追踪
- `tqdm`：用于显示进度条

可以通过以下命令安装所需的依赖项：

```bash
pip install gym torch numpy matplotlib swanlab tqdm
````

## 项目结构 🗂️

```bash
.
├── DDPG_Pendulum.py            # 主训练脚本
├── 0513_Pendulum_test.py       # 测试验证脚本
├── README.md                   # 项目说明文件
└── best_actor.pth              # 最优策略网络（Actor）
└── best_critic.pth             # 最优价值网络（Critic）
```

## 算法原理 💡

**DDPG（Deep Deterministic Policy Gradient）** 是一种适用于连续动作空间的 **Actor-Critic** 算法。算法包括两个主要组件：

* **Actor**：生成具体的动作，基于当前状态通过策略网络进行选择。
* **Critic**：评估当前动作的价值，使用 Q-函数来评估。

### DDPG 数学原理 ✨

#### 1. 确定性策略梯度 (DPG)

确定性策略（Deterministic Policy）是与随机策略（Stochastic Policy）相对的。在随机策略中，给定一个状态，采取的动作是基于一个概率分布。而在确定性策略中，给定一个状态，策略决定了一个具体的动作。因此，确定性策略梯度的优化目标可以通过如下公式来描述：

$$
\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{s \sim \rho_{\pi}} \left[ \nabla_{\theta} \pi_{\theta}(s) \nabla_a Q_{\pi}(s, a) |_{a=\pi_{\theta}(s)} \right]
$$

其中， $\pi_{\theta}(s)$ 是给定状态 $s$ 时的确定性策略， $Q_{\pi}(s, a)$ 是在状态 $s$ 和动作 $a$ 下的 Q 值。

#### 2. DDPG 中的目标

DDPG 基于 **DPG**，使用 **双网络结构** 来优化策略。具体来说，DDPG 采用了 4 个网络：

* **Actor 当前网络**：生成基于当前状态的动作。
* **Actor 目标网络**：根据经验回放池中采样的下一状态 $s'$ 来选择最优动作。
* **Critic 当前网络**：评估当前动作的 Q 值。
* **Critic 目标网络**：用于计算目标 Q 值，帮助更新 Critic 当前网络。

### 目标网络和软更新

与 DQN 不同，DDPG 使用 **软更新** 来更新目标网络参数，公式为：

$$
\theta' = \tau \theta + (1 - \tau) \theta'
$$

$$
w' = \tau w + (1 - \tau) w'
$$

其中， $\tau$ 是软更新系数，通常取较小值（如 0.1 或 0.01）。这种软更新的方式比 DQN 中的硬更新更加稳定。

### Critic 和 Actor 的损失函数

* **Critic 网络损失函数**：与 DQN 类似，Critic 网络使用均方误差（MSE）损失函数来更新 Q 值：

$$
J(w) = \frac{1}{m} \sum_{j=1}^m \left( y_j - Q_{\phi}(S_j, A_j, w) \right)^2
$$

* **Actor 网络损失函数**：由于 DDPG 使用的是确定性策略，Actor 网络的损失函数可以通过以下公式计算：

$$
\nabla J(\theta) = -\frac{1}{m} \sum_{j=1}^m Q_{\pi}(S_i, A_i, w) \nabla_{\theta} \pi_{\theta}(S_i)
$$

### 3. 从 DPG 到 DDPG 的扩展

DDPG 通过引入 **目标网络** 和 **经验回放** 来稳定训练过程，类似于从 DQN 到 DDQN 的转变。通过这两个扩展，DDPG 能够更有效地学习和优化策略。


## 训练过程 ⚙️

### 初始化

首先，我们初始化 **Actor** 和 **Critic** 网络，以及目标网络。然后，我们设置优化器和 **经验回放（Replay Buffer）**。

### 训练过程

* 每个回合，智能体从环境中获得状态，并通过 **Actor** 网络选择动作。
* 动作会带来奖励和下一个状态，智能体将此信息存入回放缓冲区。
* 每隔一段时间，智能体从回放缓冲区中随机抽取一批数据进行训练，优化 **Actor** 和 **Critic** 网络。

### 评估

每经过一定的回合数，我们使用当前策略进行评估，并记录平均奖励。当平均奖励达到新的最佳值时，我们保存模型。

## 实验结果 📊

### 训练结果

在训练过程中，智能体逐渐学习如何控制摆锤，并能在环境中获得更高的奖励。以下是训练过程中获得的奖励曲线：

![Training Rewards](https://github.com/user-attachments/assets/630b5c0d-dd37-4596-98bd-26a572fc0fee)

### 测试结果 🎯

在训练完成后，我们对智能体进行了 1000 回合的评估，平均奖励为： **491.922**。

![Evaluation Rewards](https://github.com/user-attachments/assets/d01a4d8a-e5e8-4013-a9d0-799a6952f66e)

## 保存和加载模型 💾

训练过程中，当智能体达到新的最佳平均奖励时，模型会被保存至本地文件。你可以使用以下命令加载模型：

```python
actor_model = torch.load('best_actor.pth')
critic_model = torch.load('best_critic.pth')
```

## 运行脚本 🎬

在终端运行以下命令启动训练：

```bash
python main.py
```



如果您有任何问题或建议，请提交 issue 或直接联系我！😊


