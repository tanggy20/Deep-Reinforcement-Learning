
# REINFORCE 算法在 CartPole-v1 环境中的应用

本项目实现了 REINFORCE 算法，用于解决 CartPole-v1 环境中的控制问题，使用了 Gym 库。REINFORCE 是一种基于蒙特卡洛的策略梯度方法，直接根据每一轮的奖励来更新策略。

## 🚀 项目概述

CartPole-v1 环境要求智能体尽可能长时间地保持竿子在小车上的平衡，通过对小车施加力来实现。环境提供了连续的状态空间，目标是找到一个策略，最大化总奖励。

## 🧑‍💻 环境要求

在运行本项目之前，请确保安装以下依赖项：

- Python 3.x
- `gym`（用于创建环境）：通过 `pip install gym` 安装
- `torch`（用于深度学习模型）：通过 `pip install torch` 安装
- `matplotlib`（用于绘制图表）：通过 `pip install matplotlib` 安装
- `numpy`（用于数值计算）：通过 `pip install numpy` 安装

你可以使用以下命令来安装依赖项：

```bash
pip install gym torch matplotlib numpy
````

## 📝 项目结构

```
REINFORCE-CartPole-v1/
│
├── CartPole_v1_REINFORCE.py  # 运行主脚本进行训练
├── 0509_CartPole_test.py     # 运行测试验证脚本
├── best_model.pth            # 训练好的最优模型
└── README.md                 # 项目描述
```

## ⚙️ 如何运行

1. 克隆项目：

2. 确保已安装所有必要的依赖项。

3. 运行 `CartPole_v1_REINFORCE.py` 脚本开始训练智能体：

```bash
python CartPole_v1_REINFORCE.py
```

### 评估

你可以使用提供的预训练模型 (`best_model.pth`) 来评估智能体的表现。该模型已经在 CartPole-v1 环境中经过 2000 轮训练。

### 训练

如果你想从头开始训练智能体，可以通过 `CartPole_v1_REINFORCE.py` 脚本进行训练。你可以调整如学习率、折扣因子等参数，修改 `REINFORCEAgent` 类中的设置。

## 📊 结果展示

运行 `0509_CartPole_test.py` 脚本后，控制台中会打印每个回合的总奖励，同时会显示奖励随着回合数变化的曲线图，帮助你可视化智能体的训练表现。


为了更清晰地说明REINFORCE算法的核心数学公式，我将其补充到README的`REINFORCE 算法原理`部分。以下是包括REINFORCE核心数学公式的更新版本：

---

## 🧑‍🏫 REINFORCE 算法原理

REINFORCE算法是一种基于蒙特卡洛的策略梯度方法，用于强化学习中的策略优化。它通过采样的方式估计期望回报（即奖励），并使用回报的梯度来优化策略。其核心思想是根据每个回合的累计奖励来调整策略，从而使得智能体能够更好地在环境中表现。

### 算法的基本原理：：

REINFORCE算法通过以下公式来更新策略：

$$
\theta_{t+1} = \theta_t + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) G_t
$$

其中：

* $\theta_t$ 为策略的参数。
* $\pi_\theta(a_t|s_t)$ 为在状态 $s_t$ 下采取动作 $a_t$ 的概率，表示当前策略的输出。
* $G_t$ 为从时间步 $t$ 开始的累计回报（即该回合的总奖励）。
* $\alpha$ 为学习率，控制策略更新的步长。

$G_t$ 通过以下公式计算：

$$
G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k
$$

* $r_k$ 是在时间步 $k$ 处的即时奖励。
* $\gamma$ 是折扣因子，用于折扣未来奖励。
* $T$ 是回合结束的时间步。

### 关键点：

* **蒙特卡洛估计**：REINFORCE通过对每个回合的累计奖励进行蒙特卡洛估计来更新策略，避免了对环境模型的依赖。
* **高方差**：由于REINFORCE的更新依赖于回合的最终奖励，因此它通常会面临较高的方差，导致学习过程可能不稳定。
* **无模型方法**：REINFORCE算法属于无模型强化学习方法，不需要对环境的转移概率和奖励函数有任何先验知识。


## 🏋️‍♂️ 训练结果

在训练过程中，智能体通过与环境交互逐步优化策略，提升表现。以下是训练过程中智能体在每个回合的结果变化图：

![训练结果](https://github.com/user-attachments/assets/e0ca78e2-7433-4c4b-a843-503b4aabb3c4)

## 🎯 验证结果

经过1000回合的验证测试，智能体表现出良好的稳定性，验证期间的平均奖励为 **491.922**。以下是验证过程中的奖励曲线：

![验证结果](https://github.com/user-attachments/assets/d01a4d8a-e5e8-4013-a9d0-799a6952f66e)

从这些结果中，可以看出，模型在经过训练后取得了不错的成绩，并在验证过程中展现了稳定性。


