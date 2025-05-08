# 🎮 DDQN LunarLander-v2

本项目实现了基于 PyTorch 的 **Double Deep Q-Network (DDQN)** 算法，用于解决 OpenAI Gym 中的 **LunarLander-v2** 问题。

---

## 📌 项目简介

LunarLander-v2 是一个经典的强化学习任务，目标是让飞船安全着陆在指定平台上。我们使用 DDQN 算法提升学习稳定性，防止传统 DQN 的过估计问题。

核心技术包括：

- 经验回放（Replay Buffer）
- 目标网络（Target Network）
- epsilon-greedy 策略
- swanlab 实验记录

---

## 🧩 环境依赖

请先安装以下依赖：

```bash
pip install gym numpy torch swanlab tqdm matplotlib
````

### 📦 依赖说明

* `gym`: 环境创建与交互
* `numpy`: 数据处理
* `torch`: 神经网络与优化器
* `swanlab`: 实验追踪与可视化
* `tqdm`: 训练进度条
* `matplotlib`: 训练结果可视化

---

## 📁 项目结构

```plaintext
.
├── LunarLander_v2_DDQN.py              # 主训练脚本
├── 0507_LunarLander_v2_test.py         # 测试验证脚本
├── best_model.pth                      # 训练过程中保存的最佳模型
└── README.md                           # 项目说明文档
```

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
pip install gym numpy torch swanlab tqdm matplotlib
```

### 2️⃣ 运行训练

```bash
python LunarLander_v2_DDQN.py 
```

默认会训练 600 个回合，并每 10 个回合进行一次评估。

训练期间会：

* 保存表现最好的模型到 `best_model.pth`
* 使用 swanlab 实时记录 reward、epsilon 等指标

---

## 🧠 模型架构

```text
Input: state_dim (8)
↓
Linear(8 → 36) + ReLU
↓
Linear(36 → 36) + ReLU
↓
Linear(36 → action_dim (4))
↓
Output: Q-values for each action
```

---

## 📊 模型评估与加载

### 加载最佳模型：

```python
import torch

model_path = 'best_model.pth'
model = torch.load(model_path)
```

---

## 🔍 DDQN 核心逻辑简述

* 使用当前网络选择动作（最大 Q 值）
* 使用目标网络估计下一状态的 Q 值
* 使用 `TD Target = r + γ * Q_target(s', argmax(Q_eval(s')))`
* 每隔若干步同步目标网络

---

## 🧪 使用 Swanlab 可视化指标

初始化：

```python
swanlab.init(
    project="LunarLander-v2",
    experiment_name="DDQN-LunarLander-v2",
    config={
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.99,
        "batch_size": 128,
        "episode": 600,
    }
)
```

日志记录：

```python
swanlab.log({
    "reward": total_reward,
    "epsilon": agent.epsilon,
    "eval_best_avg_reward": agent.best_avg_reward,
}, step=episode + 1)
```



## 🚀 训练结果

下图展示了经过 600 个 episode 后，训练过程中的奖励变化曲线。我们可以看到，随着训练的深入，奖励逐渐趋于稳定，表明智能体在不断学习并掌握任务的关键策略。训练初期，因随机探索奖励较低，但随着训练步数增加，智能体逐渐优化其行为，奖励呈现收敛趋势。

![训练结果](https://github.com/user-attachments/assets/630b5c0d-dd37-4596-98bd-26a572fc0fee)

### 观察：
- 初期由于探索行为较多，奖励值波动较大。
- 训练到一定回合数后，奖励趋于稳定，说明智能体开始成功地完成任务。

---

## 🔍 验证结果

在训练完成后，使用最佳模型参数对模型进行了 1000 个 episode 的验证。在验证过程中，我们将 epsilon 设置为 0，关闭了探索，确保验证结果完全基于已学习的策略。下图展示了验证过程中每个回合的奖励曲线。

![验证结果](https://github.com/user-attachments/assets/ea45cf53-2256-42f8-8518-94253b030a9b)

### 观察：
- 验证过程中的 reward 曲线保持稳定，验证智能体的策略在没有探索的情况下能够维持一致的表现。
- 最终，经过 1000 个 episode 的验证，模型的平均奖励为 **261.368**，证明了智能体在测试环境中的可靠性和有效性。

---


## 📎 参考

* [OpenAI Gym Docs](https://www.gymlibrary.dev/)
* [Deep Q-Learning Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Swanlab 实验管理工具](https://swanlab.ai)

---

## 🏁 实践建议

* 根据建模问题，合理调整Q网络层数与维度
* 合理设置 epsilon 衰减，避免探索过早停止，也需要避免后期epsilon过大，奖励一直波动
* 使用多次评估平均 reward 作为保存模型依据

---





