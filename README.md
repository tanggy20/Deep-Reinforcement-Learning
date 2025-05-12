
# Dueling DQN 在 Acrobot-v1 环境中的实现 🚀

本项目实现了 **Dueling Deep Q-Network (Dueling DQN)** 算法，用于解决 **Acrobot-v1** 环境中的控制问题，使用 **OpenAI Gym** 和 **PyTorch**。Dueling DQN 相比传统的 DQN 引入了两个流：一个是状态价值流，一个是动作优势流，从而提高了某些环境下的学习效率。

## 🧑‍🏫 项目概述 📚

该项目的目标是使用强化学习训练智能体，控制一个双摆机器人（Acrobot）摆动并使自由端达到一定高度。Dueling DQN 算法通过使用两个独立的流（一个表示状态的价值，一个表示每个动作的优势）来逼近 Q 值，学习最佳策略以最大化累积奖励。

### 关键特点：
- **Dueling Deep Q-Network 架构** 提高了学习的稳定性。
- **Epsilon-greedy 探索策略**，采用逐步衰减的 epsilon 值来平衡探索与利用。
- **目标网络更新**，使用软更新（由 tau 参数控制）。
- **Swanlab 集成**，用于实验追踪和日志记录。

## 🛠️ 环境要求

运行本项目需要以下依赖项：

- **Python 3.x**
- **gym**：用于环境创建与交互
- **numpy**：用于数值计算
- **torch**：用于构建和训练神经网络模型
- **matplotlib**：用于结果可视化
- **swanlab**：用于实验追踪
- **tqdm**：用于训练进度条显示

### 安装依赖项

使用 `pip` 安装所需依赖：

```bash
pip install gym numpy torch matplotlib swanlab tqdm
````

## 📁 项目结构

```
.
├── Dueling_DQN_Acrobot.py               # 主训练脚本
├── 0512_Acrobot_test.py                 # 测试验证脚本
├── best_model.pth                       # 最佳模型
└── README.md                            # 项目文档
```

## 🚀 算法概述

### Dueling DQN 架构

Dueling DQN 架构包括两个独立的流：

1. **价值流**（V）：估计状态的价值。
2. **优势流**（A）：估计每个动作的优势。

最终的 Q 值通过以下公式计算：

$$
Q(s, a) = V(s) + (A(s, a) - \text{mean}(A(s, \cdot)))
$$

### 关键超参数：

* **学习率**（`lr`）：0.001
* **折扣因子**（`gamma`）：0.99
* **目标网络更新频率**（`tau`）：0.005
* **Epsilon 衰减**：Epsilon 从 1.0 开始，并衰减到 0.01
* **批量大小**（`batch_size`）：128
* **记忆库大小**（`memory_size`）：100,000
* **训练回合数**：1000

## 📊 实验设置与日志记录

本项目使用 **Swanlab** 进行实验追踪。以下是日志记录的配置代码示例：

```python
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
```

### 评估

每训练 10 个回合，模型都会进行评估，评估时将 epsilon 设置为 0（即不进行探索）。如果评估得到的平均奖励高于当前最佳模型的奖励，当前模型将被保存。

## 🎯 训练结果

训练过程和模型评估结果会被记录和可视化。每个回合会显示当前的总奖励、epsilon 值和步骤数。每训练 10 个回合，会对模型进行评估并保存表现最好的模型。

## 💾 模型保存

训练过程中表现最好的模型会被保存为 `best_model.pth` 文件：

```python
agent.save_model("best_model.pth")
```

## 📈 可视化

训练奖励和模型评估结果会被记录并可视化。可以通过 **Swanlab** 生成的日志文件查看训练进度和结果。

## 📝 使用方法

运行训练脚本：

```bash
python Dueling_DQN_Acrobot.py
```

这将启动训练，训练智能体在 **Acrobot-v1** 环境中进行强化学习，使用代码中定义的超参数进行训练。

## 💬 运行结果

### 训练结果 📈

经过 **1000** 回合的训练，智能体的训练过程展示了明显的收敛趋势。以下是训练奖励的曲线图，随着回合数的增加，智能体的性能逐渐提高，最终趋于稳定。

<img width="1032" alt="1747030039048" src="https://github.com/user-attachments/assets/59d75752-d875-43f6-a415-f26e30783d07" />


### 验证结果 🎯

在 **1000** 回合的验证中，智能体的平均奖励为 **-81.89**。大部分情况下，智能体能够在 **100** 步内达到目标，显示出较好的效果。这表明模型已经成功学会了如何在该环境中进行有效控制。

![image](https://github.com/user-attachments/assets/c3a7dc21-7185-4b9e-8e62-d6b568bb2ae7)

