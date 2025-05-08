# DQN CartPole-v1 🏋️‍♂️

本项目实现了一个 **深度Q网络（DQN）** 算法，用于解决 **CartPole-v1** 控制问题。该模型使用神经网络来逼近Q值函数，并使用DQN算法训练智能体，使其能够在该环境中保持杆平衡。

## 环境要求 ⚙️

在运行该项目之前，确保已安装以下依赖项：

- Python 3.x 🐍
- `gym` - 用于创建强化学习环境 (`pip install gym`)
- `numpy` - 用于数值计算 (`pip install numpy`)
- `matplotlib` - 用于绘制图表 (`pip install matplotlib`)
- `torch` - 用于实现神经网络和训练 (`pip install torch`)
- `swanlab` - 用于实验记录和追踪 (`pip install swanlab`)
- `tqdm` - 用于显示进度条 (`pip install tqdm`)

你可以使用以下命令一次性安装所有依赖项：

```bash
pip install gym numpy matplotlib torch swanlab tqdm
```

## 项目结构 📂

```plaintext
.
├── CartPole_0506.py            # 主训练脚本
├── 0506_CartPole_test.py       # 验证效果脚本
├── README.md                   # 项目文档
└── best_model.pth              # 保存的最佳模型
```

## 如何运行 🚀

1. 克隆该项目到本地机器：

    ```bash
    git clone https://github.com/yourusername/DQN-CartPole-v1.git
    cd DQN-CartPole-v1
    ```

2. 运行主训练脚本：

    ```bash
    python CartPole_0506.py
    ```

## 超参数 🧠

- **学习率 (lr)**: Adam优化器的学习率（默认值：1e-3）。
- **折扣因子 (gamma)**: 未来奖励的折扣因子（默认值：0.99）。
- **探索率 (epsilon)**: 选择随机动作的概率（初始值：1.0，最终衰减到 0.01）。
- **批量大小 (batch_size)**: 每次训练时从经验回放中抽取的样本数量（默认值：128）。
- **回合数 (episode)**: 训练的总回合数（默认值：600）。

## 模型训练 🏃‍♂️

### 训练的关键步骤 🔑

1. **动作选择（ε-greedy）**：
   - 在每一步，智能体选择是否探索环境或利用其学到的知识（Q值）。
   - 随着训练的进行，epsilon会逐渐减小，从而减少探索并增加利用。

2. **经验回放**：
   - 智能体将其经验存储在回放缓冲区中，并从中随机抽取批量数据进行训练。这样做有助于打破数据的时间相关性，提高训练的稳定性。

3. **目标网络更新**：
   - 每隔一定步数，目标网络（Q_target）会更新为评估网络（Q_eval）的权重，以保证训练的稳定性。

4. **损失计算**：
   - 通过均方误差（MSE）计算当前Q值和目标Q值之间的损失，并通过反向传播优化Q网络的权重。

### 评估 📊

- 每训练10个回合，智能体会在环境中进行评估，并记录其表现。
- 如果模型达到了新的最佳平均奖励，模型会被保存到磁盘。

### 实验记录 📝

本项目使用 **SwanLab** 来记录和追踪实验的配置和结果。以下是实验配置的示例：

```python
swanlab.init(
    project="CartPole-v1",
    experiment_name="DQN-CartPole-v1",
    config={
        "state_dim": state_dim,
        "action_dim": action_dim,
        "batch_size": agent.batch_size,
        "gamma": agent.gamma,
        "epsilon": agent.epsilon,
        "update_freq": agent.update_freq,
        "replay_buffer_size": len(agent.replay_buffer),
        "lr": agent.optimizer.defaults['lr'],
        "episode": 600,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.99,
    }
)
```

更多关于SwanLab的信息，可以参考 [SwanLab文档](https://docs.swanlab.cn/guide_cloud/general/what-is-swanlab.html)。

### 保存最佳模型 💾

训练结束时，会保存表现最好的模型（根据平均奖励）到 `best_model.pth`。你可以加载保存的模型来进一步评估或微调智能体：

```python
agent.Q_eval.load_state_dict(torch.load('best_model.pth'))
```
## 🏆 实验结果

通过我的测试，针对这个问题不需要过于复杂的网络结构，反而较小的网络能够达到很好的效果。经过1000个回合的训练，智能体的**平均奖励**如下：

### 训练结果 📈

- **1000 episodes** 的平均奖励为： **486.783**
  
### 训练过程图 🎯

以下是训练过程中智能体的表现：

![image](https://github.com/user-attachments/assets/9bcf0b26-95b6-4ad2-adf6-a58e0f3eaa87)

