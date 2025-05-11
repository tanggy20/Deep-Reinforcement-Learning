# REINFORCE with Baseline for MountainCar-v0 🚗🏔️

## 项目概述 📚

这是一个基于REINFORCE算法的强化学习项目，应用于OpenAI Gym中的MountainCar-v0环境。该项目使用了基线（Baseline）方法来减少策略梯度的方差，优化了训练过程。

## REINFORCE算法与基线方法原理 💡

### REINFORCE算法

REINFORCE是一种基于策略梯度（Policy Gradient）的强化学习算法，通过直接优化策略函数来学习智能体的行为。REINFORCE的核心思想是通过蒙特卡洛方法估计每个动作的回报（即未来的奖励），然后根据回报调整策略。

#### 数学原理：

我们的目标是最大化智能体在环境中的总奖励（即累积奖励）。我们可以通过优化策略来最大化目标函数：

$J(\theta) = \mathbb{E_S} [V_\pi(S)]$


 $\theta$ 是策略网络的参数，![Uploading 1746928521253.png…]()
是随机变量形式的状态价值函数。REINFORCE方法通过计算策略梯度来更新参数：

$\nabla J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(A|S) Q_\pi(S,A) \right]$

其中，$\pi_\theta(A|S)$ 是随机变量形式的策略，梯度随机变量形式表达需要对$S$与$A$求期望。通过这个梯度，我们可以更新策略，使得执行更高回报的动作的概率更高。

但是因为期望形式不易计算，实际我们使用**蒙特卡洛近似**，从环境中观测一个状态$s$,根据策略网络随机抽样得到动作$a$。于是策略梯度可近似写为：
$$
g(s, a; \theta) =  Q_{\pi}(s, a)  \cdot \nabla_{\theta} \ln \pi(a | s; \theta)
$$

同时用实际观测的回报$R$来近似动作价值函数$Q_\pi(s, a)$，于是就可以进一步近似写为：

$$
\tilde{g}(s, a; \theta) =  R  \cdot \nabla_{\theta} \ln \pi(a | s; \theta)
$$


### 基线方法

REINFORCE算法的缺点是方差较大，因为回报的估计往往不准确。为了减小这个方差，我们引入了基线（Baseline）方法。基线方法通过引入一个状态值函数来估计当前状态下的期望回报，这个期望回报与实际回报的差值就是所谓的优势函数（Advantage）。

#### 数学原理：

引入基线后，我们的目标变为最大化如下的目标函数：

$J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a|s) (R - V_\pi(s)) \right]$

其中，$V_\pi(s)$ 是模型的状态价值函数，我们用神经网络$v_\pi(s;w)$ 对状态$s$的价值进行估计。$R - V_\pi(s)$ 叫做 **优势函数**（Advantage Function），它表示了实际回报与期望回报之间的差异。通过使用优势函数，算法能够更加稳定地更新策略，减少回报的方差。

基线网络的作用是通过拟合状态值函数来减少策略梯度的方差，避免了高方差问题，使得学习过程更加稳定。


具体原理讲解可以参看：
https://blog.csdn.net/qq_40206371/article/details/125012106


## Reward Shaping 在 MountainCar-v0 中的应用 💡

### 背景
在 **MountainCar-v0** 环境中，原始奖励是相当稀疏的。智能体只有在成功达到山顶时才会获得奖励，而在其他时候，奖励几乎为零。这导致智能体可能需要大量的训练回合才能获得足够的反馈，学习过程非常缓慢。为了加速训练过程，我们可以使用 **Reward Shaping** 技术，向奖励信号中添加更多的信息，帮助智能体更快地理解任务。

### 物理模型
在 **MountainCar-v0** 环境中，奖励的设计可以参考物理模型，特别是势能和动能的变化，同时又要保证额外奖励不会影响基础奖励的目标。具体而言，我们可以通过以下物理公式来设计奖励：

$$ E(s') - E(s) = (E_g(s') + E_m(s')) - (E_g(s) + E_m(s)) $$

这里，$E(s)$ 表示智能体在某状态下的总能量，由势能（$E_g$）和动能（$E_m$）组成。势能与小车的高度相关，而动能与小车的速度相关。通过计算高度和速度的变化，我们可以为智能体提供更多的反馈信号。

### 设计奖励塑形函数

基于上述物理模型，我们设计了以下奖励塑形函数：

```python
def reward_shaping(state, next_state):
    """
    根据物理模型计算奖励塑形函数
    """
    # 计算势能变化
    h_s = get_height(state[0])  # 当前状态的小车高度
    h_s_prime = get_height(next_state[0])  # 下一状态的小车高度
    height_change = h_s_prime - h_s
    
    # 计算动能变化
    v_s = state[1]  # 当前状态的速度
    v_s_prime = next_state[1]  # 下一状态的速度
    kinetic_change = (v_s_prime ** 2 - v_s ** 2) / 2
    
    # 返回总的奖励变化
    reward = height_change + kinetic_change
    return reward
```
#### 解释：
height_change：表示小车的势能变化。通过小车位置的变化（即高度差），我们鼓励小车朝着山顶方向前进。

kinetic_change：表示小车的动能变化。速度增加时奖励增加，速度减少时奖励减少。通过这个项，智能体被鼓励加速移动。

这个奖励塑形函数使得智能体不仅仅依赖于最终的奖励信号，而是通过更多的中间奖励加速学习过程。

更多细节详情可参看：
https://zhuanlan.zhihu.com/p/378129617

## 项目结构 🗂️

.
├── Mountaincar_REINFORCE_baseline.py  # 主训练脚本
├── 0510_Mountaincar_test.py           # 测试验证脚本
├── best_policy_net.pth                # 最优策略网络用于选择动作
├── best_baseline_net.pth              # 最优基线网络用于计算策略价值
└── README.md                          # 项目说明



## 环境要求 ⚙️

- Python 3.x
- `gym`：用于创建强化学习环境
- `numpy`：用于数值计算
- `matplotlib`：用于可视化图表
- `torch`：用于实现神经网络和训练
- `swanlab`：用于实验记录和追踪
- `tqdm`：用于显示进度条


## 训练步骤 🏃‍♂️

### 1. 训练模型 🧑‍💻

你可以直接运行 `Mountaincar_REINFORCE_baseline.py` 来训练智能体。训练过程包括以下几个步骤：

* 初始化环境和智能体。
* 在每一回合中，智能体根据策略网络选择动作，获得奖励并更新策略。
* 每隔20个回合，评估一次模型，并在模型性能提升时保存模型。



### 2. 模型保存 💾

训练过程中，如果模型表现提升（基于平均奖励），会自动保存模型到 `0510_REINFORCE_baseline_Mountaincar/` 目录。保存的文件包括：

* `best_policy_net.pth`：保存的策略网络
* `best_baseline_net.pth`：保存的基线网络

### 3. 实验记录 📊

本项目使用了 `swanlab` 进行实验记录和跟踪。实验的主要参数包括：

* 状态维度（state\_dim）
* 动作维度（action\_dim）
* 折扣因子（gamma）
* 学习率（lr）
* 训练回合数（episode）

### 4. 评估模型 🎯

在每20个训练回合之后，程序会对智能体进行评估。每次评估时，智能体将在 `MountainCar-v0` 环境中进行10轮测试，计算平均奖励。如果平均奖励超过当前最佳奖励，将会保存新的模型。

## 奖励设计 🏅

在 `MountainCar-v0` 环境中，奖励函数经过修改，考虑了小车的高度差和速度差，具体的奖励计算方式如下：

```python
reward += (get_height(x_next) - get_height(x_now))*9.8 + (v_next**2 - v_now**2) * 0.5
```

### 可选奖励设计 💡：

* 你可以进一步调整奖励函数，使得小车在训练过程中获得更多的动力，或探索更高效的奖励结构。

## 结果展示 📈

### 训练结果📷
#### 都进行1000个回合的训练，同时去除了环境中200步截断的限制
#### 1.没有加额外奖励的训练曲线：

![image](https://github.com/user-attachments/assets/325cd3c2-e46a-4e93-9a72-62dcdf598d74)


#### 2.加额外奖励的训练曲线：

![image](https://github.com/user-attachments/assets/075ea375-ce89-4b70-a70b-fe281eef2f27)


### 验证结果📸

#### 同样进行1000个回合的验证

#### 1.没有加额外奖励的奖励曲线：

![image](https://github.com/user-attachments/assets/47ef2cdc-f36c-46f8-9121-c606d220b4b4)


平均奖励为：-2618.676

#### 2.加额外奖励的奖励曲线：

![image](https://github.com/user-attachments/assets/6c92a944-3225-4a08-863b-57867a86eab7)


平均奖励为：-2927.607

### 对比结论
加了额外奖励后，虽然收敛速度有所提高，但在这个问题中，并没有表现出特别明显的优势。

# 总结🚩
MountainCar-v0 是一个典型的奖励稀疏问题。对比其他方法网上实验效果，如 Q-table 和 DQN，使用 REINFORCE 方法的效果较差，尤其在200步内到达山顶的能力较弱。这可能是由于REINFORCE方法在训练过程中梯度较小，不容易进行有效的梯度更新。
