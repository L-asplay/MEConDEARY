import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
#from torch.cuda.amp import autocast, GradScaler

from utils.log_utils import log_dqn
from nets.graph_encoder import Normalization

class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        # 特征提取层：全连接、ReLU激活，再接 LayerNorm 保证稳定性
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            Normalization(256, 'batch')
            # nn.LayerNorm(256)
        )
        # 优势流（A(s,a)），输出尺寸依赖于候选动作数
        self.advantage = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, action_dim)
        )
        # 价值流（V(s)）
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x 形状：[batch_size, n, feature_dim]
        x = self.feature(x)  # 输出形状：[batch_size,  256]
        advantage = self.advantage(x)  # [batch_size,  action_dim]
        value = self.value(x)          # [batch_size,  1]
        # 计算优势均值，保证优势流和价值流融合后的 Q 值平衡
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        return q_values  # 最终输出形状：[batch_size, n]

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQNAgent(nn.Module):
    """
    DQNAgent 同时维护在线网络与目标网络，通过 optimize 方法利用经验数据计算
    当前 Q 值与目标 Q 值的均方误差，并利用 Adam 优化器进行梯度更新。
    目标网络定期通过 update_target 同步在线网络参数，以保证训练稳定性。
    
    为便于前向推理，DQNAgent 实现了 __call__ 方法，
    因此在 evaluate_model 中可直接使用 Selectlayer(state) 获取 Q 值。
    """
    def __init__(self, feature_dim, action_size, select_size, dqn_device, logstep, lr=1e-5, gamma=0.99):

        super(DQNAgent, self).__init__()

        self.select_size = select_size
        self.gamma = gamma
        self.model = DQN(feature_dim, action_size)
        self.target_model = DQN(feature_dim, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target()  # 初始时同步在线和目标网络

        self.memory = ReplayBuffer(capacity = 51200)
        self.buff_size = 25600
        self.batch_size = 512

        self.dqn_device = dqn_device
        self.log_step = logstep
    
    def set_epsilon(self, x):
        self.gamma = x
    
    def get_epsilon(self):
        return self.gamma

    def getmemory(self):
        return len(self.memory)
    
    def getbuff_szie(self):
        return self.buff_size

    def bufferpush(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def get_select_size(self):
        return self.select_size
    
    def pool(self, x):
        return x.contiguous().view(x.size(0), -1)

    def forward(self, x):
        px = self.pool(x)
        return self.model(px)

    def update_target(self):
        """将在线网络参数同步到目标网络中，确保目标 Q 值稳定。"""
        self.target_model.load_state_dict(self.model.state_dict())

    def optimize(self, tb_logger, step):
        """
        通过采样经验数据，依据 Bellman 方程计算目标 Q 值，
        使用 MSE 损失更新在线网络参数，返回当前损失。
        """
        if len(self.memory) < self.buff_size:
            return None  # 数据不足时跳过更新
        # 从经验池采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = self.pool(torch.stack(states)).to(self.dqn_device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(-1).to(self.dqn_device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.int64).unsqueeze(-1).to(self.dqn_device)
        next_states = self.pool(torch.stack(next_states)).to(self.dqn_device)
        dones = torch.tensor(np.array(dones), dtype=torch.int64).unsqueeze(-1).to(self.dqn_device)
        # 计算当前状态下采样动作的 Q 值
        q_values = self.model(states).gather(1, actions)
        # 目标网络计算下一状态下所有动作的最大 Q 值
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(-1)
        # 根据 Bellman 方程构造目标 Q 值
            expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Logging
        if step % int(self.log_step) == 0:
            log_dqn(loss.item(), tb_logger, step)
        del states, actions, rewards, next_states, dones, q_values, loss

def ModelWithSelector(Selectlayer, model, data): 
    """
    利用选择器网络（DQNAgent，即 Selectlayer）和 attention model（model）对输入 x 执行 m 次动作选择，
    每一步屏蔽已选点、累计 log 概率，最终将选择结果传入 attention model 得到 cost，
    并记录选择过程中的状态转移信息供 DQN 更新使用。
    """

    x = torch.cat((data["task_position"]-data["UAV_start_pos"], data["time_window"]/(10), data["IoT_resource"], data["task_data"], data["CPU_circles"]), dim=-1)

    batch_size, n, _ = x.size()
    device = x.device
    m = Selectlayer.get_select_size()

    selected_mask = torch.zeros((batch_size, n), device=device)  # 用来标记已选择的任务
    selections = []       # 存储每个步骤的选择任务
    dqn_transitions = []  # 存储状态转移信息

    # 将 x 和 selected_mask 拼接为新的 state
    state = torch.cat((x, selected_mask.unsqueeze(-1)), dim=-1)  # 拼接 x 和 selected_mask
    next_state = state.clone()  # 初始化 next_state 与 state 相同

    for i in range(m):
        # 计算当前状态下的 Q 值，并对已选任务进行屏蔽
        q_values = Selectlayer(state)
        # 将已选择任务的 Q 值设置为负无穷，避免重复选择
        q_values = q_values.masked_fill(selected_mask == 1, -float('inf'))
        # epdilon greed
        """probs = F.softmax(q_values, dim=-1)
        taken = (torch.rand(batch_size, device=device) < Selectlayer.get_epsilon()).int()
        actions = taken*torch.multinomial(probs, 1).squeeze(-1) + (1-taken)*q_values.argmax(dim=-1)"""
        actions = q_values.argmax(dim=-1)
        selections.append(actions)
        # 更新状态，标记已选择任务
        selected_mask[torch.arange(batch_size), actions] = 1  # 更新已选择的任务
        # 最后一步标记为 done，奖励为 cost（稍后更新）
        done = torch.full((batch_size,), False, device=device)
        reward = torch.zeros(batch_size, device=device)
        if i == m - 1:
            done = torch.full((batch_size,), True, device=device)
        # 更新 next_state，保持任务特征信息（state 和 selected_mask 的拼接）
        next_state = torch.cat((x, selected_mask.unsqueeze(-1)), dim=-1)  # 更新拼接的 next_state
        # 记录状态转移
        dqn_transitions.extend([{
            'state': state[b].detach().cpu(),
            'action': actions[b].item(),
            'reward': reward[b].item(),
            'next_state': next_state[b].detach().cpu(),
            'done': done[b].item()
        } for b in range(batch_size)])
        # 将更新后的 next_state 作为新的 state
        state = next_state.clone()
        del q_values,actions,done,reward,next_state

    selections = torch.stack(selections, dim=1) + 1 # (batch_size, m)
    # 将选择的任务传入注意力模型，计算 cost
    cost, log_likelihood = model(data, selections)
    # 更新奖励为负的 cost
    for idx, transition in enumerate(dqn_transitions[-batch_size:]):
        transition['reward'] = -cost[idx].item()  # 奖励为 cost 的负值

    return cost, log_likelihood, dqn_transitions

def EvalForRam(Selectlayer, model, data):
    """
    利用选择器网络（DQNAgent，即 Selectlayer）和 attention model（model）对输入 x 执行 m 次动作选择，
    每一步屏蔽已选点、累计 log 概率，最终将选择结果传入 attention model 得到 cost，
    并记录选择过程中的状态转移信息供 DQN 更新使用。
    """

    x = torch.cat((data["task_position"]-data["UAV_start_pos"], data["time_window"]/(10), data["IoT_resource"], data["task_data"], data["CPU_circles"]), dim=-1)

    batch_size, n, _ = x.size()
    device = x.device
    m = Selectlayer.get_select_size()

    selected_mask = torch.zeros((batch_size, n), device=device)  # 用来标记已选择的任务
    selections = []       # 存储每个步骤的选择任务

    for i in range(m):
        # 将 x 和 selected_mask 拼接为新的 state
        state = torch.cat((x, selected_mask.unsqueeze(-1)), dim=-1)  # 拼接 x 和 selected_mask
        # 计算当前状态下的 Q 值，并对已选任务进行屏蔽
        q_values = Selectlayer(state)
        # 将已选择任务的 Q 值设置为负无穷，避免重复选择
        q_values = q_values.masked_fill(selected_mask == 1, -float('inf'))
        # 直接取 argmax，选最大 Q 值对应的动作
        actions = q_values.argmax(dim=-1)
        selections.append(actions)
        # 更新状态，标记已选择任务
        selected_mask[torch.arange(batch_size), actions] = 1  # 更新已选择的任务
        del q_values

    selections = torch.stack(selections, dim=1) + 1 # (batch_size, m)
    # 将选择的任务传入注意力模型，计算 cost
    cost, log_likelihood = model(data, selections)
    return cost, log_likelihood, None