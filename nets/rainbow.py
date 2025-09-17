# coding: utf-8
"""
Typical usage
-------------
```python
agent = DQNAgent(feature_dim=state_dim,
                 action_size=num_actions,
                 select_size=...,       # unchanged
                 dqn_device="cuda:0",
                 logstep=100)

q_values = agent(state_tensor)          # forward pass unchanged
agent.bufferpush(s, a, r, s_next, done) # same
loss = agent.optimize(tb_logger, step)  # same
```
"""
from __future__ import annotations
import math
import random
from collections import deque
from typing import Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.log_utils import log_dqn
from nets.graph_encoder import Normalization
# ---------------------------- Noisy linear layer ---------------------------- #
class NoisyLinear(nn.Module):
    """Factorised Gaussian Noisy layer (Fortunato 2018)."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("eps_weight", torch.empty(out_features, in_features))

        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer("eps_bias", torch.empty(out_features))

        self.sigma_init = sigma_init / math.sqrt(in_features)
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.mu_weight, -bound, bound)
        nn.init.constant_(self.sigma_weight, self.sigma_init)
        nn.init.uniform_(self.mu_bias, -bound, bound)
        nn.init.constant_(self.sigma_bias, self.sigma_init)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.eps_weight.copy_(epsilon_out.ger(epsilon_in))
        self.eps_bias.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.training:
            weight = self.mu_weight + self.sigma_weight * self.eps_weight
            bias = self.mu_bias + self.sigma_bias * self.eps_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return F.linear(input, weight, bias)

# --------------------------- Prioritised Replay ---------------------------- #
class PrioritisedReplayBuffer:
    """Proportional PER with a simple sum-tree backed by numpy array."""

    def __init__(self, capacity: int, alpha: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.size = 0

        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        self.data: List[Tuple] = [None] * capacity  # type: ignore
        self.eps = 1e-6

    def _propagate(self, idx: int, change: float):
        while idx >= 1:
           self.tree[idx] += change
           idx //= 2


    def _update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def total(self) -> float:
        return self.tree[1]

    def add(self, *transition, priority: float):
        p = (priority + self.eps) ** self.alpha
        idx = self.pos + self.capacity
        self.data[self.pos] = transition
        self._update(idx, p)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx
        right = left + 1
        if left >= len(self.tree):
            return idx
        if self.tree[left] == 0 and self.tree[right] == 0:
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def sample(self, batch_size: int, beta: float):
        segment = self.total() / batch_size
        samples = []
        indices = []
        weights = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx = self._retrieve(1, s)
            data_idx = idx - self.capacity
            transition = self.data[data_idx]
            samples.append(transition)
            indices.append(idx)

            p = max(self.tree[idx], 1e-6)  # 防止为 0
            prob = p / max(self.total(), 1e-6)  # 防止除 0
            weight = (self.size * prob) ** (-beta)

            weights.append(weight)
        weights = torch.tensor(weights, dtype=torch.float32)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        for idx, p in zip(indices, priorities):
            self._update(idx, p + self.eps)

    def __len__(self):
        return self.size

# ---------------------------- Categorical Net ------------------------------ #
class RainbowDQN(nn.Module):
    """Dueling-C51 network with NoisyLinear layers."""

    def __init__(self, input_dim: int, action_dim: int, atom_size: int = 51, v_min: float = -10.0, v_max: float = 10.0):
        super().__init__()
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.v_min, self.v_max = v_min, v_max
        self.support = torch.linspace(v_min, v_max, atom_size)
        self.base_dropout = 0.2

        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            Normalization(256, 'batch')
            #nn.LayerNorm(256)
        )
        
        self.advantage = nn.Sequential(
            NoisyLinear(256, 128), # nn.Linear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, action_dim * atom_size)
            # nn.Dropout(p=self.base_dropout),  
            # nn.Linear(128, action_dim * atom_size) 
        )
        self.value = nn.Sequential(
            NoisyLinear(256, 128), # nn.Linear(256, 128), 
            nn.ReLU(),
            NoisyLinear(128, atom_size)
            # nn.Dropout(p=self.base_dropout),
            # nn.Linear(128, atom_size) 
        )
        self._reset_noise()

    # ------------------ Dropout annealing helper ------------------ #
    def set_dropout(self, p: float):
        """Dynamically change dropout probability (0 <= p < 1)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def _reset_noise(self):
        """pass"""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
        
       
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        q = torch.sum(dist * self.support.to(x.device), dim=2)
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        adv = self.advantage(x).view(-1, self.action_dim, self.atom_size)
        val = self.value(x).view(-1, 1, self.atom_size)
        dist = val + adv - adv.mean(dim=1, keepdim=True)
        return F.softmax(dist, dim=2)

# ----------------------------- Agent Class --------------------------------- #
class DQNAgent(nn.Module):
    """Rainbow agent – interface identical to your previous DQNAgent."""

    def __init__(self,
                 feature_dim: int,
                 action_size: int,
                 select_size: int,
                 dqn_device: Union[torch.device, str],
                 logstep: int,
                 *,
                 lr: float = 5e-5,
                 gamma: float = 0.99,
                 n_steps: int = 3,
                 atom_size: int = 51,
                 v_min: float = -10.0,
                 v_max: float = 10.0,
                 capacity: int = 100000,
                 alpha: float = 0.4,
                 beta_start: float = 0.4,
                 beta_frames: int = 200000,
                 tau: float = 0.003):
        super().__init__()

        self.select_size = select_size
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = torch.device(dqn_device)
        self.log_step = logstep
        self.tau = tau

        self.model = RainbowDQN(feature_dim, action_size, atom_size, v_min, v_max).to(self.device)
        self.target_model = RainbowDQN(feature_dim, action_size, atom_size, v_min, v_max).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Replay buffer
        self.memory = PrioritisedReplayBuffer(capacity, alpha)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_idx = 0

        # Pre‑compute support
        self.register_buffer("support", torch.linspace(v_min, v_max, atom_size))
        self.delta_z = (v_max - v_min) / (atom_size - 1)

        # Multi‑step buffer for N‑step returns
        self.n_buffer = deque(maxlen=n_steps)

    # ------------------------- Interface保持不变 ------------------------- #
    def set_epsilon(self, x):  # 不再使用 – 保留兼容
        pass

    def get_epsilon(self):
        return 0.0  # NoisyNet exploration无需 ε

    def getmemory(self):
        return len(self.memory)

    def getbuff_szie(self):
        return self.memory.capacity

    def get_select_size(self):
        return self.select_size

    # ------------------------- Helper functions --------------------------- #
    @staticmethod
    def _pool(x):  # flatten [B, N, F] → [B, N*F]
        return x.contiguous().view(x.size(0), -1)

    def forward(self, x):
        px = self._pool(x).to(self.device)
        return self.model(px)

    # ---------------------- Experience management ------------------------ #
    def _append_nstep(self, *transition):
        self.n_buffer.append(transition)
        if len(self.n_buffer) < self.n_steps:
            return None
        # Form N‑step transition
        R, s, a, next_s, done = 0.0, self.n_buffer[0][0], self.n_buffer[0][1], self.n_buffer[-1][3], self.n_buffer[-1][4]
        for idx, (_, _, r, _, d) in enumerate(self.n_buffer):
            R += (self.gamma ** idx) * r
            if d:
                break
        return (s, a, R, next_s, done)

    def bufferpush(self, state, action, reward, next_state, done):
        trans = (state, action, reward, next_state, done)
        nstep = self._append_nstep(*trans)
        if nstep:
            # Use max priority for new sample
            max_prio = self.memory.tree[self.memory.capacity: self.memory.capacity + self.memory.size].max(initial=1.0)
            self.memory.add(*nstep, priority=max_prio)

    # ------------------------------ Update -------------------------------- #
    def optimize(self, tb_logger, step):
        if len(self.memory) < 1024:
            return None
        self.frame_idx += 1

        # ---------- Dropout annealing ----------
        ratio = min(1.0, self.frame_idx / 200_000)
        new_p = 0.05 + 0.5 * (0.2 - 0.05) * (1 + math.cos(math.pi * ratio))
        self.model.set_dropout(new_p)
        self.target_model.set_dropout(new_p)  # keep target net dropout一致
        # ---------------------------------------

        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame_idx / self.beta_frames)

        batch_size = 512
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size, beta)

        # Tensorise
        states = self._pool(torch.stack(states)).to(self.device)
        next_states = self._pool(torch.stack(next_states)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)
        weights = weights.unsqueeze(-1).to(self.device)

        # -------- current distribution -------- #
        dist = self.model.dist(states)
        dist_a = dist.gather(1, actions.unsqueeze(-1).expand(-1, -1, self.model.atom_size)).squeeze(1)  # (B, atom)

        # --------- target distribution -------- #
        with torch.no_grad():
            next_dist = self.model(next_states)  # Q-values for online net
            next_actions = next_dist.argmax(1)  # Double DQN – choose with online
            target_dist = self.target_model.dist(next_states)
            target_dist = target_dist[range(batch_size), next_actions]  # (B, atom)

            Tz = rewards + (1 - dones) * (self.gamma ** self.n_steps) * self.support
            Tz = Tz.clamp(self.model.v_min, self.model.v_max)
            b = (Tz - self.model.v_min) / self.delta_z
            l, u = b.floor().long(), b.ceil().long()
            l = l.clamp(0, self.model.atom_size - 1)
            u = u.clamp(0, self.model.atom_size - 1)

            m = torch.zeros_like(target_dist) 
            for i in range(self.model.atom_size):
                m.view(-1).index_add_(0, (l + i * batch_size).view(-1), (target_dist * (u.float() - b)).view(-1))
                m.view(-1).index_add_(0, (u + i * batch_size).view(-1), (target_dist * (b - l.float())).view(-1))

        loss_per_sample = F.kl_div((dist_a + 1e-8).log(), m, reduction='none').sum(1)
        if torch.isnan(weights).any():
             print("Warning: NaN in importance weights!")
             weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=0.0)
        loss = (loss_per_sample * weights.squeeze(-1)).mean()

        # optimisation step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # update priorities
        # 计算优先级   ---------- 关键修改 ----------
        priorities = loss_per_sample.detach().clamp(0, 10.0).cpu().numpy()
        self.memory.update_priorities(indices, priorities)

        # soft target update
        for t_param, s_param in zip(self.target_model.parameters(), self.model.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) + s_param.data * self.tau)

        # reset noisy layers after each optimisation
        self.model.train();  self.model._reset_noise()  # type: ignore
        self.target_model.train();  self.target_model._reset_noise()

        # Logging
        if step % self.log_step == 0:
            log_dqn(loss.item(), tb_logger, step)
        return loss.item()

