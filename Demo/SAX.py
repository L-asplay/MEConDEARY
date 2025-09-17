import torch
import numpy as np
from tqdm import tqdm
#pip install scikit-opt
from sko.SA import SA_TSP
from Demo.comute import compute_uav_energy, compute_total_energy
from Demo.comute import switched_capacitance, v

def skoSA(local_mask, data, 
          steps: int = 250,
          T_max: float = 100.0,
          T_min: float = 1e-2,):

    #local_mask [batch,N]
    assert local_mask.size(0) == 1, "仅支持 batch=1" 

    x = torch.cat((data["task_position"]-data["UAV_start_pos"], data["time_window"], 
        data["task_data"], data["CPU_circles"]), dim=-1)
    loc_energy = (switched_capacitance * (data["IoT_resource"] ** (v - 1)) * data["CPU_circles"] * (local_mask.float().unsqueeze(-1))).sum(dim=1)

    # 1. 获取 UAV 需要执行的任务索引
    uav_mask = ~local_mask[0]   # [N]
    uav_indices = torch.nonzero(uav_mask, as_tuple=False).squeeze(1)  # shape [M]
    M = uav_indices.size(0)
    x_uav = x[0, uav_indices]      # [M, dim]
    x_np = x_uav.cpu().numpy()     # -> numpy

    progress_bar = tqdm(total=steps, desc="SA", ncols=100, ascii=True)

    # 2. 实现 energy
    def energy(route_np):
        cost = loc_energy.item() + compute_uav_energy(route_np, x_np, data["UAV_resource"].item())
        progress_bar.update(1)  
        return cost

    # 3. 使用 sko.sa 优化
    init_route = np.arange(M)
    sa = SA_TSP(func=energy,
            x0=init_route,
            T_max=T_max,
            T_min=T_min,
            L=steps,
            max_stay_counter=75)
    best_route_np, best_energy = sa.run()

    progress_bar.close() 

    # 4. 还原为原始任务编号（即 x 中的索引）
    best_route = uav_indices[best_route_np]  # shape [M]
    best_route = best_route.unsqueeze(0)     # shape [1, M]
    best_energy = torch.tensor([best_energy], dtype=torch.float32, device=x.device)  # [1]
    
    return best_energy, best_route


def SAforPlot(local_mask, data, 
          steps: int = 30,
          T_max: float = 50.0,
          T_min: float = 5e-2,):

    #local_mask [batch,N]
    assert local_mask.size(0) == 1, "仅支持 batch=1" 

    x = torch.cat((data["task_position"]-data["UAV_start_pos"], data["time_window"], 
        data["task_data"], data["CPU_circles"]), dim=-1)
    loc_energy = (switched_capacitance * (data["IoT_resource"] ** (v - 1)) * data["CPU_circles"] * (local_mask.float().unsqueeze(-1))).sum(dim=1)

    # 1. 获取 UAV 需要执行的任务索引
    uav_mask = ~local_mask[0]   # [N]
    uav_indices = torch.nonzero(uav_mask, as_tuple=False).squeeze(1)  # shape [M]
    M = uav_indices.size(0)
    x_uav = x[0, uav_indices]      # [M, dim]
    x_np = x_uav.cpu().numpy()     # -> numpy

    # 2. 实现 energy
    def energy(route_np):
        cost = loc_energy.item() + compute_uav_energy(route_np, x_np, data["UAV_resource"].item())
        return cost

    # 3. 使用 sko.sa 优化
    init_route = np.arange(M)
    sa = SA_TSP(func=energy,
            x0=init_route,
            T_max=T_max,
            T_min=T_min,
            L=steps,
            max_stay_counter=15)
    _, best_energy = sa.run()
    # 4. 还原为原始任务编号（即 x 中的索引）
    best_energy = torch.tensor(best_energy, dtype=torch.float32, device=x.device)  # [1]
    return best_energy





# 需要确保合法的route
def perturb(route):
    assert 0, "to be complete SA.py/perturb"
    """
    批量 2-opt 扰动操作  route: [B, N]
    return: [B, N]，每个样本随机反转一段路径
    """
    B, N = route.shape
    device = route.device
    # 生成不同的 i, j（i != j）
    i = torch.randint(0, N - 1, (B,), device=device)
    j = torch.randint(1, N, (B,), device=device)
    i, j = torch.minimum(i, j), torch.maximum(i, j)
    mask = i != j
    # 创建索引张量
    arange = torch.arange(N, device=device).unsqueeze(0).expand(B, N)  # [B, N]
    # 构造 mask，用于判断哪些位置需要翻转
    flip_mask = (arange >= i.unsqueeze(1)) & (arange < j.unsqueeze(1)) & mask.unsqueeze(1)
    # 创建 route 的副本
    new_route = route.clone()
    # 获取需要翻转的部分的索引（注意按 batch 反向）
    idx = flip_mask.nonzero(as_tuple=False)  # shape [?, 2]，每行是 (b, k)
    # 将翻转段按 batch 分组、翻转
    from collections import defaultdict
    flip_dict = defaultdict(list)
    for b, k in idx.tolist():
        flip_dict[b].append(k)
    for b, positions in flip_dict.items():
        new_route[b, positions] = new_route[b, positions[::-1]]  # 翻转该段路径
    return new_route


def sa_algorithm(local_mask, data, steps=100, temp_start=100.0, temp_decay=0.98):
    assert 0, "to be complete SA.py/sa_algorithm"
    """
    模拟退火TSP路径优化，支持批量处理
    返回：best_route (batch, max_task)，best_energy (batch,)
    """

    x = torch.cat((data["task_position"]-data["UAV_start_pos"], data["time_window"]/(10), 
       data["IoT_resource"], data["task_data"], data["CPU_circles"]), dim=-1)

    batch, N = local_mask.shape
    device = x.device

    uav_mask = ~local_mask  # shape (batch, N)
    task_nums = uav_mask.sum(dim=1)  # 每个batch中UAV任务数量
    max_task = task_nums.max().item()

    # 获取每个 batch 中 uav 执行的任务索引（填充为 -1）
    uav_indices = torch.full((batch, max_task), -1, dtype=torch.long, device=device)
    for b in range(batch):
        idx = torch.nonzero(uav_mask[b], as_tuple=False).squeeze()
        uav_indices[b, :idx.size(0)] = idx

    # 构造初始坐标 coords, shape (batch, max_task, dim)
    coords = torch.gather(x, 1, uav_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
    # 构造初始路径：0,1,2,... (每个 batch 中无效位置标记为 -1)
    route = torch.arange(max_task, device=device).unsqueeze(0).expand(batch, -1)

    best = route.clone()
    best_energy = compute_total_energy(best, coords)
    current, current_energy = best.clone(), best_energy.clone()
    temp = torch.full((batch,), temp_start, device=device)

    for _ in range(steps):
        new_route = perturb(current)
        new_energy = compute_total_energy(new_route, coords)
        delta,prob = (new_energy - current_energy), torch.exp(-delta / temp)
        
        accept = (delta < 0) | (torch.rand_like(prob) < prob)
        current[accept] = new_route[accept]
        current_energy[accept] = new_energy[accept]

        better = new_energy < best_energy
        best[better] = new_route[better]
        best_energy[better] = new_energy[better]

        temp *= temp_decay

    # 还原原始任务点编号顺序：从 uav_indices 中按照 best 排序
    batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1, max_task)
    best_route = uav_indices[batch_idx, best]
    return best_energy, best_route
