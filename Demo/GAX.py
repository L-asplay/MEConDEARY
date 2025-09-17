import torch
import numpy as np
from tqdm import tqdm
# pip install scikit-opt
from sko.GA import GA_TSP
from Demo.comute import compute_uav_energy, compute_total_energy
from Demo.comute import switched_capacitance, v


def skoGA(
    local_mask,
    data,
    steps: int = 125,          # 进化代数（iterations） 
    pop_size: int = 64,       # 种群规模
    prob_mut: float = 0.15      # 变异概率
):

    assert local_mask.size(0) == 1, "仅支持 batch=1"

    # --------------- 1. 预处理 & IoT 本地能耗 ----------------
    x = torch.cat(
        (
            data["task_position"] - data["UAV_start_pos"],  # Δpos
            data["time_window"],
            data["task_data"],
            data["CPU_circles"],
        ),
        dim=-1,
    )                               # [1,N,dim]
    loc_energy = (
        switched_capacitance
        * (data["IoT_resource"] ** (v - 1))
        * data["CPU_circles"]
        * local_mask.float().unsqueeze(-1)
    ).sum(dim=1)                    # [1] → IoT 本地执行部分能耗

    # --------------- 2. 提取 UAV 需要执行的任务 ---------------
    uav_mask = ~local_mask[0]                       # [N]
    uav_indices = torch.nonzero(uav_mask, as_tuple=False).squeeze(1)  # [M]
    M = uav_indices.size(0)
    if M == 0:                                     # UAV 无任务
        return loc_energy.clone(), uav_indices.view(1, 0)

    x_uav = x[0, uav_indices]                      # [M, dim]
    x_np = x_uav.cpu().numpy()                     # 转 numpy

    # --------------- 3. 定义适应度 / 能耗函数 ----------------
    # 进度条：总调用次数 ≈ steps * pop_size
    pbar = tqdm(total=steps * pop_size, desc="GA", ncols=100, ascii=True)

    def energy(route_np: np.ndarray) -> float:
        """
        route_np: ndarray 长度 M，包含 [0..M-1] 的一个排列
        """
        cost = loc_energy.item() + compute_uav_energy(
            route_np, x_np, data["UAV_resource"].item()
        )
        pbar.update(1)
        return cost

    # --------------- 4. 运行 GA_TSP --------------------------
    ga = GA_TSP(
        func=energy,
        n_dim=M,
        size_pop=pop_size,
        max_iter=steps,
        prob_mut=prob_mut,
        # 若希望关闭内部打印：verbose=False
    )
    best_route_np, best_energy_scalar = ga.run()
    pbar.close()

    # --------------- 5. 结果后处理 ---------------------------
    best_route = uav_indices[best_route_np]        # [M]
    best_route = best_route.unsqueeze(0)           # [1,M]
    best_energy = torch.tensor(
        [best_energy_scalar], dtype=torch.float32, device=x.device
    )                                              # [1]

    return best_energy, best_route


def GAforPlot(
    local_mask,
    data,
    steps: int = 30,          # 进化代数（iterations） 
    pop_size: int = 10,       # 种群规模
    prob_mut: float = 0.15     # 变异概率
):

    assert local_mask.size(0) == 1, "仅支持 batch=1"

    # --------------- 1. 预处理 & IoT 本地能耗 ----------------
    x = torch.cat(
        (
            data["task_position"] - data["UAV_start_pos"],  # Δpos
            data["time_window"],
            data["task_data"],
            data["CPU_circles"],
        ),
        dim=-1,
    )                               # [1,N,dim]
    loc_energy = (
        switched_capacitance
        * (data["IoT_resource"] ** (v - 1))
        * data["CPU_circles"]
        * local_mask.float().unsqueeze(-1)
    ).sum(dim=1)                    # [1] → IoT 本地执行部分能耗

    # --------------- 2. 提取 UAV 需要执行的任务 ---------------
    uav_mask = ~local_mask[0]                       # [N]
    uav_indices = torch.nonzero(uav_mask, as_tuple=False).squeeze(1)  # [M]
    M = uav_indices.size(0)
    if M == 0:                                     # UAV 无任务
        return loc_energy.clone(), uav_indices.view(1, 0)
    x_uav = x[0, uav_indices]                      # [M, dim]
    x_np = x_uav.cpu().numpy()                     # 转 numpy
    # --------------- 3. 定义适应度 / 能耗函数 ----------------
    def energy(route_np: np.ndarray) -> float:
        """
        route_np: ndarray 长度 M，包含 [0..M-1] 的一个排列
        """
        cost = loc_energy.item() + compute_uav_energy(
            route_np, x_np, data["UAV_resource"].item()
        )
        return cost
    # --------------- 4. 运行 GA_TSP --------------------------
    ga = GA_TSP(
        func=energy,
        n_dim=M,
        size_pop=pop_size,
        max_iter=steps,
        prob_mut=prob_mut,
        # 若希望关闭内部打印：verbose=False
    )
    _, best_energy_scalar = ga.run()
    # --------------- 5. 结果后处理 ---------------------------
    best_energy = torch.tensor(
        best_energy_scalar[0], dtype=torch.float32, device=x.device
    )                                             # [1]
    return best_energy



def crossover(parent1, parent2):
    """顺序交叉（Order Crossover）"""
    batch, N = parent1.shape
    offspring = torch.full_like(parent1, -1)
    for b in range(batch):
        i, j = torch.sort(torch.randint(0, N, (2,), device=parent1.device)).values
        offspring[b, i:j] = parent1[b, i:j]
        remaining = [x for x in parent2[b].tolist() if x not in offspring[b, i:j]]
        pos = 0
        for idx in range(N):
            if offspring[b, idx] == -1:
                offspring[b, idx] = remaining[pos]
                pos += 1
    return offspring

def mutate(route, rate=0.2):
    """随机交换两点"""
    batch, N = route.shape
    mutated = route.clone()
    for b in range(batch):
        if torch.rand(1).item() < rate:
            i, j = torch.randint(0, N, (2,), device=route.device)
            mutated[b, i], mutated[b, j] = mutated[b, j], mutated[b, i]
    return mutated


def ga_algorithm(local_mask: torch.Tensor, x: torch.Tensor,
                 generations=50, pop_size=20, mutation_rate=0.2):
    """
    遗传算法TSP求解，用于 UAV 路径能耗优化
    """
    batch, N, dim = x.shape
    device = x.device
    energy = torch.zeros(batch, device=device)

    for b in range(batch):
        uav_indices = torch.nonzero(~local_mask[b], as_tuple=False).squeeze()
        if uav_indices.numel() <= 1:
            continue

        coords = x[b, uav_indices]  # shape (k, dim)
        num_tasks = coords.shape[0]

        # 初始种群
        population = torch.stack([torch.randperm(num_tasks, device=device)
                                  for _ in range(pop_size)], dim=0).repeat(batch, 1, 1)

        for _ in range(generations):
            costs = compute_total_energy(population.view(-1, num_tasks), coords.repeat(pop_size, 1, 1))
            costs = costs.view(batch, pop_size)
            _, topk = torch.topk(costs, k=pop_size//2, dim=1, largest=False)

            parents = torch.gather(population, 1,
                                   topk.unsqueeze(-1).expand(-1, -1, num_tasks))

            offspring = crossover(parents[:, 0], parents[:, 1])
            offspring = mutate(offspring, rate=mutation_rate)

            population = torch.cat([parents, offspring.unsqueeze(1)], dim=1).squeeze(1)

        final_costs = compute_total_energy(population[:, 0], coords)
        energy[b] = final_costs

    return energy



