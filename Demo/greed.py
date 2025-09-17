import torch

def congreed(algorithm, data, M):

    loc = data["task_position"]
    batch_size, N, _ = loc.size()
    device = loc.device

    selected = torch.zeros((batch_size, N), dtype=torch.bool, device=device)  # 初始本地集合 A = 空集
    result = torch.full((batch_size, M), -1, dtype=torch.long, device=device)

    for step in range(M):
        energy_list = []
        for i in range(N):
            print("Comuting best to", step, "-|-", i)
            mask = selected.clone() # 已经选过的任务不能再试探
            mask[:, i] = True  # 试探将任务 i 加入本地执行集合 A
            energy, _ = algorithm(mask, data)  # 计算总能耗
            energy_list.append(energy.unsqueeze(1))

        energy_matrix = torch.cat(energy_list, dim=1)  # shape: (batch, N)
        energy_matrix[selected] = float('inf')  # 过滤掉已选择的任务

        best_choice = torch.argmin(energy_matrix, dim=1)  # 在每个 batch 中选出使得能耗最小的任务索引
        result[:, step] = best_choice # shape: (batch,)
        selected[torch.arange(batch_size), best_choice] = True  # 添加到本地集合 A 中
        print("Greed: The", step+1, "Node selection", best_choice, "with energy --", energy_matrix.gather(1, best_choice.unsqueeze(1)).squeeze(1))
    final_cost, final_pi = algorithm(selected, data)
  
    return result, final_cost, final_pi