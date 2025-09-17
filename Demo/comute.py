import torch
import numpy as np
import math
#一些参数
Rc = 4.0  # coverage / bias
UAV_p = 3
# UAV fly
height = 1
g = 0.98  
speed = 1.0
quantity_uav = 2
Cd = 0.3
A = 0.1
air_density = 1.225
P_fly = air_density * A * Cd * pow(speed, 3) / 2 + quantity_uav * g * speed
P_stay = pow(speed, 3)
# Iot device energy compute
switched_capacitance = 3e1
v = 4
# transmit
B = 1e5
g0 = 20
G0 = 5
upload_P = 3
noise_P = -9
hm = 0
d_2 = pow(Rc, 2) + pow(height, 2)
upload_speed = B * math.log2(1 + g0 * G0 * upload_P / pow(noise_P, 2) / (pow(hm, 2) + d_2))
 
def compute_uav_energy(route: np.ndarray, x: np.ndarray, Fresou, penalty=200): # for 30-20 the punish is 200
  
    x = x[route]
    coords = x[:,:2] #(m,dim)
    pos = np.array([[0.0, 0.0]], dtype=coords.dtype)  
    positions = np.concatenate([pos, coords, pos], axis=0)  
    diffs = positions[1:] - positions[:-1]
    dists = np.sqrt((diffs * diffs).sum(axis=1))
    
    cur_time = 0.0
    uav_energy = 0.0
    for i in range(len(x)) :
      tiwind, demand, fcycle= x[i][2:4], x[i][4], x[i][5]

      fly_t = dists[i]/speed
      arrl_time = cur_time + fly_t
      wait_t = max(tiwind[0] - arrl_time, 0)
      exe_t = demand/upload_speed + fcycle/Fresou
      cur_time = arrl_time + wait_t + exe_t
      if cur_time > tiwind[1] :
        return penalty
      #uav_energy = fly_energy + wait_energy + exe_energy
      uav_energy = uav_energy + fly_t * P_fly + wait_t * P_stay + exe_t * UAV_p
    #back to pos
    uav_energy = uav_energy + dists[-1] / speed * P_fly
    return uav_energy

def EvalForDemo(Selectlayer, model, data):
    """
    利用选择器网络（DQNAgent，即 Selectlayer）和 attention model（model）对输入 x 执行 m 次动作选择，
    每一步屏蔽已选点、累计 log 概率，最终将选择结果传入 attention model 得到 cost，
    并记录选择过程中的状态转移信息供 DQN 更新使用。
    """
    x = torch.cat((data["task_position"]-data["UAV_start_pos"], data["time_window"]/(10), data["IoT_resource"], data["task_data"], data["CPU_circles"]), dim=-1)
    batch_size, n, _ = x.size()
    device = x.device
    m = Selectlayer.get_select_size()

    selected_mask = torch.zeros((batch_size, n), device=device)  
    selections = []   
    for i in range(m):
        # 将 x 和 selected_mask 拼接为新的 state
        state = torch.cat((x, selected_mask.unsqueeze(-1)), dim=-1)  # 拼接 x 和 selected_mask
        # 计算当前状态下的 Q 值，并对已选任务进行屏蔽
        q_values = Selectlayer(state)
        # 将已选择任务的 Q 值设置为负无穷，避免重复选择
        q_values = q_values.masked_fill(selected_mask == 1, -float('inf'))
        # 将 Q 值转化为概率分布并采样
        # 原先的代码：先 softmax，再采样
        #probs = F.softmax(q_values, dim=-1)
        #actions = torch.multinomial(probs, 1).squeeze(-1)
        actions = q_values.argmax(dim=-1)
        selections.append(actions)
        # 更新状态，标记已选择任务
        selected_mask[torch.arange(batch_size), actions] = 1  # 更新已选择的任务
        del q_values
    selections = torch.stack(selections, dim=1) + 1 # (batch_size, m)
    # 将选择的任务传入注意力模型，计算 cost
    cost, log_likelihood, pi = model(data, selections, return_pi=True)
    return cost, log_likelihood, pi












def compute_total_energy(route, x, demand, fcycles, upload_speed, Fresourse,
                         fresourse):
    assert 0, "to be complete compute/total_energy"
    """
    route: (batch, M) long tensor，表示每个 batch 中 UAV 的任务执行顺序
    其余参数定义不变
    """
    batch, N, dim = x.shape
    device = x.device

    # 构造 UAV mask
    uav_mask = torch.zeros((batch, N), dtype=torch.bool, device=device)
    uav_mask[torch.arange(batch)[:, None], route] = True

    # 本地任务 mask
    local_mask = ~uav_mask
    local_mask_f = local_mask.float()

    # 本地能耗
    loc_energy = (switched_capacitance * (fresourse ** (v - 1)) * fcycles * local_mask_f).sum(dim=1)

    # 提取 UAV 路径/需求/计算
    uav_coords = torch.gather(x, 1, route.unsqueeze(-1).expand(-1, -1, dim))
    uav_demand = torch.gather(demand, 1, route)
    uav_fcycles = torch.gather(fcycles, 1, route)

    fly_energy = []
    wait_energy = []
    uav_energy = []

    for b in range(batch):
        if route[b].numel() <= 1:
            fly_energy.append(torch.tensor(0.0, device=device))
            wait_energy.append(torch.tensor(0.0, device=device))
            uav_energy.append(torch.tensor(0.0, device=device))
            continue

        path = uav_coords[b]  # 按顺序访问
        dist = ((path[1:] - path[:-1]) ** 2).sum(-1).sqrt()
        fly_t = dist / speed
        fly_e = fly_t.sum() * P_fly

        mec_t = uav_demand[b] / upload_speed + uav_fcycles[b] / Fresourse
        calc_e = mec_t.sum() * UAV_p

        fly_energy.append(fly_e)
        wait_energy.append(torch.tensor(0.0, device=device))
        uav_energy.append(calc_e)

    fly_energy = torch.stack(fly_energy)
    wait_energy = torch.stack(wait_energy)
    uav_energy = torch.stack(uav_energy)

    total_energy = loc_energy + fly_energy + wait_energy + uav_energy
    breakpoint()
    return total_energy