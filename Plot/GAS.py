import numpy as np
import torch

from sko.SA import SA_TSP
from sko.GA import GA_TSP
from Demo.comute import speed, upload_speed, P_fly, P_stay, UAV_p, switched_capacitance, v 

def compute_energy(route_np, data, sel, Fresou, penalty=200):
    x = data[route_np]
    loc = x[:sel]
    uav = x[sel:]
    loc_energy = 0.0
    uav_energy = 0.0 

    for i in range(sel):
       fcycle, iotr =  loc[i][5], loc[i][6]
       loc_energy = loc_energy + switched_capacitance * (iotr ** (v - 1)) * fcycle

    coords = uav[:,:2] 
    pos = np.array([[0.0, 0.0]], dtype=coords.dtype)  
    positions = np.concatenate([pos, coords, pos], axis=0)  
    diffs = positions[1:] - positions[:-1]
    dists = np.sqrt((diffs * diffs).sum(axis=1))

    cur_time = 0.0
    for i in range(len(uav)) :
      tiwind, demand, fcycle= uav[i][2:4], uav[i][4], uav[i][5]

      fly_t = dists[i]/speed
      arrl_time = cur_time + fly_t
      wait_t = max(tiwind[0] - arrl_time, 0)
      exe_t = demand/upload_speed + fcycle/Fresou
      cur_time = arrl_time + wait_t + exe_t
      if cur_time > tiwind[1] :
        return penalty
      #uav_energy = fly_energy + wait_energy + exe_energy
      uav_energy = uav_energy + fly_t * P_fly + wait_t * P_stay + exe_t * UAV_p
    
    return loc_energy + uav_energy

def SAforPlot(data, size,
          steps: int = 40,
          T_max: float = 50.0,
          T_min: float = 5e-2,):
    
    x = torch.cat(
        (
            data["task_position"] - data["UAV_start_pos"],  # Δpos 2
            data["time_window"], # 2
            data["task_data"], # 1
            data["CPU_circles"], # 1
            data["IoT_resource"], # 1
        ),
        dim=-1,
    )

    x_np = x.squeeze(0).cpu().numpy()   

    # 2. 实现 energy
    def energy(route_np):
        cost = compute_energy(route_np, x_np, size, data["UAV_resource"].item())
        return cost

    # 3. 使用 sko.sa 优化
    init_route = np.arange(len(x_np))
    sa = SA_TSP(func=energy,
            x0=init_route,
            T_max=T_max,
            T_min=T_min,
            L=steps,
            max_stay_counter=15)
    rout, best_energy = sa.run()
 
    return best_energy

def GAforPlot(data, size,
    steps: int = 25,          # 进化代数（iterations） 
    pop_size: int = 10,       # 种群规模
    prob_mut: float = 0.1     # 变异概率
):
    
    x = torch.cat(
        (
            data["task_position"] - data["UAV_start_pos"],  # Δpos 2
            data["time_window"], # 2
            data["task_data"], # 1
            data["CPU_circles"], # 1
            data["IoT_resource"], # 1
        ),
        dim=-1,
    )

    x_np = x.squeeze(0).cpu().numpy()   

    # 2. 实现 energy
    def energy(route_np):
        cost = compute_energy(route_np, x_np, size, data["UAV_resource"].item())
        return cost

    # --------------- 4. 运行 GA_TSP --------------------------
    ga = GA_TSP(
        func=energy,
        n_dim=len(x_np),
        size_pop=pop_size,
        max_iter=steps,
        prob_mut=prob_mut,
        # 若希望关闭内部打印：verbose=False
    )
    route_scalar, best_energy_scalar = ga.run()

    best_energy = best_energy_scalar[0]
    return best_energy


def compute_energy2(route_np, uav, Fresou, penalty=200):
    x = uav[route_np]
    uav_energy = 0.0 

    coords = x[:,:2] 
    pos = np.array([[0.0, 0.0]], dtype=coords.dtype)  
    positions = np.concatenate([pos, coords, pos], axis=0)  
    diffs = positions[1:] - positions[:-1]
    dists = np.sqrt((diffs * diffs).sum(axis=1))

    cur_time = 0.0
    for i in range(len(uav)) :
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
    
    return uav_energy

def SAforPlot2(data, size,
          steps: int = 30,
          T_max: float = 30.0,
          T_min: float = 1e-1,):
    
    x = torch.cat(
        (
            data["task_position"] - data["UAV_start_pos"],  # Δpos 2
            data["time_window"], # 2
            data["task_data"], # 1
            data["CPU_circles"], # 1
            data["IoT_resource"], # 1
        ),
        dim=-1,
    )

    x_np = x.squeeze(0).cpu().numpy()   
    sel_indices = np.random.choice(len(x_np), size=size, replace=False)
    loc_mask = np.isin(np.arange(len(x_np)), sel_indices)
    loc = x_np[loc_mask]
    uav = x_np[~loc_mask]
    
    loc_energy = 0.0

    for i in range(len(loc)):
       fcycle, iotr =  loc[i][5], loc[i][6]
       loc_energy = loc_energy + switched_capacitance * (iotr ** (v - 1)) * fcycle

    # 2. 实现 energy
    def energy(route_np):
        cost = loc_energy + compute_energy2(route_np, uav, data["UAV_resource"].item())
        return cost

    # 3. 使用 sko.sa 优化
    init_route = np.arange(len(uav))
    sa = SA_TSP(func=energy,
            x0=init_route,
            T_max=T_max,
            T_min=T_min,
            L=steps,
            max_stay_counter=15)
    rout, best_energy = sa.run()
 
    return best_energy

def GAforPlot2(data, size,
    steps: int = 20,          # 进化代数（iterations） 
    pop_size: int = 10,       # 种群规模
    prob_mut: float = 0.05     # 变异概率
):
    
    x = torch.cat(
        (
            data["task_position"] - data["UAV_start_pos"],  # Δpos 2
            data["time_window"], # 2
            data["task_data"], # 1
            data["CPU_circles"], # 1
            data["IoT_resource"], # 1
        ),
        dim=-1,
    )

    x_np = x.squeeze(0).cpu().numpy()   
    sel_indices = np.random.choice(len(x_np), size=size, replace=False)
    loc_mask = np.isin(np.arange(len(x_np)), sel_indices)
    loc = x_np[loc_mask]
    uav = x_np[~loc_mask]
    
    loc_energy = 0.0

    for i in range(len(loc)):
       fcycle, iotr =  loc[i][5], loc[i][6]
       loc_energy = loc_energy + switched_capacitance * (iotr ** (v - 1)) * fcycle

    # 2. 实现 energy
    def energy(route_np):
        cost = loc_energy + compute_energy2(route_np, uav, data["UAV_resource"].item())
        return cost

    # --------------- 4. 运行 GA_TSP --------------------------
    ga = GA_TSP(
        func=energy,
        n_dim=len(uav),
        size_pop=pop_size,
        max_iter=steps,
        prob_mut=prob_mut,
        # 若希望关闭内部打印：verbose=False
    )
    route_scalar, best_energy_scalar = ga.run()

    best_energy = best_energy_scalar[0]
    return best_energy


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""