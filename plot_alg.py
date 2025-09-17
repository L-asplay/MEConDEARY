import os
import torch
import numpy as np
import torch.nn.functional as F
from datetime import datetime
def ts_print(msg: str):
    now = datetime.now().strftime('%H:%M:%S')
    print(f"[{now}] {msg}")


from nets.attention_model import set_decode_type

def run_once(data, model1, dqn, model2, rain):

    x = torch.cat((data["task_position"]-data["UAV_start_pos"], data["time_window"]/(10), data["IoT_resource"], data["task_data"], data["CPU_circles"]), dim=-1)
    batch_size, n, _ = x.size()
    device = x.device

    m = dqn.get_select_size()
    dqn_mask10 = torch.zeros((batch_size, n), device=device)  
    dqn_sel10 = []
    dqn_mask01 = torch.zeros((batch_size, n), device=device)  
    dqn_sel01 = []
    rain_mask10 = torch.zeros((batch_size, n), device=device)
    rain_sel10 = []
    rain_mask01 = torch.zeros((batch_size, n), device=device)
    rain_sel01 = []

    for i in range(m):
        # 将 x 和 selected_mask 拼接为新的 state
        dqn_state10 = torch.cat((x, dqn_mask10.unsqueeze(-1)), dim=-1)  # 拼接 x 和 selected_mask
        dqn_state01 = torch.cat((x, dqn_mask01.unsqueeze(-1)), dim=-1)  
        rain_state10 = torch.cat((x, rain_mask10.unsqueeze(-1)), dim=-1)  
        rain_state01 = torch.cat((x, rain_mask01.unsqueeze(-1)), dim=-1)  
        # 计算当前状态下的 Q 值，并对已选任务进行屏蔽
        q_dqn10 = dqn(dqn_state10).masked_fill(dqn_mask10 == 1, -float('inf'))
        q_dqn01 = dqn(dqn_state01).masked_fill(dqn_mask01 == 1, -float('inf'))
        q_rain10 = rain(rain_state10).masked_fill(rain_mask10 == 1, -float('inf'))
        q_rain01 = rain(rain_state01).masked_fill(rain_mask01 == 1, -float('inf'))       
        # 由不同的策略选出 action
        a_dqn10 = q_dqn10.argmax(dim=-1)
        a_dqn01 = torch.multinomial(F.softmax(q_dqn01, dim=-1), 1).squeeze(-1)
        a_rain10 = q_rain10.argmax(dim=-1)
        a_rain01 = torch.multinomial(F.softmax(q_rain01, dim=-1), 1).squeeze(-1)

        dqn_sel10.append(a_dqn10)
        dqn_sel01.append(a_dqn01)
        rain_sel10.append(a_rain10)
        rain_sel01.append(a_rain01)
        # 更新状态，标记已选择任务
        dqn_mask10[torch.arange(batch_size), a_dqn10] = 1  # 更新已选择的任务
        dqn_mask01[torch.arange(batch_size), a_dqn01] = 1
        rain_mask10[torch.arange(batch_size), a_rain10] = 1
        rain_mask01[torch.arange(batch_size), a_rain01] = 1

    #任务对齐
    dqn_sel10 = torch.stack(dqn_sel10, dim=1) + 1 # (batch_size, m)
    dqn_sel01 = torch.stack(dqn_sel01, dim=1) + 1
    rain_sel10 = torch.stack(rain_sel10, dim=1) + 1
    rain_sel01 = torch.stack(rain_sel01, dim=1) + 1
    # 将选择的任务传入注意力模型，计算 cost
    model1.eval()
    model2.eval()
    set_decode_type(model1, "sampling")
    cost1, _ = model1(data, dqn_sel10)
    set_decode_type(model1, "greedy")
    cost2, _ = model1(data, dqn_sel01)
    set_decode_type(model2, "sampling")
    cost3, _ = model2(data, rain_sel10)
    set_decode_type(model2, "greedy")
    cost4, _ = model2(data, rain_sel01)

    return [cost1.item(), cost2.item(), cost3.item(), cost4.item()] 

def greedcom(algorithm, data, M):
    loc = data["task_position"]
    batch_size, N, _ = loc.size()
    device = loc.device

    selected = torch.zeros((batch_size, N), dtype=torch.bool, device=device)  # 初始本地集合 A = 空集
   
    for step in range(M):
        energy_list = []
        for i in range(N):
            mask = selected.clone() # 已经选过的任务不能再试探
            if mask[:, i] == True : 
                energy = torch.tensor(float('inf'), dtype=torch.float32, device=device) 
            else :
                mask[:, i] = True  # 试探将任务 i 加入本地执行集合 A
                energy = algorithm(mask, data)  # 计算总能耗
            energy_list.append(energy.unsqueeze(0))
        energy_matrix = torch.cat(energy_list, dim=0)  # shape: (N)
        energy_matrix[selected[0]] = float('inf')  # 过滤掉已选择的任务
        best_choice = torch.argmin(energy_matrix, dim=0)  # 在每个 batch 中选出使得能耗最小的任务索引
        selected[torch.arange(batch_size), best_choice] = True  # 添加到本地集合 A 中

    final_cost = algorithm(selected, data)
    return final_cost.item()


from Plot.GAS import SAforPlot
def sa_once(data, size):
    #return greedcom(SAforPlot, data, size)
    return SAforPlot(data, size)

from Plot.GAS import GAforPlot
def ga_once(data, size):
    #return greedcom(GAforPlot, data, size)
    return  GAforPlot(data, size)


def att(model):
    return lambda mask, x : forw(model, mask, x)
def forw(model, local_mask, data):
    loc_indices = torch.nonzero(local_mask, as_tuple=False)
    selections = loc_indices[:,1:] + 1
    selections = selections.view(local_mask.size(0), -1)
    cost, _ = model(data, selections)
    return cost.squeeze(0)

def att_once(data, model1, model2, size):
    set_decode_type(model1, "sampling")
    alg = att(model1)
    cost1 = greedcom(alg, data, size)
    set_decode_type(model1, "greedy")
    alg = att(model1)
    cost2 = greedcom(alg, data, size)
    set_decode_type(model2, "sampling")
    alg = att(model2)
    cost3 = greedcom(alg, data, size)
    set_decode_type(model2, "greedy")
    alg = att(model2)
    cost4 = greedcom(alg, data, size)
    return [cost1, cost2, cost3, cost4] 

from Plot.load import load_problem, load_att, load_dqn, load_bow
from torch.utils.data import DataLoader

def runtest(opts):
    # Set the random seed
    torch.manual_seed(opts.seed)
    # Figure out what's the problem
    problem = load_problem(opts.problem)
    # Load model and agent
    model1 = load_att(opts.load_path1)
    SelDqn = load_dqn(opts.load_path1)
    model2 = load_att(opts.load_path2)
    Rainbow = load_bow(opts.load_path2)
    # Make dataset
    assert  model1.dependency == model2.dependency, "dependency dont match"
    assert  SelDqn.get_select_size() == Rainbow.get_select_size(), "select_size dont match"

    dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.num_samples, dependency=model1.dependency)
    dataloader = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader))
    
    # Take the cost and pi
    cost = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(10):
        ts_print("Model test {}".format(i))
        with torch.no_grad():
           modellist = run_once(batch, model1, SelDqn, model2, Rainbow)
        for j in range(4):
            cost[j].append(modellist[j])
        ts_print("Model end {}".format(i))
        ts_print("SA test {}".format(i))
        SAcost = sa_once(batch, SelDqn.get_select_size())
        cost[4].append(SAcost)
        ts_print("SA end {}".format(i))
        ts_print("GA test {}".format(i))
        GAcost = ga_once(batch, SelDqn.get_select_size())
        cost[5].append(GAcost)
        ts_print("GA end {}".format(i))
        """
        ts_print("4 greed test {}".format(i))
        with torch.no_grad():
            attlist = att_once(batch, model1, model2, SelDqn.get_select_size())
        ts_print("4 greed end {}".format(i))
        for j in range(6,10):
            cost[j].append(attlist[j-6])
        """
    return cost

from Demo.Mec_demo import demo_options
from Plot.plot import drawpic

if __name__ == "__main__":
    opts = demo_options()

    costlist = runtest(opts)
       
    #np_dict = { k: v.detach().cpu().numpy().reshape(-1) for k, v in tensor_dict.items()}
    np_dict = {
      "GreDqn+Sam" : np.array(costlist[0]),
      "SamDqn+Gre" : np.array(costlist[1]),
      "GreRain+Sam" : np.array(costlist[2]),
      "SamRain+Gre" : np.array(costlist[3]),
      "SAresult" : np.array(costlist[4]),
      "GAresult" : np.array(costlist[5]),
      #"SamDqnAtt" : np.array(costlist[6]),
      #"GreDqnAtt" : np.array(costlist[7]),
      #"SamRainAtt" : np.array(costlist[8]),
      #"GreRainAtt" : np.array(costlist[9])
    }
    np.savez(opts.result, **np_dict)     
    drawpic(opts.result,opts.pic)

# nohup python -u plot_alg.py --problem 'mec' --graph_size 30   
# --load_path1 
# --load_path2 
#>./     2>&1 &
