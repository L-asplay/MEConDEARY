import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_box_with_mean(data_dict, pic='./za/b.png',title=None,):
    if not isinstance(data_dict, dict):
        raise TypeError("no dict")
    if len(data_dict) < 1:
        raise ValueError("empty dict")
    
    markers = ['o', 's', '^', 'D', 'v', '*']
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    labels = list(data_dict.keys())
    x_indices = range(len(list(data_dict[labels[0]].keys())))
    plt.figure(figsize=(10, 6), facecolor='white')
    for i in range(6):
        plt.plot(list(data_dict[labels[i]].keys()), 
               list(data_dict[labels[i]].values()), 
               label=labels[i], marker=markers[i], 
               linestyle='-', color=colors[i], 
               linewidth=3, markersize=15)
    # for i in range(6):
    #     plt.plot(x_indices, 
    #            list(data_dict[labels[i]].values()), 
    #            label=labels[i], marker=markers[i], 
    #            linestyle='-', color=colors[i], 
    #            linewidth=3, markersize=15)
    #     plt.xticks(x_indices, list(map(str, list(data_dict[labels[i]].keys())))) 
    # ==================== 图表美化 ====================
    plt.rcParams['ytick.labelsize'] = 16
    plt.xlabel("Number of tasks", fontsize=30)  # 横轴标签
    plt.ylabel("Total energy", fontsize=30)  # 纵轴标签
    #plt.xlim(0, 500)  # 横轴范围
    plt.ylim(list(data_dict[labels[1]].values())[0], list(data_dict[labels[5]].values())[3])  # 纵轴范围
    # 设置图例（关键参数）
    plt.legend(
      bbox_to_anchor=(0.01, 0.98),  # 锚点在图内右侧边缘（x=0.9，y=1）
      loc='upper left',         # 图例左上角对齐锚点，放入图内
      borderaxespad=0.05,        # 图例与坐标轴的间距（可调整）
      fontsize=20,              # 图例文本字体大小
      title_fontsize=11,        # 图例标题字体大小
      ncol=2,                   # 两列显示，自动换行为两行
      frameon=True,             # 显示图例边框（可选）
      facecolor='white',        # 图例背景色（可选）
      edgecolor='gray'          # 图例边框颜色（可选）
    )
    plt.tick_params(axis='both', which='major', labelsize=23)
    # 调整子图右侧边距（避免图例被截断）
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # 右侧留出 20% 空间
    plt.savefig(pic, dpi=150, bbox_inches='tight')
    #plt.show()

import time
from datetime import datetime
def ts_print(msg: str):
    now = datetime.now().strftime('%H:%M:%S')
    print(f"[{now}] {msg}")

from nets.attention_model import set_decode_type

def run_once(data, model1, dqn):
    x = torch.cat((data["task_position"]-data["UAV_start_pos"], data["time_window"]/(10), data["IoT_resource"], data["task_data"], data["CPU_circles"]), dim=-1)
    batch_size, n, _ = x.size()
    device = x.device
    m = dqn.get_select_size()
    dqn_mask10 = torch.zeros((batch_size, n), device=device)  
    dqn_sel10 = []
    dqn_mask01 = torch.zeros((batch_size, n), device=device)  
    dqn_sel01 = []
    for i in range(m):
        dqn_state10 = torch.cat((x, dqn_mask10.unsqueeze(-1)), dim=-1) 
        dqn_state01 = torch.cat((x, dqn_mask01.unsqueeze(-1)), dim=-1)  
        q_dqn10 = dqn(dqn_state10).masked_fill(dqn_mask10 == 1, -float('inf'))
        q_dqn01 = dqn(dqn_state01).masked_fill(dqn_mask01 == 1, -float('inf'))
        a_dqn10 = q_dqn10.argmax(dim=-1)
        a_dqn01 = torch.multinomial(F.softmax(q_dqn01, dim=-1), 1).squeeze(-1)
        dqn_sel10.append(a_dqn10)
        dqn_sel01.append(a_dqn01)
        dqn_mask10[torch.arange(batch_size), a_dqn10] = 1  
        dqn_mask01[torch.arange(batch_size), a_dqn01] = 1
    dqn_sel10 = torch.stack(dqn_sel10, dim=1) + 1 
    dqn_sel01 = torch.stack(dqn_sel01, dim=1) + 1
    model1.eval()
    set_decode_type(model1, "sampling")
    cost1, _ = model1(data, dqn_sel10)
    set_decode_type(model1, "greedy")
    cost2, _ = model1(data, dqn_sel01)
    return torch.mean(cost1).item(), torch.mean(cost2).item()

from Plot.GAS import SAforPlot
def sa_once(data, size):
    return SAforPlot(data, size)

from Plot.GAS import GAforPlot
def ga_once(data, size):
    return  GAforPlot(data, size)

from Plot.GAS import SAforPlot2
def sa_once2(data, size):
    return SAforPlot2(data, size)

from Plot.GAS import GAforPlot2
def ga_once2(data, size):
    return  GAforPlot2(data, size)

from Plot.load import load_problem, load_att, load_bow
from torch.utils.data import DataLoader

def graph_value(model2,Rainbow,problem,graph_size,opts):
     
    dataset = problem.make_dataset(size=graph_size, num_samples=opts.num_samples, dependency=model2.dependency)

    dataloader = DataLoader(dataset, batch_size=opts.num_samples)
    batch = next(iter(dataloader))
    with torch.no_grad():
        data = batch
        da1,da2 = run_once(data, model2, Rainbow)
    print("2 Model over")

    Sac,Sac2,Gac,Gac2 = [],[],[],[]

    dataloader2 = DataLoader(dataset, batch_size=1)
    itter2 = iter(dataloader2)
    for i in range(opts.num_samples):
        batch2 = next(itter2)
        SAcost = sa_once(batch2, Rainbow.get_select_size())
        Sac.append(SAcost)
    print("SA over")

    dataloader3 = DataLoader(dataset, batch_size=1)
    itter3 = iter(dataloader3)
    for i in range(opts.num_samples):
        batch3 = next(itter3)
        GAcost = ga_once(batch3, Rainbow.get_select_size())
        Gac.append(GAcost)
    print("GA over")

    dataloader4 = DataLoader(dataset, batch_size=1)
    itter4 = iter(dataloader4)
    for i in range(opts.num_samples):
        batch4 = next(itter4)
        SAcost2 = sa_once2(batch4, Rainbow.get_select_size())
        Sac2.append(SAcost2)
    print("SA2 over")

    dataloader5 = DataLoader(dataset, batch_size=1)
    itter5 = iter(dataloader5)
    for i in range(opts.num_samples):
        batch5 = next(itter5)
        GAcost2 = ga_once(batch5, Rainbow.get_select_size())
        Gac2.append(GAcost2)
    print("GA2 over")

    return [da1,da2,np.mean(Gac),np.mean(Sac),np.mean(Gac2),np.mean(Sac2)]

def runtest(opts):
    # Set the random seed
    torch.manual_seed(opts.seed)
    problem = load_problem(opts.problem)

    size=[30,50,100,150]
    path=['./zzz/mec_30/epoch-49.pt',
          './zzzz/mec50/epoch-49.pt',
          './zzz/mec_100/epoch-49.pt',
          './zzzz/mec150/epoch-49.pt'
    ]

    cost = []
    for i in range(4):
        model2 = load_att(path[i])
        Rainbow = load_bow(path[i])
        cost.append(graph_value(model2,Rainbow,problem,size[i],opts))
    return cost


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="The sets in demo")
    # parser.add_argument('--problem', default='mec', help="The problem to solve, default 'mec'")
    # parser.add_argument('--num_samples', type=int, default=100, help="The size of the dataset batch")
    # parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')

    # parser.add_argument('--cost', default='./za/a.npz')

    # opts = parser.parse_args()
    # opts.use_cuda = torch.cuda.is_available()

    # costlist = runtest(opts)

    # cost_dict_i = [[],[],[],[],[],[]]
    # for i in range(6):
    #   cost_dict_i[i] = {
    #     "30" : np.array(costlist[0][i]),
    #     "50" : np.array(costlist[1][i]),
    #     "100" : np.array(costlist[2][i]),
    #     "150" : np.array(costlist[3][i]),
    #   }

    # cost_dict = {
    #   "SDEARY" : cost_dict_i[0],
    #   "GDEARY" : cost_dict_i[1],
    #   "GA" : cost_dict_i[2],
    #   "SA" : cost_dict_i[3],
    #   "RGA" : cost_dict_i[4],
    #   "RSA" : cost_dict_i[5],
    # }
    # print(cost_dict)
    # np.savez(opts.cost, **cost_dict)
    data = dict(np.load("./za/a.npz", allow_pickle=True))
    a = {}
    for key, array_val in data.items():
       inner_dict = array_val.item()  
       a[key] = {}
       for inner_key, scalar_array in inner_dict.items():
          a[key][inner_key] = scalar_array.item() 
    plot_box_with_mean(a)
    # plot_box_with_mean(cost_dict)
    print("The picture is accessable")

# nohup python -u zdraw.py  >./za/c.log 2>&1 &


