import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_box_with_mean(data_dict, pic='png.png',title=None, ylabel='Value', colors=None):
    if not isinstance(data_dict, dict):
        raise TypeError("no dict")
    if len(data_dict) < 1:
        raise ValueError("empty dict")
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',    
                      '#17becf', '#bcbd22', '#aec7e8', '#ffbb78',
                      '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
                      '#f7b6d2',  '#c7c7c7','#dbdb8d']
    colors = colors or default_colors[:len(data_dict)]
    # plt.figure(figsize=(10, 6), facecolor='white')
    # box = plt.boxplot(
    #     data_dict.values(),
    #     labels=data_dict.keys(),
    #     patch_artist=True,
    #     widths=0.7,
    #     showfliers=True,
    #     boxprops=dict(alpha=0.8)
    # )
    # for patch, color in zip(box['boxes'], colors):
    #     patch.set_facecolor(color)
    #     patch.set_edgecolor('black')
    # means = [np.mean(values) for values in data_dict.values()]
    # positions = range(1, len(data_dict) + 1)
    # plt.scatter(
    #     positions,
    #     means,
    #     marker='D',
    #     s=120,
    #     color='gold',
    #     edgecolor='black',
    #     zorder=3,
    #     label='Mean Value'
    # )

    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.ylabel(ylabel, fontsize=12)
    # plt.ylim(auto=True)
    # if title:
    #     plt.title(title, fontsize=14, pad=20)
    # plt.setp(box['whiskers'], color='gray', linestyle='--', linewidth=1.5)
    # plt.setp(box['medians'], color='black', linewidth=2)
    # plt.legend(loc='upper right', framealpha=0.9)
    # plt.tight_layout()
    # plt.savefig(pic, dpi=500, bbox_inches='tight')
    #plt.show()
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    box = ax.boxplot(
        data_dict.values(),
        labels=data_dict.keys(),
        patch_artist=True,
        widths=0.7,
        showfliers=True,
        boxprops=dict(alpha=0.8)
    )
    # 设置箱体颜色
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    # 绘制平均值点
    means = [np.mean(values) for values in data_dict.values()]
    positions = range(1, len(data_dict) + 1)
    ax.scatter(
        positions,
        means,
        marker='D',
        s=120,
        color='gold',
        edgecolor='black',
        zorder=3,
        label='Mean Value'
    )
    # 添加网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # 设置Y轴标签和字体大小
    ax.set_ylabel('Total energy', fontsize=30)  # 使用指定的y轴标签和字体大
    # 设置X轴刻度和标签
    # 箱线图默认位置是1,2,3,...，这里保持居中显示
    ax.set_xticks(positions)
    # 设置Y轴范围（自动调整）
    ax.set_ylim(auto=True)
    # 设置标题
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    # 美化箱线图元素
    plt.setp(box['whiskers'], color='gray', linestyle='--', linewidth=1.5)
    plt.setp(box['medians'], color='black', linewidth=2)
    # 设置图例
    ax.legend(fontsize=20 ,title_fontsize=11, loc='upper left', framealpha=0.9)
    # 设置刻度标签大小
    plt.tick_params(axis='both', which='major', labelsize=30)
        # 使用data_dict的键作为x轴标签
    ax.set_xticklabels(data_dict.keys(), rotation=0, fontsize=25)
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(pic, dpi=500, bbox_inches='tight')
    

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

    m = 16#dqn.get_select_size()
    dqn_mask10 = torch.zeros((batch_size, n), device=device)  
    dqn_sel10 = []
    dqn_mask01 = torch.zeros((batch_size, n), device=device)  
    dqn_sel01 = []
    # rain_mask10 = torch.zeros((batch_size, n), device=device)
    # rain_sel10 = []
    # rain_mask01 = torch.zeros((batch_size, n), device=device)
    # rain_sel01 = []

    for i in range(m):
        # 将 x 和 selected_mask 拼接为新的 state
        dqn_state10 = torch.cat((x, dqn_mask10.unsqueeze(-1)), dim=-1)  # 拼接 x 和 selected_mask
        dqn_state01 = torch.cat((x, dqn_mask01.unsqueeze(-1)), dim=-1)  
        # rain_state10 = torch.cat((x, rain_mask10.unsqueeze(-1)), dim=-1)  
        # rain_state01 = torch.cat((x, rain_mask01.unsqueeze(-1)), dim=-1)  
        # 计算当前状态下的 Q 值，并对已选任务进行屏蔽
        q_dqn10 = dqn(dqn_state10).masked_fill(dqn_mask10 == 1, -float('inf'))
        q_dqn01 = dqn(dqn_state01).masked_fill(dqn_mask01 == 1, -float('inf'))
        # q_rain10 = rain(rain_state10).masked_fill(rain_mask10 == 1, -float('inf'))
        # q_rain01 = rain(rain_state01).masked_fill(rain_mask01 == 1, -float('inf'))       
        # 由不同的策略选出 action
        a_dqn10 = q_dqn10.argmax(dim=-1)
        a_dqn01 = torch.multinomial(F.softmax(q_dqn01, dim=-1), 1).squeeze(-1)
        # a_rain10 = q_rain10.argmax(dim=-1)
        # a_rain01 = torch.multinomial(F.softmax(q_rain01, dim=-1), 1).squeeze(-1)

        dqn_sel10.append(a_dqn10)
        dqn_sel01.append(a_dqn01)
        # rain_sel10.append(a_rain10)
        # rain_sel01.append(a_rain01)
        # 更新状态，标记已选择任务
        dqn_mask10[torch.arange(batch_size), a_dqn10] = 1  # 更新已选择的任务
        dqn_mask01[torch.arange(batch_size), a_dqn01] = 1
        # rain_mask10[torch.arange(batch_size), a_rain10] = 1
        # rain_mask01[torch.arange(batch_size), a_rain01] = 1

    #任务对齐
    dqn_sel10 = torch.stack(dqn_sel10, dim=1) + 1 # (batch_size, m)
    dqn_sel01 = torch.stack(dqn_sel01, dim=1) + 1
    # rain_sel10 = torch.stack(rain_sel10, dim=1) + 1
    # rain_sel01 = torch.stack(rain_sel01, dim=1) + 1
    # 将选择的任务传入注意力模型，计算 cost
    model1.eval()
    # model2.eval()
    set_decode_type(model1, "sampling")
    cost1, _ = model1(data, dqn_sel10)
    set_decode_type(model1, "greedy")
    cost2, _ = model1(data, dqn_sel01)
    # set_decode_type(model2, "sampling")
    # cost3, _ = model2(data, rain_sel10)
    # set_decode_type(model2, "greedy")
    # cost4, _ = model2(data, rain_sel01)
    
    return [cost1, cost2] 

from Plot.GAS import SAforPlot
def sa_once(data, size):
    #return greedcom(SAforPlot, data, size)
    return SAforPlot(data, size)

from Plot.GAS import GAforPlot
def ga_once(data, size):
    #return greedcom(GAforPlot, data, size)
    return  GAforPlot(data, size)

from Plot.GAS import SAforPlot2
def sa_once2(data, size):
    #return greedcom(SAforPlot, data, size)
    return SAforPlot2(data, size)

from Plot.GAS import GAforPlot2
def ga_once2(data, size):
    #return greedcom(GAforPlot, data, size)
    return  GAforPlot2(data, size)

from Plot.load import load_problem, load_att, load_dqn, load_bow
from torch.utils.data import DataLoader

def runtest(opts):
    # Set the random seed
    torch.manual_seed(opts.seed)
    # Figure out what's the problem
    problem = load_problem(opts.problem)
    # Load model and agent
    # model1 = load_att(opts.load_path1)
    # SelDqn = load_dqn(opts.load_path1)
    model2 = load_att(opts.load_path2)
    Rainbow = load_bow(opts.load_path2)
    m = 16#dqn.get_select_size()
    # Make dataset
    # assert  model1.dependency == model2.dependency, "dependency dont match"
    # assert  SelDqn.get_select_size() == Rainbow.get_select_size(), "select_size dont match"

    dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.num_samples, dependency=model2.dependency)
    dataloader = DataLoader(dataset, batch_size=opts.num_samples)
    batch = next(iter(dataloader))

    spend = []
    Sac,Sac2,Gac,Gac2 = [],[],[],[]

    with torch.no_grad():
        data = batch
        t1 = time.perf_counter()
        modellist = run_once(data, model2, Rainbow)
        t2 = time.perf_counter()
        spend=[[(t2-t1)/2]]*2
    print("2 Model over")
    dataloader2 = DataLoader(dataset, batch_size=1)
    itter2 = iter(dataloader2)
    sat1 = time.perf_counter()
    for i in range(opts.num_samples):
        batch2 = next(itter2)
        SAcost = sa_once(batch2, m)
        Sac.append(SAcost)
    sat2 = time.perf_counter()
    spend = spend + [[sat2 - sat1]]
    print("SA over")
    dataloader3 = DataLoader(dataset, batch_size=1)
    itter3 = iter(dataloader3)
    gat1 = time.perf_counter()
    for i in range(opts.num_samples):
        batch3 = next(itter3)
        GAcost = ga_once(batch3,m)
        Gac.append(GAcost)
    gat2 = time.perf_counter()
    spend = spend + [[gat2 - gat1]]
    print("GA over")
    dataloader4 = DataLoader(dataset, batch_size=1)
    itter4 = iter(dataloader4)
    sat12 = time.perf_counter()
    for i in range(opts.num_samples):
        batch4 = next(itter4)
        SAcost2 = sa_once2(batch4, m)
        Sac2.append(SAcost2)
    sat22 = time.perf_counter()
    spend = spend + [[sat22 - sat12]]
    print("SA2 over")
    dataloader5 = DataLoader(dataset, batch_size=1)
    itter5 = iter(dataloader5)
    gat12 = time.perf_counter()
    for i in range(opts.num_samples):
        batch5 = next(itter5)
        GAcost2 = ga_once(batch5, m)
        Gac2.append(GAcost2)
    gat22 = time.perf_counter()
    spend = spend + [[gat22 - gat12]]
    print("GA2 over")
    # cost = [[],[],[],[],[],[],[],[],[],[]]
    # for i in range(10):

    #     SAcost = sa_once(batch, Rainbow.get_select_size())
    #     cost[4].append(SAcost)
 
    #     GAcost = ga_once(batch, Rainbow.get_select_size())
    #     cost[5].append(GAcost)
    return [modellist[0].tolist(),modellist[1].tolist(),Sac,Gac,Sac2,Gac2],spend

from Demo.Mec_demo import demo_options
from Plot.plot import drawpic

if __name__ == "__main__":
    opts = demo_options()

    costlist,timelist = runtest(opts)
       
    #np_dict = { k: v.detach().cpu().numpy().reshape(-1) for k, v in tensor_dict.items()}
    # np_dict = {
    #   "GreDqn+Sam" : np.array(costlist[0]),
    #   "SamDqn+Gre" : np.array(costlist[1]),
    #   "GreRain+Sam" : np.array(costlist[2]),
    #   "SamRain+Gre" : np.array(costlist[3]),
    #   "SAresult" : np.array(costlist[4]),
    #   "GAresult" : np.array(costlist[5]),
    #   #"SamDqnAtt" : np.array(costlist[6]),
    #   #"GreDqnAtt" : np.array(costlist[7]),
    #   #"SamRainAtt" : np.array(costlist[8]),
    #   #"GreRainAtt" : np.array(costlist[9])
    # }
    # np.savez(opts.result, **np_dict)     

    cost_dict = {
      "SDEARY" : np.array(costlist[0]),
      "GDEARY" : np.array(costlist[1]),
      "GA" : np.array(costlist[3]),
      "SA" : np.array(costlist[2]),
      "RGA" : np.array(costlist[5]),
      "RSA": np.array(costlist[4]),
    }
    #np.savez(opts.cost, **cost_dict)
    np.savez("./za/q.npz", **cost_dict)
    #plot_box_with_mean(cost_dict, opts.energy, ylabel='Total energy')
    cost_dict = dict(np.load('./za/q.npz', allow_pickle=True))
    plot_box_with_mean(cost_dict, "./za/r.png", ylabel='Total energy')
    print("The picture is accessable")
    # time_dict = {
    #   "GreRain+Sam" : np.array(timelist[0]),
    #   "SamRain+Gre" : np.array(timelist[1]),
    #   "SAresult" : np.array(timelist[2]),
    #   "GAresult" : np.array(timelist[3]),
    #   "SA2result" : np.array(timelist[4]),
    #   "GA2result" : np.array(timelist[5]),
    # }
    # np.savez(opts.spend, **time_dict)
    # plot_box_with_mean(time_dict,opts.time, ylabel='Objetiva Time')



# nohup python -u zzztest.py --problem 'mec' --graph_size 30   
# --load_path1 
# --load_path2 
#>./     2>&1 &
# python -u zzztest.py --problem 'mec' --graph_size 40 --load_path2 './zzzz/mec40/4/epoch-49.pt' --num_samples 100