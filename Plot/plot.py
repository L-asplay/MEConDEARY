import matplotlib.pyplot as plt
import numpy as np

def plot_box_with_mean(data_dict, pic,title=None, ylabel='Value', colors=None):
    """
    绘制带平均值标记的箱线图

    参数：
    data_dict : dict - 字典格式数据，键为标签，值为数值列表/数组
    title     : str  - 图表标题（可选）
    ylabel    : str  - Y轴标签（默认'Value'）
    colors    : list - 自定义颜色列表（可选）
    """
    # 数据校验
    if not isinstance(data_dict, dict):
        raise TypeError("输入数据必须是字典类型")
    if len(data_dict) < 1:
        raise ValueError("数据字典不能为空")

    # 设置默认颜色
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',    
                      '#17becf', '#bcbd22', '#aec7e8', '#ffbb78',
                      '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
                      '#f7b6d2',  '#c7c7c7','#dbdb8d']
    colors = colors or default_colors[:len(data_dict)]

    # 创建画布
    plt.figure(figsize=(10, 6), facecolor='white')

    # 绘制箱线图
    box = plt.boxplot(
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

    # 计算并绘制平均值
    means = [np.mean(values) for values in data_dict.values()]
    positions = range(1, len(data_dict) + 1)

    plt.scatter(
        positions,
        means,
        marker='D',  # 使用钻石形状
        s=120,
        color='gold',
        edgecolor='black',
        zorder=3,
        label='Mean Value'
    )

    # 样式配置
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(auto=True)
    if title:
        plt.title(title, fontsize=14, pad=20)

    # 设置须线和中位数样式
    plt.setp(box['whiskers'], color='gray', linestyle='--', linewidth=1.5)
    plt.setp(box['medians'], color='black', linewidth=2)

    # 添加图例
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(pic, dpi=500, bbox_inches='tight')
    #plt.show()

def drawpic(name, pic):
    data_dict = dict(np.load(name, allow_pickle=True))
     
    plot_box_with_mean(data_dict, pic, ylabel='Objetiva Value')

if __name__ == "__main__":
    data_dict = dict(np.load('result.npz', allow_pickle=True))
    plot_box_with_mean(data_dict, ylabel='Objetiva Value')
    