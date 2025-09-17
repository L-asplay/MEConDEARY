import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

data5 = dict(np.load("./za/o.npz", allow_pickle=True))
data4 = dict(np.load("./za/q.npz", allow_pickle=True))
# 示例数据生成（如果已有数据可以替换这部分）
# np.random.seed(42)  # 设置随机种子，确保结果可复现
# keys = ['A', 'B', 'C', 'D', 'E', 'F']
# a = {key: np.random.normal(10, 2, 100) for key in keys}  # 字典a，每个值是100个数字
# b = {key: np.random.normal(15, 3, 100) for key in keys}  # 字典b，每个值是100个数字
keys = list(data5.keys())
a = data5
b = data4

# # 指定颜色方案
# default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#                   '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',    
#                   '#17becf', '#bcbd22', '#aec7e8', '#ffbb78',
#                   '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
#                   '#f7b6d2', '#c7c7c7', '#dbdb8d']
# default_colors = plt.cm.tab10(np.linspace(0, 1, len(keys)*2))
# # 为两个字典分配不同的颜色
# a_colors = default_colors[:len(keys)]
# b_colors = default_colors[len(keys):len(keys)*2]

a_colors = plt.cm.tab10(np.linspace(0, 1, 6))
hsv1 = rgb_to_hsv(a_colors[:, :3])
hsv2 = hsv1.copy()
hsv2[:, 2] = hsv2[:, 2] * 0.7  
b_colors = hsv_to_rgb(hsv2)

# 设置图形
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')

# 箱体宽度和位置设置
box_width = 0.4
x = np.arange(len(keys))

# 准备数据
a_data = [a[key] for key in keys]
b_data = [b[key] for key in keys]

# 计算平均值
a_means = [np.mean(data) for data in a_data]
b_means = [np.mean(data) for data in b_data]

# 绘制箱线图
box_a = ax.boxplot(a_data, positions=x - box_width/2, widths=box_width,
                  patch_artist=True, showfliers=True,
                  boxprops=dict(alpha=0.8))

box_b = ax.boxplot(b_data, positions=x + box_width/2, widths=box_width,
                  patch_artist=True, showfliers=True,
                  boxprops=dict(alpha=0.8))

# 设置箱体颜色
for patch, color in zip(box_a['boxes'], a_colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

for patch, color in zip(box_b['boxes'], b_colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

# 设置箱线图其他元素样式
for whisker in box_a['whiskers'] + box_b['whiskers']:
    whisker.set_color('gray')
    whisker.set_linestyle('--')
    whisker.set_linewidth(1.5)

for median in box_a['medians'] + box_b['medians']:
    median.set_color('black')
    median.set_linewidth(2)

for cap in box_a['caps'] + box_b['caps']:
    cap.set_color('gray')
    cap.set_linewidth(1.5)

# 添加平均值标记（菱形）
a_mean_markers = ax.scatter(x - box_width/2, a_means, marker='D',
                           s=120, color='gold', edgecolor='black',
                           zorder=3, label='Mean Value')

b_mean_markers = ax.scatter(x + box_width/2, b_means, marker='D',
                           s=120, color='gold', edgecolor='black',
                           zorder=3)

# 添加平均值数值标签
# for i in range(len(keys)):
#     ax.text(x[i] - box_width/2, a_means[i] + 0.5, f'{a_means[i]:.1f}',
#             ha='center', va='bottom', fontsize=9, fontweight='bold')
#     ax.text(x[i] + box_width/2, b_means[i] + 0.5, f'{b_means[i]:.1f}',
#             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 设置坐标轴和标题
plt.rcParams['ytick.labelsize'] = 16
# ax.set_xlabel('Algorithm', fontsize=30)
ax.set_ylabel('Total energy', fontsize=30)
ax.set_xticks(x)
ax.set_xticklabels(keys, fontsize=30)
# 设置网格
ax.grid(axis='y', linestyle='--', alpha=0.7)

# plt.plot([], [], ' ', label='choose 4/16')
plt.legend(
      bbox_to_anchor=(0.01, 1),  # 锚点在图内右侧边缘（x=0.9，y=1）
      loc='upper left',         # 图例左上角对齐锚点，放入图内
      borderaxespad=0.01,        # 图例与坐标轴的间距（可调整）
      fontsize=20,              # 图例文本字体大小
      title_fontsize=10,        # 图例标题字体大小
      ncol=1,                   # 两列显示，自动换行为两行
      frameon=True,             # 显示图例边框（可选）
      facecolor='white',        # 图例背景色（可选）
      edgecolor='gray'          # 图例边框颜色（可选）
)
# 美化图表
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=23)
plt.subplots_adjust(right=0.8)  # 右侧留出 20% 空间
# 调整布局并显示
plt.tight_layout()

# 显示图形
plt.savefig('./za/s.png', dpi=500, bbox_inches='tight')
print("The picture is accessable")


# def plot_box_with_mean(data_dict, pic='./za/k.png',title=None,):
#     default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',    
#                       '#17becf', '#bcbd22', '#aec7e8', '#ffbb78',
#                       '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
#                       '#f7b6d2',  '#c7c7c7','#dbdb8d']
#     colors = colors or default_colors[:len(data_dict)]
#     fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
#     box = ax.boxplot(
#         data_dict.values(),
#         labels=data_dict.keys(),
#         patch_artist=True,
#         widths=0.7,
#         showfliers=True,
#         boxprops=dict(alpha=0.8)
#     )
#     # 设置箱体颜色
#     for patch, color in zip(box['boxes'], colors):
#         patch.set_facecolor(color)
#         patch.set_edgecolor('black')
#     # 绘制平均值点
#     means = [np.mean(values) for values in data_dict.values()]
#     positions = range(1, len(data_dict) + 1)
#     ax.scatter(
#         positions,
#         means,
#         marker='D',
#         s=120,
#         color='gold',
#         edgecolor='black',
#         zorder=3,
#         label='Mean Value'
#     )
#     # 添加网格线
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#     # 设置Y轴标签和字体大小
#     ax.set_ylabel('Total energy', fontsize=30)  # 使用指定的y轴标签和字体大
#     # 设置X轴刻度和标签
#     # 箱线图默认位置是1,2,3,...，这里保持居中显示
#     ax.set_xticks(positions)
#     # 设置Y轴范围（自动调整）
#     ax.set_ylim(auto=True)
#     # 设置标题
#     if title:
#         ax.set_title(title, fontsize=14, pad=20)
#     # 美化箱线图元素
#     plt.setp(box['whiskers'], color='gray', linestyle='--', linewidth=1.5)
#     plt.setp(box['medians'], color='black', linewidth=2)
#     # 设置图例
#     ax.legend(fontsize=20 ,title_fontsize=11, loc='upper left', framealpha=0.9)
#     # 设置刻度标签大小
#     plt.tick_params(axis='both', which='major', labelsize=30)
#         # 使用data_dict的键作为x轴标签
#     ax.set_xticklabels(data_dict.keys(), rotation=0, fontsize=25)
#     # 调整布局并保存
#     plt.tight_layout()
#     plt.savefig(pic, dpi=500, bbox_inches='tight')
#     #plt.show()


# if __name__ == "__main__":

#     data5 = dict(np.load("./za/g.npz", allow_pickle=True))
#     data4 = dict(np.load("./za/i.npz", allow_pickle=True))
#     plot_box_with_mean(data5, data4, './za/k.png')
#     print("The picture is accessable")



