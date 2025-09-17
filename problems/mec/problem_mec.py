from torch.utils.data import Dataset
import torch,math
import os
import pickle
from problems.mec.state_mec import StateMEC
from utils.beam_search import beam_search
import numpy as np

class MEC(object):

    NAME = 'mec'
    """
    Rc = 40.0  # coverage / bias
    UAV_p = 50
    # UAV fly
    height = 10
    g = 9.8  # gravity 
    speed = 15
    quantity_uav = 2
    Cd = 0.3
    A = 0.1
    air_density = 1.225
    P_fly = air_density * A * Cd * pow(speed, 3) / 2 + quantity_uav * g * speed
    P_stay = pow(speed, 3)
    # Iot device energy compute
    switched_capacitance = 1e-6
    v = 4
    # transmit
    B = 1e6
    g0 = 20
    G0 = 5
    upload_P = 3
    noise_P = -90
    hm = 0
    d_2 = pow(Rc, 2) + pow(height, 2)
    upload_speed = B * math.log2(1 + g0 * G0 * upload_P / pow(noise_P, 2) / (pow(hm, 2) + d_2))
    """
    @staticmethod
    def get_costs(dataset, pi, selections, ccost=None, tw=None):
        task_data = dataset['task_data'] 
        dependencys = dataset['dependencys']

        batch_size, gena_size, _ = task_data.size()
        _, graph_size = pi.size()
        assert batch_size==pi.size(0), "batch not match "

        # 检查任务选择 pi 与 selections
        combined = torch.cat([pi, selections], dim=1) 
        # 检查是否构成全集 {1..N}
        sorted_combined, _ = combined.sort(dim=1)
        full_set = torch.arange(1, gena_size + 1, device=pi.device).expand(batch_size, gena_size)
        match_full = (sorted_combined == full_set).all(dim=1)
        # 构造 one-hot，检查是否有重复（每个元素在每行中只能出现一次）
        combined_adj = combined - 1  # 值域应为 [0, gena_size-1]
        one_hot = torch.nn.functional.one_hot(combined_adj, num_classes=gena_size)  # [B, N, gena_size]
        count = one_hot.sum(dim=1)  # [B, gena_size]
        has_duplicates = (count > 1).any(dim=1)  # [B]
        # 满足全集 + 无重复
        ok = match_full & (~has_duplicates)
        if not ok.all():
           bad_batches = (~ok).nonzero(as_tuple=False).squeeze(1).tolist()
           raise AssertionError( f"Invalid batches (pi ∩ selections ≠ ∅ or not full set): {bad_batches}")
        
        # 检查先后顺序
        index = dependencys.unsqueeze(-1).expand(-1, -1, 2) 
        dep_tw = torch.gather(tw, dim=1, index=index)
        assert (dep_tw[:, :-1, 0] <= dep_tw[:, 1:, 0]).all(), "dependency error"
        
        # 检查是否违背时间窗
        assert (tw[:, :, 0] <= tw[:, :, 1]).all(), "time_windows error"
        
        return ccost.squeeze(dim=1),None# 总能耗

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MECDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMEC.initialize(*args, **kwargs)

    @staticmethod
    def beam_search():
        # TODO: complete this function
        pass

# def make_instance(args):
#     task_data, UAV_start_pos, task_position, CPU_circles, IoT_resource, UAV_resource, time_window, dependencys, *args = args
#     return {
#         'task_data' : torch.tensor(task_data, dtype=torch.float),
#         'UAV_start_pos' : torch.tensor(UAV_start_pos, dtype=torch.float),
#         'task_position' : torch.tensor(task_position, dtype=torch.float),
#         'CPU_circles' : torch.tensor(CPU_circles, dtype=torch.float),
#         'IoT_resource' : torch.tensor(IoT_resource, dtype=torch.float),
#         'UAV_resource': torch.tensor(UAV_resource, dtype=torch.float),
#         'time_window' : torch.tensor(time_window, dtype=torch.float),
#         'dependencys' : torch.tensor(dependencys),
#     }

# class MECDataset(Dataset):

#     def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, dependency=[],distribution=None):
#         super(MECDataset, self).__init__()
#         self.dependency=dependency
#         if filename is not None:
#             assert os.path.splitext(filename)[1] == '.pkl'
#             with open(filename, 'rb') as f:
#                 data = pickle.load(f)
#         else:
#             #task_data = np.random.uniform(size=(dataset_size, uav_size, 1), low=0, high=1000)
#             #UAV_start_pos = np.random.randint(size=(dataset_size, 1, 2), low = 0, high = 500)
#             #task_position = np.random.uniform(size=(dataset_size, uav_size, 2), low=0, high=500)
#             #CPU_circles = np.random.randint(size=(dataset_size, uav_size, 1), low=0, high=1000)
#             #IoT_resource = np.random.randint(size=(dataset_size, uav_size, 1), low=100, high=200)
#             #UAV_resource = np.max(CPU_circles, axis=1, keepdims=True) // 4
    
#             task_position = np.random.uniform(size=(num_samples, size, 2))
#             UAV_start_pos = np.random.uniform(size=(num_samples, 1, 2))
#             task_data = np.random.uniform(size=(num_samples, size, 1), low=0, high=1)
#             CPU_circles = np.random.uniform(size=(num_samples, size, 1), low=0, high=1)
#             IoT_resource = np.random.uniform(size=(num_samples, size, 1), low=2, high=5)/10
#             UAV_resource = np.max(CPU_circles, axis=1, keepdims=True)/2

#             dependency = [ i -1 for  i in dependency]

#             time_window = np.random.uniform(size=(num_samples, size, 2), low=0, high=10)
#             time_window = np.sort(time_window, axis=2)
#             dep_window = np.take(time_window, indices=dependency, axis=1)
#             dep_window = np.sort(dep_window.reshape(num_samples, -1), axis=1).reshape(num_samples, len(dependency), 2)
#             np.put_along_axis(time_window, np.array(dependency)[None, :, None].astype(int), dep_window, axis=1)
#             time_window[:, :, 1] = 1000

#             dependency = [ i + 1 for  i in dependency]

#             dependencys = [dependency] * num_samples


#             data = list(zip(
#                         task_data.tolist(),
#                         UAV_start_pos.tolist(),
#                         task_position.tolist(),
#                         CPU_circles.tolist(),
#                         IoT_resource.tolist(),
#                         UAV_resource.tolist(),
#                         time_window.tolist(),
#                         dependencys
#                         ))


#         self.data = [make_instance(args) for args in data[offset:offset + num_samples]]
#         self.size = len(self.data)

#     def __len__(self):
#         return self.size

#     def __getitem__(self, idx):
#         return self.data[idx]
def make_instance(task_data, UAV_start_pos, task_position, CPU_circles, 
                 IoT_resource, UAV_resource, time_window, dependencys):
    # 使用torch.from_numpy共享内存而非复制，节省内存
    return {
        'task_data': torch.from_numpy(task_data).float(),
        'UAV_start_pos': torch.from_numpy(UAV_start_pos).float(),
        'task_position': torch.from_numpy(task_position).float(),
        'CPU_circles': torch.from_numpy(CPU_circles).float(),
        'IoT_resource': torch.from_numpy(IoT_resource).float(),
        'UAV_resource': torch.from_numpy(UAV_resource).float(),
        'time_window': torch.from_numpy(time_window).float(),
        'dependencys': torch.tensor(dependencys),  # 依赖关系通常较小，保持tensor
    }

class MECDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=1000000, 
                 offset=0, dependency=[], distribution=None):
        super(MECDataset, self).__init__()
        self.dependency = dependency
        self.num_samples = num_samples
        self.size = size

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            data_slice = data[offset:offset + num_samples]
            
            # 转换为numpy数组而非列表存储，减少内存占用
            self.task_data = np.array([d[0] for d in data_slice])
            self.UAV_start_pos = np.array([d[1] for d in data_slice])
            self.task_position = np.array([d[2] for d in data_slice])
            self.CPU_circles = np.array([d[3] for d in data_slice])
            self.IoT_resource = np.array([d[4] for d in data_slice])
            self.UAV_resource = np.array([d[5] for d in data_slice])
            self.time_window = np.array([d[6] for d in data_slice])
            self.dependencys = dependency  # 共享同一依赖关系列表
            
        else:
            # 直接生成numpy数组，不转换为列表
            self.task_position = np.random.uniform(size=(num_samples, size, 2))
            self.UAV_start_pos = np.random.uniform(size=(num_samples, 1, 2))
            self.task_data = np.random.uniform(size=(num_samples, size, 1), low=0, high=1)
            self.CPU_circles = np.random.uniform(size=(num_samples, size, 1), low=0, high=1)
            self.IoT_resource = np.random.uniform(size=(num_samples, size, 1), low=2, high=5)/10
            self.UAV_resource = np.max(self.CPU_circles, axis=1, keepdims=True)/2

            # 处理依赖关系索引
            dep_indices = [i - 1 for i in dependency]
            
            # 生成时间窗口
            self.time_window = np.random.uniform(size=(num_samples, size, 2), low=0, high=10)
            self.time_window = np.sort(self.time_window, axis=2)
            
            # 处理依赖窗口（避免中间列表转换）
            dep_window = np.take(self.time_window, indices=dep_indices, axis=1)
            dep_window = np.sort(dep_window.reshape(num_samples, -1), axis=1)\
                          .reshape(num_samples, len(dep_indices), 2)
            np.put_along_axis(self.time_window, 
                             np.array(dep_indices)[None, :, None].astype(int), 
                             dep_window, axis=1)
            self.time_window[:, :, 1] = 1000

            self.dependencys = dependency  # 共享依赖关系，不重复存储

    def __getitem__(self, idx):
        # 动态生成样本，避免预加载所有张量
        return make_instance(
            self.task_data[idx],
            self.UAV_start_pos[idx],
            self.task_position[idx],
            self.CPU_circles[idx],
            self.IoT_resource[idx],
            self.UAV_resource[idx],
            self.time_window[idx],
            self.dependencys
        )

    def __len__(self):
        return self.num_samples