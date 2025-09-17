import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NodeSelector(nn.Module):
    def __init__(self,  graph_size, embed_dim, hidden_dim=128, dropout=0.1):
        super(NodeSelector, self).__init__()

        self.graph_size = graph_size

        # 用多层全连接网络进行特征变换
        self.fc1 = nn.Linear(embed_dim, hidden_dim)  # 第一层全连接层
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出一个得分
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, embed_task, embed_uav):
        
        y = embed_uav
        x = embed_task
        batch_size, gena_size, _ = x.size()
        x = torch.cat((y,x),dim=1)

        # 全连接网络：提取每个节点的特征
        x = self.relu(self.fc1( x ))  # 第一层全连接 + ReLU 激活
        x = self.dropout( x )  # Dropout
        # 计算每个节点的选择得分
        node_scores = self.fc2( x ) 
        node_scores = node_scores[:,1:].view(batch_size,gena_size) # 输出节点得分 (batch_size, gena_size , 1)
        # 通过 softmax 将得分转换为选择概率
        node_probs = F.softmax(node_scores, dim=1)  # (batch_size, gena_size , 1)

        # 根据选择概率进行采样，选择 graph_size 个节点
        log_node_probs = F.log_softmax(node_scores, dim=1)
        _ , selected_indices = torch.topk(log_node_probs.squeeze(-1), self.graph_size)
        #i
        # if self.decode_type == "greedy":
        #    selected_indices = torch.max(node_probs dim=1)[1]
        #elif self.decode_type == "sampling":
        #    selected_indices = torch.multinomial(node_probs.squeeze(-1), self.graph_size)

        #selected_indices = torch.multinomial(node_probs.squeeze(-1), self.graph_size)  # (batch_size, graph_size)
       
        return  log_node_probs,selected_indices
    

class AttNodeSelector(nn.Module):
    def __init__(self, 
                graph_size,
                embed_dim,
                ):
        """
        Task Selector with attention mechanism.
        Args:
            input_dim (int): Dimension of task features (e.g., position, time window, etc.)
            hidden_dim (int): Dimension of hidden states in attention layers
            n_heads (int): Number of attention heads
            output_dim (int): Number of tasks to select from
        """
        super(AttNodeSelector, self).__init__()
        input_dim = embed_dim
        self.graph_size = graph_size
        self.norm_factor = 1 / math.sqrt(input_dim) 

        self.W_query = nn.Parameter(torch.Tensor( input_dim, embed_dim))
        self.W_key = nn.Parameter(torch.Tensor(input_dim, embed_dim))
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
        
    def forward(self, embed_task, embed_uav):
        
        q = embed_uav    # (batch, 1, emb_dim)
        h = embed_task   # (batch, gena, emb_dim)

        batch_size, gena_size, input_dim = h.size()

        # 展平 q 和 h 以便进行矩阵乘法
        hflat = h.contiguous().view(-1, input_dim)  # (batch_size * gena_size, input_dim)
        qflat = q.contiguous().view(-1, input_dim)  # (batch_size, input_dim)

        # 计算 Queries, Keys 和 Values
        Q = torch.matmul(qflat, self.W_query).view(batch_size, 1, self.key_dim)  # (batch_size, 1, input_dim)
        K = torch.matmul(hflat, self.W_key).view(batch_size, gena_size, self.key_dim)  # (batch_size, gena_size, input_dim)
       
        # 计算注意力分数 (batch_size, 1, gena_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))  # (batch_size, 1, gena_size)

        # 计算注意力权重 (batch_size, gena_size, 1)
        attn = F.softmax(compatibility.transpose(1, 2), dim=1)

        selected_indices = torch.multinomial(attn.squeeze(-1), self.graph_size)  # (batch_size, graph_size)
       
        return  attn,selected_indices
        
