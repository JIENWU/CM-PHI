import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# 读取CSV文件
df = pd.read_csv('PBI 最终异构网络.csv')
phages = df.iloc[:, 0].values
hosts = df.iloc[:, 1].values

# 构建节点和边的映射
all_nodes = list(set(phages) | set(hosts))
node_mapping = {node: i for i, node in enumerate(all_nodes)}

# 将交互关系转换为边
edge_index = torch.tensor([[node_mapping[phage], node_mapping[host]] for phage, host in zip(phages, hosts)], dtype=torch.long).t().contiguous()

# 假设节点特征为单位矩阵，每个节点都是独立的一个特征
x = torch.eye(len(all_nodes), dtype=torch.float)

# 构建PyTorch Geometric数据对象
data = Data(x=x, edge_index=edge_index)

# 定义Multi-Hop Attention GNN模型
class MultiHopAttentionGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, num_hops=3):
        super(MultiHopAttentionGNN, self).__init__()
        self.num_hops = num_hops
        self.attention_layers = torch.nn.ModuleList()
        for _ in range(num_hops):
            self.attention_layers.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
            in_channels = hidden_channels * heads
        self.out_layer = GATConv(in_channels, out_channels, heads=heads, concat=False)

    def forward(self, x, edge_index):
        for i in range(self.num_hops):
            x = self.attention_layers[i](x, edge_index)
            x = torch.relu(x)
        x = self.out_layer(x, edge_index)
        return x

# 模型参数
in_channels = data.num_node_features
hidden_channels = 2
out_channels = 512  # 输出的嵌入维度
heads = 4
num_hops = 8

# 初始化模型
model = MultiHopAttentionGNN(in_channels, hidden_channels, out_channels, heads=heads, num_hops=num_hops)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 模型训练
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # 输出节点的嵌入表示
    loss = torch.sum(out)  # 可以根据任务定义损失函数
    loss.backward()
    optimizer.step()

# 训练循环
for epoch in range(1, 51):
    train()
    print(f'Epoch {epoch}: Training...')

# 提取节点的拓扑特征
with torch.no_grad():
    model.eval()
    node_embeddings = model(data.x, data.edge_index)
    print("节点嵌入表示：", node_embeddings)

# 将节点嵌入表示转换为DataFrame
node_embeddings_np = node_embeddings.cpu().numpy()  # 转换为numpy数组
node_ids = list(node_mapping.keys())  # 获取节点的原始ID

# 构建DataFrame
embedding_df = pd.DataFrame(node_embeddings_np, index=node_ids)
embedding_df.index.name = 'Node_ID'  # 设置索引列名称

# 保存到CSV文件
embedding_df.to_csv('topological_features_512_hop8.csv', index=True)
print("拓扑结构特征已保存到 topological_features.csv 文件")
