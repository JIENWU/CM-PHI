import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# 定义 Gated Convolutional Network 模型
class GatedConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, num_filters=64):
        super(GatedConvNet, self).__init__()

        # 卷积层
        self.conv = nn.Conv1d(input_dim, num_filters, kernel_size, padding=kernel_size // 2)

        # 门控机制
        self.gate = nn.Conv1d(input_dim, num_filters, kernel_size, padding=kernel_size // 2)

        # 激活函数
        self.sigmoid = nn.Sigmoid()

        # 输出层
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        # 卷积输出
        conv_out = self.conv(x)

        # 门控输出
        gate_out = self.sigmoid(self.gate(x))

        # 门控卷积的结果
        gated_output = conv_out * gate_out

        # 平均池化
        pooled_output = torch.mean(gated_output, dim=-1)

        # 全连接层
        output = self.fc(pooled_output)
        return output


# 加载数据
input_csv = 'PBI sequence.csv'  # 输入的CSV文件路径
output_csv = 'GateCNprotein_features_512.csv'  # 输出的CSV文件路径

# 读取CSV文件
data = pd.read_csv(input_csv)

# 获取蛋白质名称和序列
protein_names = data.iloc[:, 0].values  # 第一列为蛋白质名称
protein_sequences = data.iloc[:, 1].values  # 第二列为蛋白质序列

# 定义蛋白质序列的最大长度（根据实际数据长度进行调整）
MAX_LEN = 2000


# 定义一个函数，将蛋白质序列转换为数值编码
def encode_sequence(seq, max_len=MAX_LEN):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20 种氨基酸
    encoder = LabelEncoder().fit(list(amino_acids))
    seq_encoded = encoder.transform(list(seq))

    # 如果序列长度不足 max_len，进行填充
    if len(seq_encoded) < max_len:
        seq_encoded = list(seq_encoded) + [0] * (max_len - len(seq_encoded))
    return seq_encoded[:max_len]


# 将所有蛋白质序列编码为数值矩阵
encoded_sequences = [encode_sequence(seq) for seq in protein_sequences]

# 转换为Tensor
encoded_sequences = torch.tensor(encoded_sequences, dtype=torch.float32).unsqueeze(
    1)  # 添加通道维度 (batch_size, 1, sequence_length)

# 初始化模型
input_dim = 1  # 输入的维度为1，表示单通道序列
output_dim = 512  # 输出的特征向量维度
model = GatedConvNet(input_dim=input_dim, output_dim=output_dim)

# 提取特征
protein_features = []
model.eval()  # 设置模型为评估模式，不进行梯度计算

with torch.no_grad():  # 关闭梯度计算，节省内存
    for seq in tqdm(encoded_sequences):
        seq = seq.unsqueeze(0)  # 增加 batch 维度
        features = model(seq)  # 提取特征
        protein_features.append(features.squeeze(0).numpy())  # 去掉 batch 维度并转换为 numpy

# 保存特征到CSV文件
output_data = pd.DataFrame(protein_features, index=protein_names)
output_data.to_csv(output_csv, header=False)

print(f"特征已提取并保存至 {output_csv}")
