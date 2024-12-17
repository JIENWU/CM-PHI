import pandas as pd
import torch
import torch.nn as nn


# Transformer模型定义
class FeatureFusionTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(FeatureFusionTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)  # 输入特征映射到模型维度
        self.positional_encoding = nn.Parameter(torch.randn(1, 500, model_dim))  # 位置编码

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_fc = nn.Linear(model_dim, output_dim)  # 输出层

    def forward(self, feature1, feature2):
        # 将两个特征拼接为一个序列 (batch_size, 2, input_dim)
        combined_features = torch.cat((feature1, feature2), dim=1)  # (batch_size, seq_len, input_dim)
        combined_features = self.input_fc(combined_features)  # 映射到 model_dim 维度

        # 加上位置编码
        combined_features = combined_features + self.positional_encoding[:, :combined_features.size(1), :]

        # 通过 Transformer 编码器
        fused_features = self.transformer_encoder(combined_features)

        # 输出融合后的特征，使用 mean 聚合特征
        fused_features = self.output_fc(fused_features.mean(dim=1))

        return fused_features


# 加载CSV文件并提取特征
def load_features_from_csv(file_path):
    df = pd.read_csv(file_path)
    protein_names = df.iloc[:, 0].values  # 获取蛋白质名称
    features = df.iloc[:, 1:].values  # 获取特征向量
    features = torch.tensor(features, dtype=torch.float32)  # 转换为PyTorch张量
    return protein_names, features


# 保存融合后的特征到CSV
def save_fused_features_to_csv(protein_names, fused_features, output_file):
    # 转换为 numpy 格式
    fused_features_np = fused_features.detach().numpy()
    # 构造保存的 DataFrame，蛋白质名称为第一列，融合后的特征为后续列
    df = pd.DataFrame(fused_features_np, columns=[f'Feature_{i}' for i in range(fused_features_np.shape[1])])
    df.insert(0, 'Protein', protein_names)  # 插入蛋白质名称作为第一列
    df.to_csv(output_file, index=False)  # 保存为CSV文件


# 主函数：融合两个特征文件
def fuse_protein_features_with_transformer(file1, file2, output_file, input_dim, model_dim, num_heads, num_layers,
                                           output_dim):
    # 读取两个CSV文件的蛋白质名称和特征
    protein_names1, features1 = load_features_from_csv(file1)
    protein_names2, features2 = load_features_from_csv(file2)

    # 检查蛋白质名称是否一致
    if not (protein_names1 == protein_names2).all():
        raise ValueError("两个文件的蛋白质名称不匹配")

    # 定义Transformer融合网络
    fusion_model = FeatureFusionTransformer(input_dim, model_dim, num_heads, num_layers, output_dim)

    # 进行前向传播，得到融合后的特征
    fused_features = fusion_model(features1.unsqueeze(1), features2.unsqueeze(1))

    # 保存融合后的特征到CSV文件
    save_fused_features_to_csv(protein_names1, fused_features, output_file)


# 示例文件路径
file1 = 'GateCNprotein_features_512.csv'  # 第一种特征
file2 = 'topological_features_512_hop8.csv'  # 第二种特征
output_file = 'fused_protein_features_512_hop8.csv'  # 输出融合特征的文件

# 参数设置
input_dim = 512  # 假设输入特征的维度为64
model_dim = 128  # Transformer的内部维度
num_heads = 4  # 多头注意力的头数
num_layers = 2  # Transformer的层数
output_dim = 512  # 最终输出融合特征的维度

# 执行特征融合
fuse_protein_features_with_transformer(file1, file2, output_file, input_dim, model_dim, num_heads, num_layers,
                                       output_dim)
