import torch
import torch.nn as nn
import torch.nn.functional as F

class Enhancelib():
    def __init__(self, hidden_dim, num_data, all_data, all_labels):
        self.hidden_dim = hidden_dim
        self.num_data = num_data
        self.topk_similarities = torch.zeros(num_data, dtype=torch.float32)
        self.cosin_max = torch.zeros(num_data, dtype=torch.float32)
        self.all_data = torch.tensor(all_data, dtype=torch.float32)
        self.all_labels = torch.tensor(all_labels, dtype=torch.float32)

    def to(self, device):
        self.cosin_max = self.cosin_max.to(device)
        self.all_labels = self.all_labels.to(device)
        self.all_data = self.all_data.to(device)



class Block(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(Block, self).__init__()
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # Residual connection
        residual = x
        out = self.layernorm(x)
        out = self.linear1(out)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out + residual
    

class RAGP_light(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks, dropout_rate):
        super(RAGP_light, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([Block(hidden_dim, dropout_rate) for _ in range(n_blocks)])
        self.fc3 = nn.Linear(hidden_dim, output_dim)


    def build_enhancelib(self, hidden_dim, num_data, all_data, all_labels):
        self.enhancelib = Enhancelib(hidden_dim, num_data, all_data, all_labels)
        self.enhancelib.to(next(iter(self.parameters())))

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc3(x)

        return x
    


    def get_re_train(self, x_index, x, fusion_alpha):
        similarities = self.enhancelib.topk_similarities[x_index]
        weights = torch.nn.functional.softmax(similarities, dim=-1).unsqueeze(-1)
        indices = self.enhancelib.cosin_max[x_index].long()
        # top特征
        top_features = self.enhancelib.all_data[indices]
        top_features_mean = (weights * top_features).sum(dim=1)
        f_re = fusion_alpha * x + ( 1 - fusion_alpha ) * top_features_mean
        # top标签
        top_labels = self.enhancelib.all_labels[indices]
        top_labels_mean = (weights * top_labels).sum(dim=1)
        # 特征+标签拼接
        inputs = torch.cat([f_re, top_labels_mean * 0.01], dim=-1)
        return inputs
     

    def get_re_test(self, x, fusion_alpha):
        with torch.no_grad():
            cosin_matrix = torch.mm(F.normalize(x), F.normalize(self.enhancelib.all_data).t())
            similarities, indices = torch.topk(cosin_matrix, k=5, dim=1, largest=True, sorted=True)
            weights = torch.nn.functional.softmax(similarities, dim=-1).unsqueeze(-1)
            # top特征
            top_features = self.enhancelib.all_data[indices]
            top_features_mean = (weights * top_features).sum(dim=1)
            f_re = fusion_alpha * x + ( 1 - fusion_alpha ) * top_features_mean
            # top标签
            top_labels = self.enhancelib.all_labels[indices]
            top_labels_mean = (weights * top_labels).sum(dim=1)
            # 特征+标签拼接
            inputs = torch.cat([f_re, top_labels_mean * 0.01], dim=-1) 
        return inputs
    
   

    def update(self):
        with torch.no_grad():
            data_list_norma = F.normalize(self.enhancelib.all_data)
            cosin_matrix = torch.mm(data_list_norma, data_list_norma.t()) - torch.eye(len(self.enhancelib.all_data)).to(self.enhancelib.all_data.device)
            topk_similarities, topk_indices = torch.topk(cosin_matrix, k=5, dim=1, largest=True, sorted=True)
            self.enhancelib.topk_similarities = topk_similarities
            self.enhancelib.cosin_max = topk_indices