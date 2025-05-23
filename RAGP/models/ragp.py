import torch
import torch.nn as nn
import torch.nn.functional as F

class Enhancelib():
    def __init__(self, hidden_dim, num_data, all_data, all_labels):
        self.hidden_dim = hidden_dim
        self.num_data = num_data
        self.topk_similarities = torch.zeros(num_data, dtype=torch.float32)
        self.cosin_max = torch.zeros(num_data, dtype=torch.float32)
        self.features_list = torch.zeros([num_data, hidden_dim], dtype=torch.float32)
        self.all_data = torch.tensor(all_data, dtype=torch.float32)
        self.all_labels = torch.tensor(all_labels, dtype=torch.float32)
        # self.hamming_train_index=self.hamming_distance(self.all_data)

    def to(self, device):
        self.cosin_max = self.cosin_max.to(device)
        self.features_list = self.features_list.to(device)
        self.all_labels = self.all_labels.to(device)
        self.all_data = self.all_data.to(device)

    # def hamming_distance(self, data, threshold=521):
    #     distances = torch.sum(data.unsqueeze(0) != data.unsqueeze(1), dim=2)
    #     mask = distances < threshold
    #     results = []
    #     for i in range(data.shape[0]):
    #         indices = torch.nonzero(mask[i] & (torch.arange(data.shape[0]) != i), as_tuple=False).squeeze()
    #         results.append((i, indices))
    #     return results


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.fc(x)


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
    

class RAGP(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, n_blocks, dropout_rate):
        super(RAGP, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.encoder = Encoder(input_dim, hidden_dim_1)
        self.blocks = nn.ModuleList([Block(hidden_dim_2, dropout_rate) for _ in range(n_blocks)])
        self.fc = nn.Linear(hidden_dim_2, output_dim)


    def build_enhancelib(self, hidden_dim, num_data, all_data, all_labels):
        self.enhancelib = Enhancelib(hidden_dim, num_data, all_data, all_labels)
        self.enhancelib.to(next(iter(self.parameters())))

    def forward(self, x):

        for block in self.blocks:
            x = block(x)
        x = self.fc(x)

        return x
    



    def get_re_train(self, x_index, x, method, fusion_alpha):
        if method == 'L2':
            f_re, x_encoded = self.get_re_train_L2(x_index, x)
            return f_re, x_encoded
        elif method == 'cosine':
            f_re, x_encoded = self.get_re_train_cos(x_index, x, fusion_alpha)
            return f_re, x_encoded
        elif method == 'hamming':
            f_re, x_encoded = self.get_re_train_hamming(x_index, x)
            return f_re, x_encoded
        
    def get_re_test(self, x, method, fusion_alpha):
        if method == 'L2':
            f_re = self.get_re_test_L2(x)
            return f_re
        elif method == 'cosine':
            f_re = self.get_re_test_cos(x, fusion_alpha)
            return f_re
        elif method == 'hamming':
            f_re = self.get_re_test_hamming(x)
            return f_re  


    def get_re_train_cos(self, x_index, x, fusion_alpha):
        x_encoded = self.encoder(x)
        if torch.norm(self.enhancelib.features_list) == 0:
            zeros = torch.zeros(x_encoded.shape[0], 1).to(x_encoded.device)
            inputs = torch.cat([x_encoded, zeros], dim=-1)
            return inputs, x_encoded
        similarities = self.enhancelib.topk_similarities[x_index]
        weights = torch.nn.functional.softmax(similarities, dim=-1).unsqueeze(-1)
        indices = self.enhancelib.cosin_max[x_index].long()
        # top特征
        top_features = self.enhancelib.features_list[indices]
        top_features_mean = (weights * top_features).sum(dim=1)
        f_re = fusion_alpha * x_encoded + ( 1 - fusion_alpha ) * top_features_mean
        # top标签
        top_labels = self.enhancelib.all_labels[indices]
        top_labels_mean = (weights * top_labels).sum(dim=1)
        # 特征+标签拼接
        inputs = torch.cat([f_re, top_labels_mean * 0.01], dim=-1)
        return inputs, x_encoded
     

    def get_re_test_cos(self, x, fusion_alpha):
        with torch.no_grad():
            x_encoded = self.encoder(x)
            if torch.norm(self.enhancelib.features_list) == 0:
                zeros = torch.zeros(x_encoded.shape[0], 1).to(x_encoded.device)
                inputs = torch.cat([x_encoded, zeros], dim=-1)
                return inputs
            cosin_matrix = torch.mm(F.normalize(x_encoded), F.normalize(self.enhancelib.features_list).t())
            similarities, indices = torch.topk(cosin_matrix, k=5, dim=1, largest=True, sorted=True)
            weights = torch.nn.functional.softmax(similarities, dim=-1).unsqueeze(-1)
            # top特征
            top_features = self.enhancelib.features_list[indices]
            top_features_mean = (weights * top_features).sum(dim=1)
            f_re = fusion_alpha * x_encoded + ( 1 - fusion_alpha ) * top_features_mean
            # top标签
            top_labels = self.enhancelib.all_labels[indices]
            top_labels_mean = (weights * top_labels).sum(dim=1)
            # 特征+标签拼接
            inputs = torch.cat([f_re, top_labels_mean * 0.01], dim=-1) 
        return inputs
    
    def get_re_train_L2(self, x_index, x, k=5):
        x_encoded = self.encoder(x) 
        with torch.no_grad():
            if torch.norm(self.enhancelib.features_list) == 0:
                return x_encoded, x_encoded
            distances = torch.cdist(x_encoded, self.enhancelib.features_list, p=2) 
            batch_size = x_encoded.shape[0]
            for i in range(batch_size):
                distances[i, x_index[i]] = float('inf')
            _, indices = torch.topk(-distances, k, dim=1)
            top_features = self.enhancelib.features_list[indices]
            f_re = torch.mean(torch.cat([x_encoded.unsqueeze(1), top_features], dim=1), dim=1)
        return f_re, x_encoded
    
    def get_re_test_L2(self, x, k=5):
        with torch.no_grad():
            x_encoded = self.encoder(x)
            if torch.norm(self.enhancelib.features_list) == 0:
                return x_encoded
            distances = torch.cdist(x_encoded, self.enhancelib.features_list, p=2) 
            _, indices = torch.topk(-distances, k, dim=1)
            top_features = self.enhancelib.features_list[indices]
            f_re = torch.mean(torch.cat([x_encoded.unsqueeze(1), top_features], dim=1), dim=1)  
        return f_re
    
    
    def get_re_train_hamming(self, x_index, x, k=5):
        # x_encoded = F.normalize(self.encoder(x))
        x_encoded = self.encoder(x)
        with torch.no_grad():
            f_re = []
            d = 0
            for idx in x_index:
                candidate_data_index = self.enhancelib.hamming_train_index[idx][1]
                candidate_data = self.enhancelib.all_data[candidate_data_index] 
                # candidate_encoded = F.normalize(self.encoder(candidate_data))
                candidate_encoded = self.encoder(candidate_data)
                # L2
                distances = torch.cdist(x_encoded[d].unsqueeze(0), candidate_encoded, p=2)
                _, indices = torch.topk(-distances, k, dim=1)
                top_features = candidate_encoded[indices]
                top_features_mean = top_features.squeeze(0).mean(dim=0)
                x_encoded_d_squeezed = x_encoded[d]
                final_mean = (top_features_mean + x_encoded_d_squeezed) / 2
                f_re.append(final_mean)
                d+=1
            f_ree = torch.stack(f_re)
        return f_ree, x_encoded
    

    def get_re_test_hamming(self, x, k=5, threshold=521):
        with torch.no_grad():
            x_encoded = self.encoder(x)
            # x_encoded = F.normalize(self.encoder(x))
            f_re=[]
            for i in range(x.shape[0]):
                distances = torch.sum(x[i] != self.enhancelib.all_data, dim=1)
                indices = torch.where(distances < threshold)[0]
                candidate_data = self.enhancelib.all_data[indices] 
                # candidate_encoded = F.normalize(self.encoder(candidate_data))
                candidate_encoded = self.encoder(candidate_data)
                # L2
                distances = torch.cdist(x_encoded[i].unsqueeze(0), candidate_encoded, p=2)
                _, indices = torch.topk(-distances, k, dim=1)
                top_features = candidate_encoded[indices]
                top_features_mean = top_features.squeeze(0).mean(dim=0)
                x_encoded_d_squeezed = x_encoded[i]
                final_mean = (top_features_mean + x_encoded_d_squeezed) / 2
                f_re.append(final_mean)
            f_ree = torch.stack(f_re)
        return f_ree



    def get_references(self, x):
        with torch.no_grad():
            x_encoded = self.encoder(x)
            cosin_matrix = torch.mm(F.normalize(x_encoded), F.normalize(self.enhancelib.features_list).t())
            similarities, indices = torch.topk(cosin_matrix, k=5, dim=1, largest=True, sorted=True)
            weights = torch.nn.functional.softmax(similarities, dim=-1).unsqueeze(-1)
            # top标签
            top_labels = self.enhancelib.all_labels[indices]
            top_features = self.enhancelib.features_list[indices]
        return indices.cpu(), top_labels.cpu(), top_features.cpu()


    def update(self, method):
        with torch.no_grad():
            if method == 'cosine':
                self.enhancelib.features_list = self.encoder(self.enhancelib.all_data)
                feature_list_norma = F.normalize(self.enhancelib.features_list)
                cosin_matrix = torch.mm(feature_list_norma, feature_list_norma.t()) - torch.eye(len(self.enhancelib.features_list)).to(self.enhancelib.features_list.device)
                topk_similarities, topk_indices = torch.topk(cosin_matrix, k=5, dim=1, largest=True, sorted=True)
                self.enhancelib.topk_similarities = topk_similarities
                self.enhancelib.cosin_max = topk_indices
            else:
                self.enhancelib.features_list = self.encoder(self.enhancelib.all_data)