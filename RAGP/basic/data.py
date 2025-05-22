import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

class TorchDataset(Dataset):
    def __init__(self, data, label, seed):
        self.seed = seed
        if isinstance(data, str):
            if data.endswith('.pkl') or data.endswith('.pickle'):
                with open(data, 'rb') as f:
                    data = pd.read_pickle(f)
                    self.col_names = data.columns.values.tolist()
                    if isinstance(data, pd.DataFrame):
                        self.data = data.to_numpy()  
                    else:
                        self.data = data 
            elif data.endswith('.csv'):
                data = pd.read_csv(data)
                self.col_names = data.columns.values.tolist()
                if isinstance(data, pd.DataFrame):
                    self.data = data.to_numpy()  
                else:
                    self.data = data 
            else:
                raise ValueError("Unsupported file format. Only .pkl, .pickle, and .csv are supported.")
            self.label = pd.read_csv(label).to_numpy()
        else:
            self.data = data.astype(np.float32)
            self.label = label.astype(np.float32)

    def split(self, rate= 0.2):
        X_train, X_val, y_train, y_val = train_test_split(
            self.data, self.label, test_size=rate, random_state= self.seed
        )
        train = TorchDataset(X_train, y_train, self.seed)
        val = TorchDataset(X_val, y_val, self.seed)
        return train, val

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        label = torch.from_numpy(self.label[idx])
        return x, label, idx
    
    def get_col(self):
        return self.col_names
    
    def get_all_data(self):
        return self.data, self.label

    def update_y(self, label_file):
        self.label = pd.read_csv(label_file).to_numpy()


def get_loss_func(method, task_type="classification"):
    if task_type == "classification":
        return torch.nn.BCELoss()
    elif task_type == "regression":
        if method == "pearson":
            return PearsonCorrelationLoss()
        elif method == "mse":
            return torch.nn.MSELoss()
    else:
        raise ValueError("task_type must be classification or regression")


def get_metric_func(task_type="classification"):
    if task_type == "classification":
        return roc_auc_score
    elif task_type == "regression":
        return mean_squared_error
    else:
        raise ValueError("task_type must be classification or regression")


class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, preds, targets):
        preds_mean = torch.mean(preds)
        targets_mean = torch.mean(targets)

        covariance = torch.mean((preds - preds_mean) * (targets - targets_mean))
        preds_std = torch.std(preds)
        targets_std = torch.std(targets)
        pearson = covariance / (preds_std * targets_std)
        loss = 1 - pearson

        return loss


def build_triplet_pairs(target, num_pos=5, num_neg=5, batch_size=32):
    indices = torch.randperm(len(target))[:batch_size]
    target_batch = target[indices]
    diff = torch.abs(target_batch - target_batch.T)
    diff.fill_diagonal_(float('inf'))
    pos_indices = torch.topk(-diff, k=num_pos, dim=1).indices
    neg_indices = torch.topk(diff, k=num_neg, dim=1).indices
    triplets = list(zip(pos_indices, neg_indices))
    return triplets


def triplet_loss(method, anchor, pos, neg, margin=1.0):
    if method=='cosine':
        anchor = F.normalize(anchor.unsqueeze(0))
        pos = F.normalize(pos)
        neg = F.normalize(neg)
        pos_dot = torch.sum(anchor * pos, dim=2) 
        neg_dot = torch.sum(anchor * neg, dim=2) 
        loss = torch.clamp(margin - pos_dot + neg_dot, min=0.0) 
        return loss.sum()
    elif method == 'L2' or method == 'hamming':
        anchor = anchor.unsqueeze(0)
        pos_dist = torch.norm(anchor - pos, p=2, dim=2)
        neg_dist = torch.norm(anchor - neg, p=2, dim=2)
        loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        return loss.mean()
    