import torch
import os
import numpy as np
from tqdm import tqdm
from basic.data import get_loss_func, get_metric_func, triplet_loss, build_triplet_pairs
from scipy.stats import pearsonr

class Trainer_Embedding_Retrieval:
    def __init__(self, target_name, dataset, model, train_loader, val_loader, optimizer, task_type, config, epochs, loss_alpha, scheduler=None, early_stopper=None):
        self.target_name = target_name
        self.dataset = dataset
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.task_type = task_type
        self.config = config
        self.n_epoch = epochs
        self.loss_alpha = loss_alpha
        self.scheduler = scheduler
        self.early_stopper = early_stopper
        self.best_pearson_corr = -1 
        self.best_epoch = -1

        self.loss_fn = get_loss_func(config['loss'], task_type)
        self.evaluate_fn = get_metric_func(task_type)


    def fit(self):
        for epoch_i in range(self.n_epoch):
            train_loss = self.train_one_epoch(epoch_i, self.train_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            val_score, pearson_corr = self.evaluate(self.model, self.val_loader)
            print(f"Epoch {epoch_i + 1}/{self.n_epoch}, Training Loss: {train_loss:.4f}, "
                  f"Validation Score: {val_score:.4f}, Pearson Correlation: {pearson_corr:.4f}")

            if pearson_corr > self.best_pearson_corr:
                self.best_pearson_corr = pearson_corr
                self.best_epoch = epoch_i
                # best_model_path = os.path.join(
                #     self.config['model_path'], 
                #     f"{self.config['dataset']}_{self.target_name}_best_model.pth"
                # )
                # torch.save(self.model.state_dict(), best_model_path)

            if self.early_stopper is not None:
                if self.early_stopper.stop_training(val_score, self.model.state_dict()):
                    print(f"Early stopping at epoch {epoch_i + 1} with best validation score: {self.early_stopper.best_score:.6f}")
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break


    def train_one_epoch(self, epoch_i, train_dataloader):
        self.model.train()
        running_loss = 0.0

        for batch_x, batch_y, index in train_dataloader:
            batch_x = batch_x.to(self.config['device'])
            batch_y = batch_y.to(self.config['device'])
            index = index.to(self.config['device'])
            self.optimizer.zero_grad()

            if self.config['enhance']:
                inputs, x_encoded = self.model.get_re_train(index, batch_x, self.config['distance'], self.config['fusion_alpha'])
            else:
                x_encoded = self.model.encoder(batch_x)
                inputs = x_encoded

            if self.config['contrastive_loss']:
                triplets = build_triplet_pairs(batch_y)
                tloss = 0.0
                for i, (pos_indices, neg_indices) in enumerate(triplets):
                    anchor = x_encoded[i].unsqueeze(0)
                    pos = x_encoded[pos_indices]
                    neg = x_encoded[neg_indices]
                    tloss += triplet_loss(self.config['distance'], anchor, pos, neg)
                tloss /= len(triplets)

                predictions = self.model(inputs)
                alpha= self.loss_alpha * 0.1
                loss = self.loss_fn(predictions, batch_y) + alpha * tloss 
            else:
                predictions = self.model(inputs)
                loss = self.loss_fn(predictions, batch_y)

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        if self.config['enhance']:
            self.model.update(self.config['distance'])
        return running_loss / len(train_dataloader)


    def evaluate(self, model, val_dataloader):
        model.eval()
        all_predictions, all_labels = [], []

        with torch.no_grad():
            for x_test, y_test, index in val_dataloader:
                x_test = x_test.to(self.config['device'])
                y_test = y_test.to(self.config['device'])
                index = index.to(self.config['device'])
                if self.config['enhance']:
                    x = model.get_re_test(x_test, self.config['distance'], self.config['fusion_alpha'])
                else:
                    x = model.encoder(x_test)
                predictions = model(x)

                all_predictions.extend(predictions.cpu().numpy())  
                all_labels.extend(y_test.cpu().numpy())

        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()
        pearson_corr, _ = pearsonr(all_predictions, all_labels)
        
        if self.task_type == "classification":
            all_predictions = 1 / (1 + np.exp(-all_predictions))
            score = self.evaluate_fn(all_labels, all_predictions)
        elif self.task_type == "regression":
            score = self.evaluate_fn(all_labels, all_predictions)

        return score, pearson_corr

    def get_reference(self, model, val_dataloader):
        model.eval()
        all_indices, all_self_labels, all_labels, all_features = [], [], [], []

        with torch.no_grad():
            for x_test, y_test, index in val_dataloader:
                x_test = x_test.to(self.config['device'])
                indices, labels, features = model.get_references(x_test)

                all_indices.append(indices)
                all_labels.append(labels)
                all_features.append(features)
                all_self_labels.append(y_test.cpu()) 

        all_indices = torch.cat(all_indices, dim=0)  
        all_labels = torch.cat(all_labels, dim=0)
        all_features = torch.cat(all_features, dim=0)
        all_self_labels = torch.cat(all_self_labels, dim=0)

        return all_indices, all_labels, all_features, all_self_labels



class Trainer_SNP_Retrieval:
    def __init__(self, target_name, dataset, model, train_loader, val_loader, optimizer, task_type, config, epochs, loss_alpha, scheduler=None, early_stopper=None):
        self.target_name = target_name
        self.dataset = dataset
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.task_type = task_type
        self.config = config
        self.n_epoch = epochs
        self.loss_alpha = loss_alpha
        self.scheduler = scheduler
        self.early_stopper = early_stopper
        self.best_pearson_corr = -1 
        self.best_epoch = -1

        self.loss_fn = get_loss_func(config['loss'], task_type)
        self.evaluate_fn = get_metric_func(task_type)


    def fit(self):
        self.model.update()
        for epoch_i in range(self.n_epoch):
            train_loss = self.train_one_epoch(epoch_i, self.train_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            val_score, pearson_corr = self.evaluate(self.model, self.val_loader)
            print(f"Epoch {epoch_i + 1}/{self.n_epoch}, Training Loss: {train_loss:.4f}, "
                  f"Validation Score: {val_score:.4f}, Pearson Correlation: {pearson_corr:.4f}")

            if pearson_corr > self.best_pearson_corr:
                self.best_pearson_corr = pearson_corr
                self.best_epoch = epoch_i

            if self.early_stopper is not None:
                if self.early_stopper.stop_training(val_score, self.model.state_dict()):
                    print(f"Early stopping at epoch {epoch_i + 1} with best validation score: {self.early_stopper.best_score:.6f}")
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break


    def train_one_epoch(self, epoch_i, train_dataloader):
        self.model.train()
        running_loss = 0.0

        for batch_x, batch_y, index in train_dataloader:
            batch_x = batch_x.to(self.config['device'])
            batch_y = batch_y.to(self.config['device'])
            index = index.to(self.config['device'])
            self.optimizer.zero_grad()
            inputs = self.model.get_re_train(index, batch_x, self.config['fusion_alpha'])
            predictions = self.model(inputs)
            loss = self.loss_fn(predictions, batch_y)

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / len(train_dataloader)


    def evaluate(self, model, val_dataloader):
        model.eval()
        all_predictions, all_labels = [], []

        with torch.no_grad():
            for x_test, y_test, index in val_dataloader:
                x_test = x_test.to(self.config['device'])
                y_test = y_test.to(self.config['device'])
                index = index.to(self.config['device'])
                x = model.get_re_test(x_test, self.config['fusion_alpha'])
                predictions = model(x)

                all_predictions.extend(predictions.cpu().numpy())  
                all_labels.extend(y_test.cpu().numpy())

        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()
        pearson_corr, _ = pearsonr(all_predictions, all_labels)
        
        if self.task_type == "classification":
            all_predictions = 1 / (1 + np.exp(-all_predictions))
            score = self.evaluate_fn(all_labels, all_predictions)
        elif self.task_type == "regression":
            score = self.evaluate_fn(all_labels, all_predictions)
        return score, pearson_corr