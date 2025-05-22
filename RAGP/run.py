import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from trainers.trainer import Trainer_Embedding_Retrieval, Trainer_SNP_Retrieval
from models.ragp import RAGP
from models.ragp_light import RAGP_light
from basic.data import TorchDataset
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with config file.")
    parser.add_argument("--config", type=str, required=False, default="./config/config_wheat599.json")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)


    data_path = config['data_path']
    if config['retrieval_time'] == 'Embedding':
        y_files = config['y_files']
    elif config['retrieval_time'] == 'SNP':
        y_files = config['y_files_origin']

    model_config = config['model']
    training_config = config['training']
    seeds_config = config['seeds']


    results = []


    for task_idx, y_data in enumerate(y_files):
        print(f"\nStarting training for Task {task_idx + 1}\n")

        task_seeds = seeds_config['task_seeds']
        task_seed = task_seeds.get(str(task_idx))
        torch.manual_seed(3407)
        np.random.seed(task_seed)
          
        dataset = TorchDataset(data_path, y_data, task_seed)
        train_dataset, val_dataset = dataset.split()

        X_train, y_train = train_dataset.get_all_data()
        num_data = len(y_train)
        train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=True)

        if config['retrieval_time'] == 'Embedding':
            input_dim = X_train.shape[1]
            hidden_dims = model_config['hidden_dim']
            hidden_dim = hidden_dims.get(str(task_idx))
            loss_alpha = config['loss_alpha'].get(str(task_idx))
            if config['enhance']:
                hidden_dim_2 = hidden_dim + 1
            else:
                hidden_dim_2 = hidden_dim
            model = RAGP(
                input_dim=input_dim,
                hidden_dim_1=hidden_dim,
                hidden_dim_2=hidden_dim_2,
                output_dim=model_config['output_dim'],
                n_blocks=model_config['n_blocks'],
                dropout_rate=model_config['dropout_rate'],
            ).to(config['device'])

            model.build_enhancelib(hidden_dim, num_data, X_train, y_train)
            optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=training_config['gamma'])

            trainer = Trainer_Embedding_Retrieval(
                target_name=task_idx,
                dataset=config['dataset'],
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                task_type=training_config['task_type'],
                config=config,
                loss_alpha=loss_alpha,
                epochs=training_config['epochs'],
                scheduler=scheduler,
                early_stopper=None
            )

        elif config['retrieval_time'] == 'SNP':
            input_dim = X_train.shape[1] + 1
            hidden_dims = model_config['hidden_dim']
            hidden_dim = hidden_dims.get(str(task_idx))
            loss_alpha = config['loss_alpha'].get(str(task_idx))
            model = RAGP_light(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=model_config['output_dim'],
                n_blocks=model_config['n_blocks'],
                dropout_rate=model_config['dropout_rate'],
            ).to(config['device'])

            model.build_enhancelib(hidden_dim, num_data, X_train, y_train)
            optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=training_config['gamma'])

            trainer = Trainer_SNP_Retrieval(
                target_name=task_idx,
                dataset=config['dataset'],
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                task_type=training_config['task_type'],
                config=config,
                loss_alpha=loss_alpha,
                epochs=training_config['epochs'],
                scheduler=scheduler,
                early_stopper=None
            )
        
        trainer.fit()

        task_results = {
            'task_idx': task_idx,
            'pearson_corr': trainer.best_pearson_corr,
            'best_epoch': trainer.best_epoch
        }
        results.append(task_results)

    print("\nSummary of all tasks:")
    for task_result in results:
        print(f"Task {task_result['task_idx']+1}: Final Pearson Correlation = {task_result['pearson_corr']:.4f}")