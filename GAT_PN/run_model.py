from GAT_PN import *
from torch.utils.data import DataLoader
import pprint as pp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random
import os
import time
import argparse
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from DataGenerator import DataGenerator
from RL_args import get_arguments
from RL_baselines import *
from RL_TrainPointer import *

def run_rl(args_rl):

    DG = DataGenerator()

    pp.pprint(vars(args_rl))

    torch.manual_seed(args_rl.seed)

    model = GPN(2, args_rl.hidden_dim, args_rl.tanh_clipping, args_rl.vector_context, args_rl.graph_encoding).to(args_rl.device)

    if args_rl.baseline == 'exponential':
        baseline = ExponentialBaseline(args_rl.exp_beta)
    elif args_rl.baseline == 'critic':
        baseline = CriticBaseline(
            CriticNetworkLSTM(
                2,
                args_rl.hidden_dim
            ).to(args_rl.device))
    else:
        assert args_rl.baseline is None
        baseline = NoBaseline()
    
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': args_rl.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': args_rl.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    #lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: args_rl.lr_decay ** epoch)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, range(args_rl.lr_decay_steps, args_rl.lr_decay_steps*1000,
                                     args_rl.lr_decay_steps), gamma=args_rl.lr_decay)
    val_dataset = torch.Tensor(np.array(DG.test_batch(
        batch_size=args_rl.val_size, max_length=args_rl.graph_size, dimension=2, seed = args_rl.seed))).to(args_rl.device)

    start_time = time.time()
    if args_rl.eval_only:
        validate(model, val_dataset, args_rl)
    else:
        val_costs = []
        best_val_cost = np.inf
        patience = 10
        epochs_no_improve = 0
        for epoch in range(args_rl.n_epochs):
            avg_rwd = train_epoch(
                model = model,
                optimizer = optimizer,
                baseline = baseline,
                lr_scheduler = lr_scheduler,
                epoch = epoch,
                val_set = val_dataset,
                args_rl = args_rl
            )
            val_costs.append(avg_rwd)

            if val_costs[-1] < best_val_cost:
                best_val_cost = val_costs[-1]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs without improvement.")
                break

    computation_time = time.time() - start_time
    val_costs.append([computation_time])
    #Plot the progression in validation
    model_name = 'GPN_' + str(args_rl.graph_size) + '_' + str(args_rl.baseline) + '_VC-' + str(args_rl.vector_context) + '_GE-' + str(args_rl.graph_encoding)
    epochs = range(1, len(val_costs))
    #Save val_costs
    with open('VAL_COSTS/' + model_name + '_val_costs.pkl', 'wb') as f:
        pickle.dump(val_costs, f)   

    try:
        plt.plot(epochs, val_costs[:-1], marker = 'o', label = model_name)
        plt.xlabel('Epochs')
        plt.ylabel('Average Cost')
        plt.title('Average Validation cost per Epoch')
        plt.legend()
        plt.show()
    except:
        pass
    finally:
        #Save model
        torch.save(model, 'RL_Models/' + model_name + '.pt')
if __name__ == "__main__":
    #run_rl(get_arguments(['--baseline', 'critic', '--graph_size', '20', '--n_epochs', '50', '--graph_encoding', 'GAT', '--vector_context', '1', '--val_size','1000']))
    #run_rl(get_arguments(['--baseline', 'exponential', '--graph_size', '20', '--n_epochs', '50', '--graph_encoding', 'GAT', '--vector_context', '1', '--val_size','1000']))
    #run_rl(get_arguments(['--baseline', 'critic', '--graph_size', '20', '--n_epochs', '50', '--graph_encoding', 'GAT', '--vector_context', '0', '--val_size','1000']))
    #run_rl(get_arguments(['--baseline', 'exponential', '--graph_size', '20', '--n_epochs', '50', '--graph_encoding', 'GAT', '--vector_context', '0', '--val_size','1000']))
    run_rl(get_arguments(['--baseline', 'critic', '--graph_size', '50', '--n_epochs', '30', '--graph_encoding', 'GAT', '--vector_context', '1', '--epoch_size','6400','--batch_size','256', '--val_size','500']))
    run_rl(get_arguments(['--baseline', 'exponential', '--graph_size', '50', '--n_epochs', '30', '--graph_encoding', 'GAT', '--vector_context', '1', '--epoch_size','6400','--batch_size','256', '--val_size','500']))
    run_rl(get_arguments(['--baseline', 'exponential', '--graph_size', '75', '--n_epochs', '30', '--graph_encoding', 'GAT', '--vector_context', '1', '--epoch_size','3200','--batch_size','128', '--val_size','250']))
    run_rl(get_arguments(['--baseline', 'critic', '--graph_size', '75', '--n_epochs', '30', '--graph_encoding', 'GAT', '--vector_context', '1', '--epoch_size','3200','--batch_size','128', '--val_size','250']))
