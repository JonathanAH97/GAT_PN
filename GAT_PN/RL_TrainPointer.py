# TRAIN GPN pointer with actor-critic
from GAT_PN import *
from torch.utils.data import DataLoader
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
from tqdm import tqdm
import gc
from DataGenerator import DataGenerator


def move_all(args_rl):
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if obj.device.type != args_rl.device:
                obj.data = obj.data.to(args_rl.device)


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def generate_data(args_rl):
    DG = DataGenerator()
    data_set = torch.Tensor(np.array(DG.train_batch(
        batch_size=args_rl.epoch_size, max_length=args_rl.graph_size, dimension=2)))
    return data_set


def clip_grad_norms(param_groups, max_norm=math.inf):
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            # Inf so no clipping but still call to calc
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm)
                          for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def validate(model, val_set, args_rl):
    print('Validating...')
    cost, _ = eval_model_val(model, val_set, args_rl)
    avg_cost = cost.mean()
    print(
        f'Validation avg cost: {avg_cost} +- {torch.std(cost) / math.sqrt(len(val_set))}')
    return avg_cost

#Decoding for validation
def eval_model_val(model, data_set, args_rl):
    n_nodes = args_rl.graph_size
    data_size = len(data_set)

    data_set
    mask = torch.zeros(data_size, n_nodes)
    model.eval()

    R = 0  # Actor cost
    logprobs = 0
    cost = 0

    ds_view = data_set.view(data_size, n_nodes, 2)
    x = ds_view[:, 0, :]  # First point in each batch
    h = None
    c = None
    for k in range(n_nodes):
        output, h, c, _ = model(x=x, X_all=data_set, h=h, c=c, mask=mask)

        sampler = torch.distributions.Categorical(output)
        idx = sampler.sample()
        Y1 = ds_view[[i for i in range(data_size)], idx.data].clone()
        if k == 0:
            Y_ini = Y1.clone()
        if k > 0:
            cost = torch.norm(Y1-Y0, dim=1)

        Y0 = Y1.clone()

        x = ds_view[[i for i in range(data_size)], idx.data].clone()

        R += cost

        BUFFER = 1e-15
        logprobs += torch.log(output[[i for i in range(data_size)],
                              idx.data]+BUFFER)

        # Mask updated to prevent visiting the same node twice
        mask[[i for i in range(data_size)], idx.data] += -np.inf

    
    R += torch.norm(Y1-Y_ini, dim=1)

    return R, logprobs

#Decoding for training
def eval_model(model, data_set, args_rl):
    n_nodes = args_rl.graph_size
    data_size = len(data_set)
    mask = torch.zeros(data_size, n_nodes).to(args_rl.device)
    model.train()

    R = 0  
    logprobs = 0
    cost = 0

    ds_view = data_set.view(data_size, n_nodes, 2)
    x = ds_view[:, 0, :] 
    h = None
    c = None

    for k in range(n_nodes):
        output, h, c, _ = model(x=x, X_all=data_set, h=h, c=c, mask=mask)

        sampler = torch.distributions.Categorical(output)
        idx = sampler.sample()
        Y1 = ds_view[[i for i in range(data_size)], idx.data].clone()
        if k == 0:
            Y_ini = Y1.clone()
        if k > 0:
            cost = torch.norm(Y1-Y0, dim=1)

        Y0 = Y1.clone()

        x = ds_view[[i for i in range(data_size)], idx.data].clone()

        R += cost

        BUFFER = 1e-15
        logprobs += torch.log(output[[i for i in range(data_size)],
                              idx.data]+BUFFER)

        # Mask updated to prevent visiting the same node twice
        mask[[i for i in range(data_size)], idx.data] += -np.inf
    
    R += torch.norm(Y1-Y_ini, dim=1)
    return R, logprobs



def train_batch(model, optimizer, baseline, epoch, step, batch, args_rl):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, args_rl.device)
    bl_val = move_to(bl_val, args_rl.device) if bl_val is not None else None
    # Evaluate model
    cost, logprobs = eval_model(model, x, args_rl)
    # Evaluate baseline
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    rl_loss = ((cost - bl_val) * logprobs).mean()
    loss = rl_loss + bl_loss

    # Backprop and optimize
    optimizer.zero_grad()
    loss.backward()

    # Clip gradient norms
    max_grad_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        max_grad_norm, norm_type=2)
    optimizer.step()


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_set, args_rl):
    print("Start train epoch {}, lr={} for run {}".format(
        epoch, optimizer.param_groups[0]['lr'], args_rl.run_name))
    train_step = epoch * (args_rl.epoch_size // args_rl.batch_size)
    start_time = time.time()

    # Generate new training data for each epoch
    train_set = baseline.wrap_dataset(generate_data(args_rl))
    train_loader = DataLoader(
        train_set, batch_size=args_rl.batch_size, num_workers=1)

    model.train()
    for batch_id, batch in enumerate(tqdm(train_loader)):

        #move_all(args_rl) #Ensure all tensors on same device

        train_batch(model = model, optimizer = optimizer, baseline = baseline,
                    epoch = epoch, step = train_step, batch = batch, args_rl = args_rl)

        train_step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(
        epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    avg_reward = validate(model, val_set, args_rl)

    baseline.epoch_callback(model, epoch)
    
    lr_scheduler.step()

    return avg_reward

