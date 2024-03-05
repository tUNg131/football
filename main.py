import os
import copy
import time
import random

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from model import TransformerModel
from dataset import HumanPoseDataset


CHECKPOINT_DIR = "/rds/user/tl526/hpc-work/football/checkpoints"
DATA_DIR = "/rds/user/tl526/hpc-work/football/h5/10fps/t64.hdf5"
MODEL_ARGS = ()
MODEL_KWARGS = dict(n_timestep=64,
                    n_joint=15,
                    d_joint=3,
                    d_x=3,
                    n_head=8,
                    n_layers=8,
                    d_model=1024,
                    d_hid=2048,
                    dropout=0.2)
WORLD_SIZE = 2
LOCAL_BATCH_SIZE = 64

def save(model):
    # Create directory if not exist
    os.makedirs(os.path.dirname(CHECKPOINT_DIR), exist_ok=True)

    # Save the model
    state = [MODEL_ARGS, MODEL_KWARGS, copy.deepcopy(model.module.state_dict())]
    
    filename = time.strftime(f"model_{os.environ['SLURM_JOB_ID']}.pt")
    torch.save(state, os.path.join(CHECKPOINT_DIR, filename))


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def drop(data, gap_size):
    sequence_length = data.size(dim=0)

    assert sequence_length - gap_size - 2 >= 1

    start_index = random.randint(1, sequence_length - gap_size - 2)

    data[start_index:start_index + gap_size] = float('nan')

    return data

def add_gaussian_noise(data):
    noise = torch.randn_like(data, mean=0, std=0.1)
    return data + noise

def train(rank, world_size):
    setup(rank, world_size)

    ds = HumanPoseDataset(DATA_DIR)
    generator = torch.Generator().manual_seed(42)
    train_dataset, eval_dataset = random_split(ds, [0.7, 0.3], generator=generator)

    local_model = TransformerModel(*MODEL_ARGS, **MODEL_KWARGS).to(rank)
    
    ddp_model = DDP(local_model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=LOCAL_BATCH_SIZE,
                              sampler=train_sampler)
    
    eval_sampler = DistributedSampler(eval_dataset)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=LOCAL_BATCH_SIZE,
                             sampler=eval_sampler)
    
    best_val_loss = float('inf')
    epoch = 1
    patience = 5
    gap_size = 1

    while True:
        # Training
        train_sampler.set_epoch(epoch)

        ddp_model.train()

        total_train_loss = 0.
        for raw in train_loader:
            target = raw.clone()
            data = drop(raw, gap_size)
            data = add_gaussian_noise(data)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda"):
                output = ddp_model(data)

                nan_mask = torch.isnan(data)
                loss = criterion(target[nan_mask], output[nan_mask])
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Evaluating
        eval_sampler.set_epoch(epoch)

        ddp_model.eval()

        total_eval_loss = 0.
        with torch.no_grad():
            for raw in eval_loader:
                target = raw.clone()
                data = drop(raw, gap_size)
                with torch.autocast("cuda"):
                    output = ddp_model(data)

                    nan_mask = torch.isnan(data)
                    loss = criterion(target[nan_mask], output[nan_mask])

                total_eval_loss += loss.item()

        # Sum loss from all devices
        total_train_loss = torch.tensor(total_train_loss, device=rank)
        dist.reduce(total_train_loss, dst=0)

        total_eval_loss = torch.tensor(total_eval_loss, device=rank)
        dist.reduce(total_eval_loss, dst=0)

        stop_early = torch.tensor(False, device=rank)
        if rank == 0:
            train_loss = total_train_loss.item() / len(train_dataset)
            eval_loss = total_eval_loss.item() / len(eval_dataset)

            if eval_loss < best_val_loss:
                save(ddp_model)

                best_val_loss = eval_loss
                patience = 3
            elif patience <= 0:
                if gap_size >= 15:
                    stop_early = torch.tensor(True, device=rank)
                else:
                    gap_size += 1
                    patience = 3
                    best_val_loss = float('inf')
            else:
                patience -= 1
            
            print(f"Time {datetime.now()} | epoch {epoch} | train loss {train_loss:5.6f} | eval loss {eval_loss:5.6f} | patience {patience}", flush=True)
        
        # Check for early stopping
        dist.broadcast(stop_early, src=0)
        if stop_early:
            break

        scheduler.step()

        epoch += 1

    cleanup()


if __name__ == "__main__":
    print(WORLD_SIZE)
    print(MODEL_KWARGS)
    mp.spawn(train,
             args=(WORLD_SIZE,),
             nprocs=WORLD_SIZE,
             join=True)