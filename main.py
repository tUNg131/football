import os
import copy
import time
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


CHECKPOINT_DIR = "checkpoints"
DATA_DIR = "/home/tl526/rds/hpc-work/football/h5/t32.hdf5"
MODEL_ARGS = ()
MODEL_KWARGS = dict(n_timestep=32,
                    n_joint=29,
                    d_x=3,
                    d_model=256,
                    n_head=8,
                    d_hid=512,
                    n_layers=8)

def save(model):
    return
    # Create directory if not exist
    os.makedirs(os.path.dirname(CHECKPOINT_DIR), exist_ok=True)

    # Save the model
    state = [MODEL_ARGS, MODEL_KWARGS, copy.deepcopy(model.state_dict())]
    
    filename = time.strftime("model_%Y-%m-%d_%H-%M-%S.pt")
    torch.save(state, os.path.join(CHECKPOINT_DIR, filename))


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.barrier()
    dist.destroy_process_group()


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
                              batch_size=16,
                              sampler=train_sampler)
    
    eval_sampler = DistributedSampler(eval_dataset)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=16,
                             sampler=eval_sampler)
    
    best_val_loss = float('inf')
    # epoch = 1
    patience = 3

    # while True:
    for epoch in range(1, 11):
        tstart = time.time()

        # Training
        train_sampler.set_epoch(epoch)

        ddp_model.train()

        total_train_loss = 0.
        for data, target in train_loader:
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
            for data, target in eval_loader:                
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

        stop_early = torch.zeros(1, dtype=torch.bool, device=rank)

        if rank == 0:
            train_loss = total_train_loss.item() / len(train_loader)
            eval_loss = total_eval_loss.item() / len(eval_loader)

            print(f"epoch {epoch} | train loss {train_loss:5.4f} | eval loss {eval_loss:5.4f}")
            if eval_loss < best_val_loss:

                save(ddp_model)

                best_val_loss = eval_loss
                patience = 3
            elif patience <= 0:
                stop_early = True
            else:
                patience -= 1
        
        # # Check for early stopping
        # dist.broadcast(stop_early, src=0)
        # if stop_early:
        #     break

        scheduler.step()

        epoch += 1

    cleanup()


if __name__ == "__main__":
    # 4 GPUs
    world_size = 4

    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)