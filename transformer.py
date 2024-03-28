import os
from argparse import ArgumentParser, Namespace
from typing import Tuple, Type, Optional

import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import optimizer
from model import FootballTransformer
from dataset import (
    HumanPoseMidHipDataset,
    DropRandomChunkVariableSize,
    DropRandomUniform,
    DropRandomChunk,
    DEFAULT_JOINT_INDICES
)


DEFAULT_LOADER_WORKER_COUNT = min(os.cpu_count() - 1, 32)


class TrainableFootballTransformer(pl.LightningModule):

    class DataModule(pl.LightningDataModule):
        __dataset_class__  = HumanPoseMidHipDataset

        def __init__(self, model: Type[pl.LightningModule]) -> None:
            super().__init__()
            self.save_hyperparameters(model.hparams)
            self.model = model

        def setup(self, stage: str) -> None:
            drop_type, *drop_args = self.hparams.masking_strategy
            if drop_type == "vchunk":
                drop = DropRandomChunkVariableSize(*drop_args)
            elif drop_type == "chunk":
                drop = DropRandomChunk(*drop_args)
            elif drop_type == "random":
                drop = DropRandomUniform(*drop_args)
            else:
                raise

            if stage == "fit":
                self._train_dataset = self.__dataset_class__(
                    path=self.hparams.train_path,
                    drop=drop,
                    noise=self.hparams.training_noise_std,
                )

                self._val_dataset = self.__dataset_class__(
                    path=self.hparams.val_path,
                    drop=DropRandomChunkVariableSize(15),
                )

        def train_dataloader(self) -> DataLoader:
            """Function that loads the train set."""
            return DataLoader(dataset=self._train_dataset,
                              shuffle=True,
                              batch_size=self.hparams.train_batch_size,
                              num_workers=self.hparams.loader_workers,
                              pin_memory=self.model.on_gpu)

        def val_dataloader(self) -> DataLoader:
            """Function that loads the validation set."""
            return DataLoader(dataset=self._val_dataset,
                              shuffle=False,
                              batch_size=self.hparams.val_batch_size,
                              num_workers=self.hparams.loader_workers,
                              pin_memory=self.model.on_gpu)

        def get_test_dataloader(self, gap_size) -> DataLoader:
            dataset = self.__dataset_class__(
                path=self.hparams.test_path,
                drop=DropRandomChunk(gap_size),
            )
            return DataLoader(
                dataset=dataset,
                shuffle=False,
                batch_size=self.hparams.test_batch_size,
                num_workers=self.hparams.loader_workers,
                pin_memory=self.model.on_gpu
            )

    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        # Build Data module
        self.datamodule = self.DataModule(self)

        # Build model
        self.__build_model()

    def __build_model(self) -> None:
        """Init the model."""
        self._model = FootballTransformer(n_timesteps=self.hparams.n_timesteps,
                                          n_joints=self.hparams.n_joints,
                                          d_in=self.hparams.d_in,
                                          d_out=self.hparams.d_out,
                                          n_heads=self.hparams.n_heads,
                                          n_layers=self.hparams.n_layers,
                                          d_model=self.hparams.d_model,
                                          d_feedforward=self.hparams.d_feedforward,
                                          dropout=self.hparams.dropout,
                                          activation=self.hparams.activation,
                                          n_gaussian=self.hparams.n_gaussian)
    
    def __select_dim1(self, x: Tensor, mask: Tensor) -> Tensor:
        return x[mask]

    def __to_model_format(self, *inputs):
        return (
            x.view(-1, self._model.n_tokens, self.hparams.d_in)
            for x in inputs
        )
        
    def __loss(self, x: Tensor, y: Tensor) -> Tensor:
        nan_position = self.__nan_mask(x)
        pi, sigma, mu = self._model(x)

        pi = self.__select_dim1(pi, nan_position)
        sigma = self.__select_dim1(sigma, nan_position)
        mu = self.__select_dim1(mu, nan_position) 
        y = self.__select_dim1(y, nan_position).repeat(1, self.hparams.n_gaussian)

        log_pi = torch.log_softmax(pi, dim=-1)
        log_normal_prob = (
            (- torch.log(sigma) + 0.5 * torch.pow((y - mu) / sigma, 2))
            .view(-1, self.hparams.n_gaussian, self.hparams.d_out)
            .sum(dim=-1)
        )
        return -torch.logsumexp(log_pi + log_normal_prob, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)
    
    def __nan_mask(self, x: Tensor) -> Tensor:
        return x.isnan().any(dim=(2,))
    
    def __nan_start_end_timesteps(self, x: Tensor) -> Tensor:
        # Find positions of NaN values
        nan_positions = self.__nan_mask(x).float()

        # Find the first and last NaN positions
        start_nan = nan_positions.argmax(dim=1)
        end_nan =  self._model.n_tokens - nan_positions.flip(dims=(1,)).argmax(dim=1) - 1

        return start_nan, end_nan

    def training_step(self,
                      batch: Tuple[Tensor, Tensor],
                      batch_index: int) -> Tensor:
        
        inputs, targets = self.__to_model_format(*batch)
        loss = self.__loss(inputs, targets).mean()
        self.log("train_nll", loss, sync_dist=True)
        return loss

    def validation_step(self,
                        batch: Tuple[Tensor, Tensor],
                        batch_index: int) -> Tensor:
        
        inputs, targets = self.__to_model_format(*batch)
        loss =  self.__loss(inputs, targets).mean()
        self.log("val_nll", loss, sync_dist=True)
        return loss
    
    def on_test_epoch_start(self) -> None:
        # Initialize test losses at the start of each epoch
        self.model_loss = []
    
    def test_step(self,
                  batch: Tuple[Tensor, Tensor],
                  batch_index: int,
                  dataloader_idx: Optional[int] = 0) -> None:

        inputs, targets = self.__to_model_format(*batch)
        batch_size = inputs.size(0)
        out = inputs.clone()

        while out.isnan().any():
            # Inference
            pi, sigma, mu = self._model(out)
            pi = torch.softmax(pi, dim=-1)
            sigma = (
                sigma
                .view(batch_size, -1, self.hparams.n_gaussian, self.hparams.d_out)
                .prod(dim=-1)
            )
            mu = mu.view(batch_size, -1, self.hparams.n_gaussian, self.hparams.d_out)
            idx = (
                torch.argmax(pi / sigma, dim=-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, -1, mu.size(-1))
            )

            pred = torch.gather(mu, 2, idx)

            s, e = self.__nan_start_end_timesteps(out)

            i = torch.arange(batch_size)
            out[i, s] = pred[i, s]
            out[i, e] = pred[i, e]

        nan_position = self.__nan_mask(inputs)
        loss = (
            nn.functional
            .mse_loss(targets[nan_position], out[nan_position], reduction='none')
            .view(batch_size, -1, self.hparams.n_joints, self.hparams.d_out)
        )

        self.model_loss.append(loss)

    def on_test_epoch_end(self) -> None:
        """
        Performs actions at the end of each test epoch.
        """

        np_model_loss_avg = (
            torch.cat(self.model_loss, dim=1)
            .mean(dim=(0, 2, 3), keepdim=False)
            .cpu()
            .numpy()
        )

        missing_timesteps_count = len(np_model_loss_avg)
        x = np.arange(missing_timesteps_count)
        rects = plt.bar(x, np_model_loss_avg)
        plt.bar_label(rects, padding=3)

        plt.xlabel("MSE")
        plt.ylabel("Timesteps")
        plt.yticks(x, x + 1)
        plt.title(f"MSE by strategies (gap size = {missing_timesteps_count})")
        
        # Add the plot to TensorBoard
        self.logger.experiment.add_figure(f"Loss/gapsize = {missing_timesteps_count}", plt.gcf())

        # Clear the test losses for the next epoch
        self.model_loss.clear()
 
    def configure_optimizers(self) -> Optimizer:
        optimizer_name = self.hparams.optimizer
        try:
            optimizer_class = getattr(optimizer, optimizer_name)
        except AttributeError:
            optimizer_class = getattr(torch.optim, optimizer_name)

        optimizer_obj = optimizer_class(
            self._model.parameters(),
            lr=self.hparams.learning_rate
        )
        return optimizer_obj

    @classmethod
    def update_parser_with_model_args(cls: Type[pl.LightningModule],
                                      parser: ArgumentParser) -> None:
        parser.add_argument(
            "--n-timesteps",
            default=32,
            type=int,
            help="# of timesteps in the input."
        )
        parser.add_argument(
            "--n-joints",
            default=len(DEFAULT_JOINT_INDICES),
            type=int,
            help="# of joints in one timestep."
        )
        parser.add_argument(
            "--d-in",
            default=3,
            type=int,
            help="Dimension of input tokens."
        )
        parser.add_argument(
            "--d-out",
            default=3,
            type=int,
            help="Dimension of output tokens."
        )
        parser.add_argument(
            "--n-heads",
            default=8,
            type=int,
            help="# of attention heads."
        )
        parser.add_argument(
            "--n-layers",
            default=8,
            type=int,
            help="# of layers."
        )
        parser.add_argument(
            "--d-model",
            default=1024,
            type=int,
            help="Dimension of the model (embedding size)."
        )
        parser.add_argument(
            "--d-feedforward",
            default=2048,
            type=int,
            help="Dimension of the feedforward layer."
        )
        parser.add_argument(
            "--dropout",
            default=0.2,
            type=float,
            help="Dropout probability."
        )
        parser.add_argument(
            "--activation",
            default="relu",
            type=str,
            help="The activation function of the layers"
        )
        parser.add_argument(
            "--n-gaussian",
            default=5,
            type=int,
            help="Number of Gaussian components in mixture density network."
        )
        parser.add_argument(
            "--learning-rate",
            default=0.0015,
            type=float,
            help="Model learning rate."
        )
        parser.add_argument(
            "--loader-workers",
            default=DEFAULT_LOADER_WORKER_COUNT,
            type=int,
            help=(
                "How many subprocesses to use for data loading."
                " 0 means that the data will be loaded in the main process."
            )
        )
        parser.add_argument(
            "--optimizer",
            default="JITLamb",
            type=str,
            help="Name of the optimizer. Can be inside torch.optim."
        )
        parser.add_argument(
            "--train-path",
            default="/home/tl526/rds/hpc-work/football/data/h5/10fps/train.hdf5",
            type=str,
            help="Path to the file containing the training data."
        )
        parser.add_argument(
            "--val-path",
            default="/home/tl526/rds/hpc-work/football/data/h5/10fps/val.hdf5",
            type=str,
            help="Path to the file containing the validation data."
        )
        parser.add_argument(
            "--test-path",
            default="/home/tl526/rds/hpc-work/football/data/h5/10fps/test.hdf5",
            type=str,
            help="Path to the file containing the test data."
        )
        parser.add_argument(
            "--train-batch-size",
            default=512,
            type=int,
            help="Batch size to be used."
        )
        parser.add_argument(
            "--val-batch-size",
            default=512,
            type=int,
            help="Batch size to be used."
        )
        parser.add_argument(
            "--test-batch-size",
            default=5120,
            type=int,
            help="Batch size to be used."
        )
        parser.add_argument(
            "--training-noise-std",
            default=0.02,
            type=float,
            help="Training noise standard deviation."
        )
        parser.add_argument(
            "--masking-strategy",
            default=["random", 60],
            nargs="+",
            help="Mask strategy. e.g. vchunk15, chunk15, random50.",
        )