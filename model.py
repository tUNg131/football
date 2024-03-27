import torch
from torch import nn, Tensor


class MixtureDensityHead(nn.Module):
    
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 n_gaussian: int) -> None:
        super().__init__()

        self.pi = nn.Linear(d_in, n_gaussian)
        self.sigma = nn.Sequential(
            nn.Linear(d_in, d_out * n_gaussian),
            nn.ELU(),
        )
        self.mu = nn.Linear(d_in, d_out * n_gaussian)

    def forward(self, x: Tensor) -> Tensor:
        pi = self.pi(x)
        # Applying modified ELU activation
        sigma = self.sigma(x) + 1 + 1e-15
        mu = self.mu(x)
        return pi, sigma, mu


class Embedding(nn.Module):

    def __init__(self,
                 n_timesteps: int,
                 n_joints: int,
                 d_in: int,
                 d_model: int) -> None:
        super().__init__()

        self.linear = nn.Linear(d_in, d_model)
        self.time_embedding = nn.Embedding(n_timesteps, d_model)
        self.joint_embedding = nn.Embedding(n_joints, d_model)
        self.nan_embedding = nn.Embedding(2, d_model)

        self.register_buffer(
            "timestep_labels",
            torch.arange(n_timesteps, dtype=torch.int).repeat_interleave(n_joints),
            persistent=False
        )

        self.register_buffer(
            "joint_labels",
            torch.arange(n_joints, dtype=torch.int).repeat(n_timesteps),
            persistent=False
        )

    def forward(self, x: Tensor):
        batch_size, *_ = x.size()

        time_emb = self.time_embedding(self.timestep_labels.repeat(batch_size, 1))

        joint_emb = self.joint_embedding(self.joint_labels.repeat(batch_size, 1))

        nan_emb = self.nan_embedding(x.isnan().any(dim=-1, keepdim=False).int())

        val_emb = self.linear(x.nan_to_num(nan=0.0))

        return val_emb + time_emb + joint_emb + nan_emb


class FootballTransformer(nn.Module):

    def __init__(self,
                 n_timesteps: int,
                 n_joints: int,
                 d_in: int,
                 d_out: int,
                 n_heads: int,
                 n_layers: int,
                 d_model: int,
                 d_feedforward: int,
                 dropout: int,
                 activation: int,
                 n_gaussian: int) -> None:
        super().__init__()

        self.embedding = Embedding(n_timesteps=n_timesteps,
                                   n_joints=n_joints,
                                   d_in=d_in,
                                   d_model=d_model)
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True
            ),
            num_layers=n_layers,
        )

        self.head = MixtureDensityHead(d_in=d_model,
                                       d_out=d_out,
                                       n_gaussian=n_gaussian)
        
        self.n_tokens = n_timesteps * n_joints

    def forward(self, x: Tensor) -> Tensor:
        embed = self.embedding(x)
        feature = self.encoder(embed)
        output = self.head(feature)
        return output