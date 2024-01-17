import math
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(num_embeddings, embedding_dim)

        k = torch.arange(0, num_embeddings).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[x].requires_grad_(False)


class Embedding(nn.Module):

    def __init__(self, timesteps, n_joint, d_joint, d_x, d_model, dropout=0.2):
        super().__init__()

        self.d_x = d_x

        self.linear = nn.Linear(
            in_features=d_x, out_features=d_model
        )

        self.time_embedding = TimeEmbedding(
            num_embeddings=timesteps, embedding_dim=d_model, dropout=dropout
        )

        num_space_embeddings = (n_joint * d_joint) // d_x
        self.space_embedding = nn.Embedding(
            num_embeddings=num_space_embeddings, embedding_dim=d_model
        )

        self.nan_embedding = nn.Embedding(
            num_embeddings=2, embedding_dim=d_model
        )


    def forward(self, x):
        bsize, timesteps, n_joint, d_joint = x.shape

        n_token = (n_joint * d_joint) // self.d_x

        # time embedding
        time_emb = self.time_embedding(
            torch.arange(timesteps, dtype=torch.int, device=x.device)
            .view(1, -1, 1)
            .repeat(bsize, 1, n_token)
            .view(bsize, -1)
        )

        # space embedding
        space_emb = self.space_embedding(
            torch.arange(n_token, dtype=torch.int, device=x.device)
            .repeat(bsize, timesteps)
        )

        # nan embedding
        nan_emb = self.nan_embedding(
            torch.isnan(x)
            .view(bsize, -1, self.d_x)
            .any(-1)
            .int()
        )

        x = self.linear(
            torch.nan_to_num(x)
            .view(bsize, -1, self.d_x)
        )

        emb = x + time_emb + space_emb + nan_emb
        return emb
    

class TransformerModel(nn.Module):

    def __init__(self,
                 n_timestep=32,
                 n_joint=29,
                 d_joint=3,
                 d_x=3,
                 n_head=32,
                 n_layers=8,
                 d_model=256,
                 d_hid=512,
                 dropout=0.2):

        super().__init__()

        self.embedding = Embedding(n_timestep, n_joint, d_joint, d_x, d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.linear = nn.Linear(d_model, d_x)

    def forward(self, x):
        emb = self.embedding(x)

        output = self.transformer_encoder(emb)

        output = self.linear(output)

        output = output.view(x.shape)

        return output


class InterpolationModel(nn.Module):
    def forward(self, x):
        nan_mask = torch.flatten(torch.isnan(x), start_dim=2).any(dim=2).float()

        start = nan_mask.argmax(dim=1) - 1
        end = x.size(dim=1) - nan_mask.flip(dims=(1,)).argmax(dim=1)

        for i, (s, e) in enumerate(zip(start, end)):
            gap_size = e - s - 1
            weights = torch.linspace(0, 1, gap_size + 2)[1:-1]
            x[i, s + 1:e] = torch.lerp(x[i, s:s+1].expand(gap_size, -1, -1),
                                       x[i, e:e+1].expand(gap_size, -1, -1),
                                       weights.view(-1, 1, 1))
            
        return x