import torch
import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self, timesteps, n_joint, d_x, d_model):
        super().__init__()

        self.linear = nn.Linear(
            in_features=d_x * n_joint, out_features=d_model
        )

        self.time_embedding = nn.Embedding(
            num_embeddings=timesteps, embedding_dim=d_model
        )

        self.space_embedding = nn.Embedding(
            num_embeddings=d_x * n_joint, embedding_dim=d_model
        )

        self.nan_embedding = nn.Embedding(
            num_embeddings=2, embedding_dim=d_model
        )

    def forward(self, x):
        bsize, timesteps, n_joint, d_x = x.shape

        # time embedding
        pos_emb = self.time_embedding(
            torch.arrange(timesteps, dtype=torch.int)
            .to(x.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(bsize, -1, d_x * n_joint)
            .contiguous()
            .view(bsize, -1)
        )

        # space embedding
        space_emb = self.space_embedding(
            torch.arrange(d_x * n_joint, dtype=torch.int)
            .to(x.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(bsize, -1, timesteps)
            .contiguous()
            .view(bsize, -1)
        )

        # flatten x
        x = torch.flatten(x, start_dim=1)

        # nan embedding
        nan_emb = self.nan_embedding(torch.isnan(x))

        x = self.linear(torch.nan_to_num(x))
        emb = x + pos_emb + space_emb + nan_emb
        return emb
    

class TransformerModel(nn.Module):

    def __init__(self, timesteps, d_x, d_model, n_head, d_hid, n_layers, dropout=0.2):
        super().__init__()

        self.embedding = Embedding(timesteps, d_x, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        bsize, timesteps, n_joint, d_x = x.shape

        emb = self.embedding(x)

        output = self.transformer_encoder(emb)

        output = self.linear(output)

        output = output.view(bsize, timesteps, n_joint, d_x)
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