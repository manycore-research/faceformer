import math

import torch
import torch.nn as nn


class VanillaEmedding(nn.Module):
    def __init__(self, input_dim, num_model, token):
        super(VanillaEmedding, self).__init__()
        self.num_tokens = token.len
        
        # embedding for special tokens
        self.embedding_token = nn.Embedding(self.num_tokens, num_model)  
        self.embedding_value = nn.Sequential(
            nn.Linear(input_dim, num_model),
            nn.ReLU(),
            nn.Linear(num_model, num_model)
        )
    
    def embed_points(self, lines):
        return lines.flatten(-2, -1)

    def forward(self, coord):
        """
        coord: N x L x P x D, N batch size, L num_edges/lines, P num_points, D num_axes
        E: num_model, model input dimension
        """
        N = coord.size(0)

        token = torch.arange(self.num_tokens, dtype=torch.long).to(coord.device)
        token_embed = self.embedding_token(token)
        token_embed = token_embed.unsqueeze(0).expand(N, self.num_tokens, -1)  # N x 4 x E

        coord_embed = self.embedding_value(self.embed_points(coord)) # N x L x E
 
        value_embed = torch.cat((token_embed, coord_embed), dim=1) # N x (4+L) x E

        return value_embed


class CoordinateEmbedding(nn.Module):
    def __init__(self, num_axes, num_bits, num_embed, num_model, dependent_embed=False):
        super(CoordinateEmbedding, self).__init__()

        ntoken = 2**num_bits if dependent_embed else 2**num_bits * num_axes

        # embedding
        self.embedding_token = nn.Embedding(3, num_model)
        self.embedding_value = nn.Embedding(ntoken, num_embed)
        self.linear_proj = nn.Linear(num_axes*num_embed, num_model, bias=False)

    def forward(self, coord):
        N, S, _ = coord.shape

        # N x S x E
        token = torch.arange(3, dtype=torch.long).to(coord.device)
        token = token.unsqueeze(0).expand(N, -1)

        token_embed = self.embedding_token(token)
        coord_embed = self.embedding_value(coord)
        coord_embed = self.linear_proj(coord_embed.view(N, S, -1))

        value_embed = torch.cat((token_embed, coord_embed), dim=1)

        return value_embed


class PositionalEncoding(nn.Module):
    """
    This is a more standard version of the position embedding
    used by the Attention is all you need paper.
    """

    def __init__(self, num_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, num_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, num_model, 2).float() * (-math.log(10000.0) / num_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]   # N x S x E


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_model, max_len=5000):
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(0)
        self.register_buffer('position', position)
        self.pos_embed = nn.Embedding(max_len, num_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[:, :x.size(1)]  # N x L x E
        return self.pos_embed(pos)
