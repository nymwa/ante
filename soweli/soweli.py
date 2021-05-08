import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Soweli(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_encoder_layers=1, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers = num_encoder_layers)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, dropout = dropout)
        self.proj = nn.Linear(hidden_size * num_encoder_layers * 2, hidden_size)

    def encode(self, batch):
        h = self.encoder(batch)
        h = h.transpose(0, 1).flatten(1).unsqueeze(0)
        h = self.proj(h)
        return h

    def forward(self, batch):
        h = self.encode(batch)
        x = self.decoder(batch, h)
        return x

