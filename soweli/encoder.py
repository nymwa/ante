import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, bidirectional=True, num_layers=num_layers)

    def forward(self, batch):
        x = self.embedding(batch.encoder_inputs)
        packed = pack(x, batch.encoder_lengths, enforce_sorted=False)
        _, h = self.rnn(packed)
        return h

