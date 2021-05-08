import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.1):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, batch, hidden):
        x = self.embeddings(batch.decoder_inputs)
        packed = pack(x, batch.decoder_lengths, enforce_sorted=False)
        x, _ = self.rnn(packed, hidden)
        x, _ = unpack(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x

