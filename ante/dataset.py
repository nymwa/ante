import random as rd
import torch
from torch.nn.utils.rnn import pad_sequence as pad
from .batch import Batch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sents, vocab):
        self.sents = sents
        self.vocab = vocab
        self.pad = self.vocab.pad_id
        self.eos = self.vocab.eos_id

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        return self.sents[index]

    def collate(self, batch):
        ei = [torch.tensor(sent) for sent in batch]
        di = [torch.tensor([self.eos] + sent) for sent in batch]
        do = [torch.tensor(sent + [self.eos]) for sent in batch]
        el = torch.tensor([len(sent) for sent in batch])
        dl = torch.tensor([len(sent) + 1 for sent in batch])
        return Batch(
                pad(ei, padding_value = self.pad),
                pad(di, padding_value = self.pad),
                pad(do, padding_value = self.pad),
                el, dl)

class AnteDataset(Dataset):
    def collate(self, batch):
        ei = [sent[:] for sent in batch]
        for x in ei:
            rd.shuffle(x)
        ei = [torch.tensor(sent) for sent in ei]

        di = [torch.tensor([self.eos] + sent) for sent in batch]
        do = [torch.tensor(sent + [self.eos]) for sent in batch]
        el = torch.tensor([len(sent) for sent in batch])
        dl = torch.tensor([len(sent) + 1 for sent in batch])
        return Batch(
                pad(ei, padding_value = self.pad),
                pad(di, padding_value = self.pad),
                pad(do, padding_value = self.pad),
                el, dl)

