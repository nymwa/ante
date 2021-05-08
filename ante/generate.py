import sys
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from ante.batch import Batch
from tunimi.normalizer import Normalizer
from tunimi.tokenizer import Tokenizer
from soweli.soweli import Soweli

class ScoredSentence:
    def __init__(self, log_probs, sent, hidden):
        self.log_probs = log_probs
        self.sent = sent
        self.hidden = hidden

    def last(self):
        return self.sent[-1]

    def score(self):
        penalty = (len(self.log_probs) + 5) / 6
        penalty = penalty ** 0.6
        return sum(self.log_probs) / penalty

def make_original_list(sent, vocab):
    original = [0 for i in range(len(vocab))]
    for token in sent:
        if token != vocab.pad_id:
            original[token] += 1
    original[vocab.eos_id] = 1
    return original

class AnteScoredSentence(ScoredSentence):
    def __init__(self, log_probs, sent, hidden, original, vocab_size):
        super().__init__(log_probs, sent, hidden)
        self.vocab_size = vocab_size
        self.original = original

    def constrain(self, score):
        for i in range(self.vocab_size):
            if self.original[i] == 0:
                score[i] = - float('inf')
        return score

    def get_new_sent(self, log_prob, token, hidden):
        log_probs = self.log_probs + [log_prob]
        sent = self.sent + [token]
        original = self.original[:]
        original[token] -= 1
        return AnteScoredSentence(log_probs, sent, hidden, original, vocab_size=self.vocab_size)

def predict(model, beam):
    decoder_inputs = torch.tensor([[sent.last() for sent in beam]]).cuda()
    hidden_states = torch.cat([sent.hidden for sent in beam], dim=-2).cuda()
    lengths = [1 for _ in range(len(beam))]
    with torch.no_grad():
        x = model.decoder.embeddings(decoder_inputs)
        packed = pack(x, lengths, enforce_sorted=False)
        x, hidden_states = model.decoder.rnn(packed, hidden_states)
        x, _ = unpack(x)
        x = model.decoder.proj(x)
    return x, hidden_states

def update_beam(old_beam, scores, hidden_states, width):
    new_beam = []
    for n in range(len(old_beam)):
        score = scores[0, n]
        score = old_beam[n].constrain(score)
        score = torch.log_softmax(score, dim=-1)
        values, indices = score.topk(width)
        for v, i in zip(values, indices):
            if v != -float('inf'):
                sent = old_beam[n].get_new_sent(v.item(), i.item(), hidden_states[:,n:n+1,:])
                new_beam.append(sent)
    new_beam.sort(key = lambda sent:-sent.score())
    new_beam = new_beam[:width]
    return new_beam

def split_beam(vocab, old_beam):
    new_beam = []
    ended = []
    for sent in old_beam:
        if sent.last() == vocab.eos_id:
            if all(num == 0 for num in sent.original):
                ended.append(sent)
        else:
            new_beam.append(sent)
    return new_beam, ended

def beam_search(model, vocab, hidden, width, original, max_len = 128):
    model.eval()
    beam = [AnteScoredSentence(log_probs = [], sent = [vocab.eos_id], hidden = hidden, original=make_original_list(original, vocab), vocab_size=len(vocab))]
    output = []
    for i in range(max_len):
        if len(beam) == 0:
            break
        scores, hidden_states = predict(model, beam)
        beam = update_beam(beam, scores, hidden_states, width - len(output))
        beam, ended = split_beam(vocab, beam)
        output += ended
    output.sort(key = lambda sent:-sent.score())
    return output

def encode(model, tokenizer, x):
    encoder_inputs = torch.tensor([x]).cuda()
    batch = Batch(
            pad(encoder_inputs, padding_value = tokenizer.vocab.pad_id),
            None, None, torch.tensor([len(x)]), None)
    with torch.no_grad():
        h = model.encode(batch)
    return h

def main():
    normalizer = Normalizer()
    tokenizer = Tokenizer()
    x = sys.stdin.readline().strip()
    x = normalizer(x)
    x = tokenizer(x)
    model = Soweli(len(tokenizer.vocab), 32, 128, 1, 0)
    model.load_state_dict(torch.load('checkpoints/checkpoint.99.pt', map_location='cpu'))
    model = model.cuda()
    h = encode(model, tokenizer, x)
    beam = beam_search(model, tokenizer.vocab, h, 12, x)
    for sent in beam:
        score = np.exp(sent.score())
        print(' '.join([tokenizer.vocab[n] for n in sent.sent]) + '\t' + str(score))

