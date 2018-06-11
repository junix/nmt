import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, lang, max_seq_len, embedding_dim=100, hidden_size=512, num_layers=2):
        super(Decoder, self).__init__()
        self.lang = lang
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        self.embedding = nn.Embedding(self.lang.vocab_size, embedding_dim)

    def forward(self, word):
        pass

    def __repr__(self):
        return '<Decoder>'
