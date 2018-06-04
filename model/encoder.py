import torch
import torch.nn as nn
from conf import run_device


class Encoder(nn.Module):
    def __init__(self, lang, embedding_dim=100, hidden_size=512):
        super(Encoder, self).__init__()
        self.lang = lang
        self.num_layers = 2
        self.hidden_size = hidden_size
        self.bidirectional = True
        self.embedding = nn.Embedding(num_embeddings=lang.vacab_size, embedding_dim=embedding_dim)
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            dropout=0.5,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional
        )

    def begin_state(self):
        bidirect = 2 if self.bidirectional else 1
        return torch.zeros(self.num_layers * bidirect, 1, self.hidden_size, dtype=torch.float, device=run_device())

    def forward(self, words, state):
        words = words.to(run_device())
        input_len = len(words)
        words = self.embedding(words).view(input_len, 1, -1)
        output, state = self.run(words, state)
        return output, state
