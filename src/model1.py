import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch


class Baseline(nn.Module):
    def __init__(self, tuple_size, vocab_sizes, embedding_sizes=[256, 256, 256], n_layers=3, hidden_size=512,
                 dropout=0.0):
        super(Baseline, self).__init__()
        self.tuple_size = tuple_size
        self.vocab_sizes = vocab_sizes
        self.embedding_sizes = embedding_sizes
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embeddings = nn.ModuleList([nn.Embedding(v + 1, s) for v, s in zip(vocab_sizes, embedding_sizes)])
        self.scoring_layers = nn.ModuleList([nn.Linear(hidden_size, v + 1) for v in vocab_sizes])
        self.rnn = nn.LSTM(input_size=sum(embedding_sizes), num_layers=n_layers, hidden_size=hidden_size,
                           dropout=dropout)
        self.init_c = nn.Parameter(torch.randn((n_layers, 1, hidden_size)))
        self.init_u = nn.Parameter(torch.randn((n_layers, 1, hidden_size)))

    def forward(self, x):
        tuples, lengths = x
        max_len = len(tuples)
        batch_size = len(lengths)
        embeddings = torch.cat([self.embeddings[i](tuples[:, :, i]) for i in range(self.tuple_size)], dim=2)
        packed_embeddings = rnn_utils.pack_padded_sequence(embeddings, lengths)
        init_c = self.init_c.expand(-1, batch_size, -1).contiguous()
        init_u = self.init_u.expand(-1, batch_size, -1).contiguous()
        outputs, _ = self.rnn(packed_embeddings, (init_c, init_u))
        padded_outputs, lengths = rnn_utils.pad_packed_sequence(outputs)
        logits = [self.scoring_layers[i](padded_outputs.view(-1, self.hidden_size)).view(max_len, batch_size, -1) for i
                  in range(self.tuple_size)]
        return logits, lengths
