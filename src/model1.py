import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Baseline(nn.Module):
    def __init__(self, tuple_size, vocab_sizes, embedding_sizes=[256, 256, 256], n_layers=3, hidden_size=512,
                 dropout=0.0):
        super(Baseline, self).__init__()
        self.tuple_size = tuple_size
        self.vocab_sizes = vocab_sizes
        self.eos_indices = vocab_sizes.copy()
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
        return logits, 

    def decode(self,max_length=500,style=2):

        max_len = 1
        batch_size = 1
        x = self.eos_indices
        inp = torch.tensor(x).unsqueeze(0).unsqueeze(1).to(device)
        init_c = self.init_c.expand(-1, batch_size, -1).contiguous()
        init_u = self.init_u.expand(-1, batch_size, -1).contiguous()
        i=0
        generated_output=[]
        while(i<max_length):
            embeddings = torch.cat([self.embeddings[i](inp[:, :, i]) for i in range(self.tuple_size)], dim=2)            
            outputs, (init_c,init_u) = self.rnn(embeddings, (init_c, init_u))
            logits = [self.scoring_layers[i](outputs.view(-1, self.hidden_size)).view(max_len, batch_size, -1) for i
                    in range(self.tuple_size)]
            op_tuple = self.get_sample(logits)
            if i>0:
                op_tuple, stop = self.check_tuple(op_tuple,generated_output[-1],style) #check for end of sequence
                if stop:
                    break
            generated_output.append(op_tuple)
            inp = torch.tensor(op_tuple).unsqueeze(0).unsqueeze(1).to(device)
            i=i+1
        generated_outputs = np.array(generated_output)
        return generated_outputs

    def get_sample(self,logits):
        pitch = torch.distributions.Categorical(logits=logits[0]).sample().item()
        offset = torch.distributions.Categorical(logits=logits[1]).sample().item()
        duration = torch.distributions.Categorical(logits=logits[2]).sample().item()
        return (pitch,offset,duration)

    def check_tuple(self,op_tuple,prev_tuple,style):
        stop=0
        if style==1:
            if op_tuple == self.eos_indices:
                stop=1
            if op_tuple[0]== self.eos_indices[0] :
                p = prev_tuple[0]
            else:
                p = op_tuple[0]
            if op_tuple[1]== self.eos_indices[1] :
                o = prev_tuple[1]
            else:
                o = op_tuple[1] 
            if op_tuple[2]== self.eos_indices[2] :
                d = prev_tuple[2]
            else:
                d = op_tuple[2]
            
            return (p,o,d),stop
        else:
            if op_tuple[0]== self.eos_indices[0] or op_tuple[1]== self.eos_indices[1] or op_tuple[2]== self.eos_indices[2]:
                stop=1
            return op_tuple,stop



