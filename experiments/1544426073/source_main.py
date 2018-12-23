from utils.train_utils import *
from utils.pianoroll_utils import *

import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
import torch.nn.functional as F
import pickle

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_path = "../data"
file_names = ["jsb.pkl", "muse.pkl", "nottingham.pkl", "piano.pkl"]

class PianoRollDatasetBase:

    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.data) - 1) // self.batch_size + 1
    
    def __iter__(self):
        n_batches = len(self)
        indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(n_batches):
            yield self.data[indices[i * self.batch_size: (i + 1) * self.batch_size]]

class PianoRollDatasetChunked:

    def __init__(self, data, batch_size, l_mu1=150, l_mu2=75, l_s1=10, l_s2=10, n1_prob=0.95, shuffle=True):
        self.data = []
        for song in data:
            length = len(song)
            matrix = np.zeros((length, 89), dtype=np.int64)
            for i, point in enumerate(song):
                matrix[i, point - 21] = 1
            self.data.append(matrix)
        self.data = np.array(self.data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.eos = [0] * 88 + [1]
        self.l_mu1 = l_mu1
        self.l_mu2 = l_mu2
        self.l_s1 = l_s1
        self.l_s2 = l_s2
        self.n1_prob = n1_prob
    
    def __iter__(self):
        if self.shuffle:
            temp = self.data.copy()
            np.random.shuffle(temp)
        else:
            temp = self.data
        temp = np.concatenate([np.concatenate([i, [self.eos]], axis=0) for i in temp], axis=0)
        temp = np.concatenate([[self.eos], temp], axis=0)
        n_notes = len(temp) - 1
        ip = temp[:-1]
        op = temp[1:]
        ip = ip[: self.batch_size * (n_notes // self.batch_size)]
        op = op[: self.batch_size * (n_notes // self.batch_size)]
        ip = ip.reshape((self.batch_size, -1, 89)).transpose(1, 0, 2)
        op = op.reshape((self.batch_size, -1, 89)).transpose(1, 0, 2)
        left = len(op)
        ptr = 0
        while left > 0:
            L = 0
            samp = np.random.random()
            if samp < self.n1_prob:
                while L <= 0:
                    L = int(np.random.normal(self.l_mu1, self.l_s1))
            else:
                while L <= 0:
                    L = int(np.random.normal(self.l_mu2, self.l_s2))
            L = min(left, L, int(self.l_mu1 + 3 * self.l_s1))
            left -= L
            batch_x, batch_y = ip[ptr : ptr + L], op[ptr : ptr + L]
            ptr += L
            lengths = torch.LongTensor([L] * self.batch_size).to(device)
            batch_x = torch.LongTensor(batch_x).float().to(device)
            batch_y = torch.LongTensor(batch_y).float().to(device)
            inv_order = np.arange(self.batch_size)
            yield (batch_x, lengths), (batch_y, lengths), inv_order

    def __len__(self):
        return sum([len(i) for i in self.data]) // int((self.l_mu1 * self.n1_prob + self.l_mu2 * (1- self.n1_prob)) * self.batch_size)


class BaselineModel(nn.Module):

    def __init__(self):
        super(BaselineModel, self).__init__()
        self.embedding  = nn.Sequential(nn.Linear(89, 128), nn.ELU(), nn.Linear(128, 256), nn.ELU(), nn.Linear(256, 256))
        self.score = nn.Sequential(nn.Linear(256, 89), nn.Sigmoid())
        self.rnn = nn.LSTM(input_size=256, num_layers=3, hidden_size=256)
        self.h0 = nn.Parameter(torch.randn((3, 1, 256)))
        self.c0 = nn.Parameter(torch.randn((3, 1, 256)))
    
    def forward(self, x):
        feats, lengths = x
        batch_size = len(lengths)
        max_len = len(feats)
        embeddings = self.embedding(feats.view(-1, 89)).view(max_len, batch_size, -1)
        packed = rnn_utils.pack_padded_sequence(embeddings, lengths)
        h0 = torch.cat([self.h0] * batch_size, dim=1)
        c0 = torch.cat([self.c0] * batch_size, dim=1)
        outputs, _ = self.rnn(packed, (h0, c0))
        padded, lengths = rnn_utils.pad_packed_sequence(outputs)
        scores = self.score(padded.view(batch_size * max_len, -1)).view(max_len, batch_size, -1)
        return scores, lengths
    
    def binary_sample(self, x):
        return (np.random.random(len(x)) < x ).astype(np.int64)

    
    def sample(self, max_len=250):
        with torch.no_grad():
            inp = np.zeros((1, 1, 89), dtype=np.float32)
            inp[0, 0, 88] = 1
            sequence = []
            h0, c0 = self.h0, self.c0
            for i in range(max_len):
                inp = torch.FloatTensor(inp).to(device)
                embeddings = self.embedding(inp.view(-1, 89)).view(1, 1, -1)
                output, (h0, c0) = self.rnn(embeddings, (h0, c0))
                scores = self.score(output.view(1, -1)).view(1, 1, -1)
                inp = self.binary_sample(scores.data.cpu().numpy().squeeze()).astype(np.float32)
                if inp[88] > 0:
                    continue
                else:
                    sequence.append(inp)
                    inp = inp[None, None, :]
            return np.array(sequence).astype(np.int64)

class BaselineLoss(nn.Module):

    def __init__(self):
        super(BaselineLoss, self).__init__()
    
    def forward(self, x, targets, reduce=True):
        probs, _ = x
        max_length = probs.shape[0]
        batch_size = probs.shape[1]
        targs, lengths = targets
        loss = -(targs * torch.log(probs) + (1 - targs) * torch.log(1 - probs))
        ranges = torch.arange(max_length).to(device).view(-1, 1)
        mask = (ranges < lengths.view(1, -1).to(device)).float().unsqueeze(2)
        mask = mask.expand_as(loss)
        loss = loss * mask
        if reduce:
            return loss.sum() / batch_size
        else:
            return loss


def load_dataset(train_batch_size=8, eval_batch_size=8, dataset=3):
    data = pickle.load(open(join(data_path, file_names[dataset]), "rb"))
    trainds = PianoRollDatasetChunked(data["train"], train_batch_size)
    valds = PianoRollDatasetChunked(data["valid"], eval_batch_size, shuffle=False)
    testds = PianoRollDatasetChunked(data["test"], eval_batch_size, shuffle=False)
    return trainds, valds, testds

def validate(model, valds, criterion):
    model.eval()
    with torch.no_grad():
        lengths_sum = 0
        loss_sum = 0
        acc_sum = 0
        bar = tqdm(valds, desc="Validating")
        for ip, targ, inv_order in bar:
            op = model(ip)
            loss = criterion(op, targ, reduce=False).sum().item()
            loss_sum += loss
            lengths_sum += ip[1].sum().item()
            op = (op[0].data.cpu().numpy() > 0.5).astype(np.int64)
            targ = targ[0].data.cpu().numpy().astype(np.int64)
            eq = (op == targ).astype(np.int64)
            max_len = op.shape[0]
            batch_size = op.shape[1]
            ranges = np.arange(max_len).reshape((-1, 1))
            lengths = ip[1].cpu().numpy()
            mask = (ranges < lengths.reshape((1, -1))).reshape((max_len, batch_size, 1))
            eq = eq * mask
            acc_sum += eq.sum()
        return loss_sum / lengths_sum, acc_sum / lengths_sum
    
def main():
    trainds, valds, testds = load_dataset()
    model = BaselineModel().to(device)
    criterion = BaselineLoss().to(device)
    optimizer = optim.Adam(model.parameters())
    #Runner.load_state("../experiments", 1544266417, ("checkpoint", 105), device, model)
    print(validate(model, testds, criterion))
    """for i in range(20):
        sample = model.sample()
        piano_matrix_to_midi(sample, "binary{}.mid".format(i))"""
    runner = Runner(val_fn=validate)
    runner.train(model, trainds, valds, criterion, optimizer, 200)

if __name__ == "__main__":
    main()
