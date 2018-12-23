from utils.train_utils import *
from utils.pianoroll_utils import *

import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
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

class PianoRollDatasetWholeSong:

    def __init__(self, data, batch_size, ip_pad=0, op_pad=0, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_dataset = PianoRollDatasetBase(data, batch_size, shuffle)
        self.ip_pad = ip_pad
        self.op_pad = op_pad
    
    def __len__(self):
        return len(self.base_dataset)

    def __iter__(self):
        for batch in self.base_dataset:
            yield self.make_batch(batch)
    
    def make_batch(self, batch):
        batch_size = len(batch)
        lengths = np.array([len(i) for i in batch])
        order = np.argsort(lengths)[::-1]
        lengths = lengths[order] + 1
        batch = batch[order]
        max_len = lengths[0]
        ip = np.full((max_len, batch_size, 90), fill_value=self.ip_pad, dtype=np.float32)
        op = np.full((max_len, batch_size, 90), fill_value=self.op_pad, dtype=np.float32)
        for i in range(batch_size):
            for j in range(lengths[i] - 1):
                if len(batch[i][j]) > 0:
                    ip[j + 1, i, batch[i][j] - 21] = 1
                    op[j, i, batch[i][j] - 21] = 1
                else:
                    ip[j + 1, i, 88] = 1
                    op[j, i, 88] = 1
            ip[0, i, 89] = 1
            op[lengths[i] - 1, i, 89] = 1
        inv_order = np.argsort(order)
        ip = torch.FloatTensor(ip).to(device)
        op = torch.FloatTensor(op).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        return (ip, lengths), (op, lengths), inv_order

class PianoRollDatasetChunked:

    def __init__(self, data, batch_size, l_mu1=250, l_mu2=100, l_s1=50, l_s2=25, n1_prob=0.95, shuffle=True):
        self.data = []
        for song in data:
            length = len(song)
            matrix = np.zeros((length, 90), dtype=np.int64)
            for i, point in enumerate(song):
                if len(point) > 0:
                    matrix[i, point - 21] = 1
                else:
                    matrix[i, 88] = 1
            self.data.append(matrix)
        self.data = np.array(self.data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.eos = [0] * 89 + [1]
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
        ip = ip.reshape((self.batch_size, -1, 90)).transpose(1, 0, 2)
        op = op.reshape((self.batch_size, -1, 90)).transpose(1, 0, 2)
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

class TransInvDataset:
    
    def __init__(self, one_hot_dataset, first_pad_op=999):
        self.one_hot_dataset = one_hot_dataset
        self.first_pad_op = first_pad_op
    
    def __len__(self):
        return len(self.one_hot_dataset)
    
    def __iter__(self):
        for batch in self.one_hot_dataset:
            yield self.make_batch(batch)
    
    def make_batch(self, batch):
        (x, lengths), (y, _), inv_order = batch
        x = x.data.cpu().numpy().astype(np.int64)
        max_len = x.shape[0]
        batch_size = x.shape[1]
        lengths = lengths.data.cpu().numpy()
        y = y.data.cpu().numpy().astype(np.int64)
        inp_firsts = np.zeros((max_len, batch_size), dtype=np.int64)
        inp_patterns = np.zeros((max_len, batch_size, 89), dtype=np.float32)
        op_firsts = np.full((max_len, batch_size), fill_value=self.first_pad_op, dtype=np.int64)
        op_patterns = np.zeros((max_len, batch_size, 89), dtype=np.float32)
        for i in range(batch_size):
            length = lengths[i]
            inp = x[:, i, :]
            op = y[:, i, :]
            for j in range(length):
                inp_first = np.nonzero(inp[j])[0][0]
                inp_firsts[j, i] = inp_first
                inp_patterns[j, i, : 89 - inp_first] = inp[j, inp_first + 1:]
                op_first = np.nonzero(op[j])[0][0]
                op_firsts[j, i] = op_first
                op_patterns[j, i, : 89 - op_first] = op[j, op_first + 1:]
        inp_firsts = torch.LongTensor(inp_firsts).to(device)
        op_firsts = torch.LongTensor(op_firsts).to(device)
        inp_patterns = torch.FloatTensor(inp_patterns).to(device)
        op_patterns = torch.FloatTensor(op_patterns).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        return ((inp_firsts, inp_patterns), lengths), ((op_firsts, op_patterns), lengths), inv_order

        

class TransInvModel(nn.Module):

    def __init__(self):
        super(TransInvModel, self).__init__()
        self.first_embedding = nn.Embedding(90, 128)
        self.pattern_embedding  = nn.Sequential(nn.Linear(89, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128))
        self.score_first = nn.Sequential(nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 90))
        self.score_pattern = nn.Sequential(nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 89), nn.Sigmoid())
        self.rnn = nn.LSTM(input_size=256, num_layers=3, hidden_size=256, dropout=0.5)
        self.h0 = nn.Parameter(torch.randn((3, 1, 256)))
        self.c0 = nn.Parameter(torch.randn((3, 1, 256)))
    
    def forward(self, x):
        (firsts, patterns), lengths = x
        batch_size = len(lengths)
        max_len = len(firsts)
        first_embedding = self.first_embedding(firsts)   
        pattern_embedding = self.pattern_embedding(patterns.view(-1, 89)).view(max_len, batch_size, -1)
        inp = torch.cat([first_embedding, pattern_embedding], dim=2)
        packed = rnn_utils.pack_padded_sequence(inp, lengths)
        h0 = torch.cat([self.h0] * batch_size, dim=1)
        c0 = torch.cat([self.c0] * batch_size, dim=1)
        outputs, _ = self.rnn(packed, (h0, c0))
        padded, _ = rnn_utils.pad_packed_sequence(outputs)
        first_scores = self.score_first(padded.view(batch_size * max_len, -1)).view(max_len, batch_size, -1)
        pattern_scores = self.score_pattern(padded.view(batch_size * max_len, -1)).view(max_len, batch_size, -1)
        return (first_scores, pattern_scores), lengths
    
    def binary_sample(self, x):
        return (np.random.random(len(x)) < x.squeeze()).astype(np.int64)[None, None, :]
    
    def categorical_sample(self, x):
        x = torch.FloatTensor(x).squeeze(0)
        sample = dist.Categorical(logits=x).sample().numpy()
        return sample[None, :]
    
    def sample(self, max_len=100, stop_on_eos=False):
        with torch.no_grad():
            inp_first = np.array([[89]])
            inp_pattern = np.zeros((1, 1, 89), dtype=np.float32)
            sequence = []
            h0, c0 = self.h0, self.c0
            for i in range(max_len):
                inp_first = torch.LongTensor(inp_first).to(device)
                inp_pattern = torch.FloatTensor(inp_pattern).to(device)

                first_embedding = self.first_embedding(inp_first)
                pattern_embedding = self.pattern_embedding(inp_pattern.view(-1, 89)).view(1, 1, -1)
                inp = torch.cat([first_embedding, pattern_embedding], dim=2)
                outputs, (h0, c0) = self.rnn(inp, (h0, c0))
                first_scores = self.score_first(outputs.view(1, -1)).view(1, 1, -1)
                pattern_scores = self.score_pattern(outputs.view(1, -1)).view(1, 1, -1)

                inp_first = self.categorical_sample(first_scores.data.cpu().numpy())
                inp_pattern = self.binary_sample(pattern_scores.data.cpu().numpy())

                if inp_first[0, 0] == 89:
                    if stop_on_eos:
                        break
                    else:
                        continue
                else:
                    step = np.zeros(88, dtype=np.float32)
                    if inp_first[0, 0] == 88:
                        #sequence.append(step)
                        continue
                    step[inp_first[0, 0]] = 1
                    step[inp_first[0, 0] + 1 :] = inp_pattern[0, 0, :87 - inp_first[0, 0]]
                    sequence.append(step)
            print(len(sequence))
            return np.array(sequence).astype(np.int64)

class TransInvLoss(nn.Module):

    def __init__(self):
        super(TransInvLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none", ignore_index=999)
    
    def forward(self, x, targets, reduce=True):
        (first_scores, pattern_scores), _ = x
        max_length = first_scores.shape[0]
        batch_size = first_scores.shape[1]
        (first_targs, pattern_targs), lengths = targets
        first_losses = self.ce(first_scores.view(-1, 90), first_targs.view(-1)).view(max_length, batch_size)
        mask = np.zeros((max_length, batch_size, 89), dtype=np.float32)
        for i in range(batch_size):
            length = lengths[i].item()
            for j in range(length):
                mask[j, i, : 89 - first_targs[j, i]] = 1
        mask = torch.FloatTensor(mask).to(device)
        pattern_losses = -(pattern_targs * torch.log(pattern_scores + 1e-28) + (1 - pattern_targs) * torch.log(1 - pattern_scores + 1e-28))
        pattern_losses = (pattern_losses * mask).sum(dim=2)
        losses = first_losses + pattern_losses
        if reduce:
            return losses.sum() / batch_size
        else:
            return losses


def load_dataset(train_batch_size=8, eval_batch_size=8, dataset=3, chunked=True):
    data = pickle.load(open(join(data_path, file_names[dataset]), "rb"))
    if chunked:
        trainbase = PianoRollDatasetChunked(data["train"], train_batch_size)
        valbase = PianoRollDatasetChunked(data["valid"], eval_batch_size, shuffle=False)
        testbase = PianoRollDatasetChunked(data["test"], eval_batch_size, shuffle=False)
    else:
        trainbase = PianoRollDatasetWholeSong(data["train"], train_batch_size)
        valbase = PianoRollDatasetWholeSong(data["valid"], eval_batch_size, shuffle=False)
        testbase = PianoRollDatasetWholeSong(data["test"], eval_batch_size, shuffle=False)
    trainds = TransInvDataset(trainbase)
    valds = TransInvDataset(valbase)
    testds = TransInvDataset(testbase)
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
            """op = (op[0].data.cpu().numpy() > 0.5).astype(np.int64)
            targ = targ[0].data.cpu().numpy().astype(np.int64)
            eq = (op == targ).astype(np.int64)
            max_len = op.shape[0]
            batch_size = op.shape[1]
            ranges = np.arange(max_len).reshape((-1, 1))
            lengths = ip[1].cpu().numpy()
            mask = (ranges < lengths.reshape((1, -1))).reshape((max_len, batch_size, 1))
            eq = eq * mask
            acc_sum += eq.sum()"""
        return loss_sum / lengths_sum#, acc_sum / lengths_sum
    
def main():
    trainds, valds, testds = load_dataset(chunked=True)
    model = TransInvModel().to(device)
    criterion = TransInvLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    Runner.load_state("../experiments", 1544430025, ("checkpoint", 35), device, model)
    print(validate(model, testds, criterion))
    print(validate(model, trainds, criterion))
    """for i in range(20):
        sample = model.sample()
        piano_matrix_to_midi(sample, "transinv{}.mid".format(i))"""
    runner = Runner(val_fn=validate)
    runner.train(model, trainds, valds, criterion, optimizer, 200)

if __name__ == "__main__":
    main()
