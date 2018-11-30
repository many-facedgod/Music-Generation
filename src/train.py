#####################################################################
# Do not run on CPU. PyTorch has a bug where the ignore_index       #
# parameter doesn't work with CPU if the ignored index is not       #
# smaller than the size of the vocabulary.                          #
#####################################################################
import numpy as np
import gzip
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from os.path import join, isdir
from os import mkdir
import inspect
import sys
import time
import model1

data_path = "../data"
exp_path = "../experiments"
run_desc = "Baby steps"
run_id = None
log_file = None

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def load_dictionaries():
    """Loads the dictionaries and returns them along with the vocab_sizes

    Returns:
        list of int: the vocab sizes of pitch, offset, duration in that order
        dictionary: index_to_pitch
        dictionary: index_to_offset
        dictionary: index_to_duration
    """
    dictionaries = np.load(join(data_path,music_file+'_dicts.npy'))
    index_to_pitch = dictionaries[0]
    index_to_offset = dictionaries[1]
    index_to_duration = dictionaries[2]
    vocab_sizes = [len(index_to_pitch),len(index_to_offset),len(index_to_duration)]
    return vocab_sizes, index_to_pitch, index_to_offset, index_to_duration	

music_file = "music_lmd"
#TODO: For use in Decoder
vocab_sizes,_,_,_ = load_dictionaries()  # not including eos
eos_indices = vocab_sizes.copy()  # EOS and SOS have the same index
ignored_index = 999  # for the cross-entropy loss


class MusicDatasetNoConcat:
    """Doesn't do much, just yields the data in batches
    """

    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(indices)
        n_batches = (len(self.data) - 1) // self.batch_size + 1
        for i in range(n_batches):
            batch = self.data[indices[i * self.batch_size: (i + 1) * self.batch_size]]
            yield batch

    def __len__(self):
        return (len(self.data) - 1) // self.batch_size + 1

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


def make_batch(batch, inp_pad=0, op_pad=ignored_index):
    """Does the legwork for making the batches ready for training

    Args:
        batch: A batch yielded by the MusicDatasetNoConcat
        inp_pad (int, optional): Defaults to 0. The value to pad the model input with
        op_pad (int, optional): Defaults to ignored_index. The value to pad the model output with
    """

    lengths = np.array([len(x) for x in batch]) + 1  # adding EOS/SOS
    tuple_shape = batch[0].shape[1]
    batch_size = len(lengths)
    max_len = max(lengths)
    order = np.argsort(lengths)[::-1]
    batch = batch[order]
    lengths = lengths[order]
    padded_batch_ip = np.full((max_len, batch_size, tuple_shape), fill_value=inp_pad, dtype=np.int64)
    padded_batch_ip[0, :, :] = eos_indices
    padded_batch_op = np.full((max_len, batch_size, tuple_shape), fill_value=op_pad, dtype=np.int64)
    for i, x in enumerate(batch):
        padded_batch_ip[1:lengths[i], i, :] = x
        padded_batch_op[:lengths[i] - 1, i, :] = x
        padded_batch_op[lengths[i] - 1, i, :] = eos_indices
    inv_order = np.argsort(order)
    padded_batch_op = torch.LongTensor(padded_batch_op).to(device)
    padded_batch_ip = torch.LongTensor(padded_batch_ip).to(device)
    lengths = torch.LongTensor(lengths).to(device)
    return (padded_batch_ip, lengths), (padded_batch_op, lengths), inv_order


class CELoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0], vocab_sizes=vocab_sizes):
        """Handles the loss for all tuple elements

        Args:
            weights (list, optional): Defaults to [1.0, 1.0, 1.0]. Weights for the three elements of the tuple
            vocab_sizes (list, optional): Defaults to vocab_sizes. Vocab sizes (without eos)
        """
        super(CELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none", ignore_index=ignored_index)
        self.weights = weights

    def forward(self, logits, targets, reduce=True):
        """Calculates the loss given logits and targets

        Args:
            logits (tuple of tensors): Logits for each element of the tuple
            targets (Tensor): Expected
            reduce (bool): Defaults to True. Whether to reduce the loss or not

        Returns:
            torch scalar: The calculated loss
        """
        logits = logits[0]
        targets = targets[0]
        loss = 0.0
        batch_size = targets.shape[1]
        for i in range(len(logits)):
            loss = loss + self.weights[i] * self.ce(logits[i].view(-1, vocab_sizes[i] + 1), targets[:, :, i].view(-1))
        if reduce:
            return loss.sum() / batch_size
        else:
            return loss.view(targets.shape[0], targets.shape[1])


def validate_batch_perplexity(model, batch, make_batch_fn, criterion=CELoss()):
    """Calculates the perplexity for one batch

    Args:
        model (nn.Module): The model to be validated
        batch (ndarray): The batch yielded by the dataset
        make_batch_fn (function): Function that generates model inputs/targets
        criterion (nn.Module): Defaults to CELoss(). The cross entropy criterion

    Returns:
        float: The calculated perplexity
    """

    model.eval()
    batch_size = len(batch)
    inp, targets, inv_order = make_batch_fn(batch)
    with torch.no_grad():
        outputs = model(inp)
        loss = criterion(outputs, targets, reduce=False).sum(dim=0) / inp[1].float()
        return torch.exp(loss).mean().item()


def train_batch(model, optimizer, batch, make_batch_fn, criterion=CELoss()):
    """Trains the model on one batchc

    Args:
        model (nn.Modul): The model to be trained
        optimizer (optim.Optimizer): The optimizer to be used
        batch (ndarray): A batch yielded by MusicDatasetNoConcat
        make_batch_fn (function): Function that generates model inputs/targets
        criterion (nn.Module, optional): Defaults to CELoss(). The cross entropy criterion

    Returns:
        float: The loss for the batch
    """

    model.train()
    optimizer.zero_grad()
    inp, targets, inv_order = make_batch_fn(batch)
    outputs = model(inp)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def log(x):
    """Logs a string to the log file and stdout
    """
    tqdm.write(x)
    log_file.write(str(x) + "\n")
    log_file.flush()


def init_run():
    """Initialies a run
    """

    global log_file, run_id
    if not isdir(exp_path):
        mkdir(exp_path)
    run_id = str(int(time.time()))
    mkdir(join(exp_path, run_id))
    mkdir(join(exp_path, run_id, "models"))
    log_file = open(join(exp_path, run_id, "log.txt"), "w")
    descriptions_file = open(join(exp_path, "descriptions.txt"), "a")
    descriptions_file.write("{}: {}\n".format(run_id, run_desc))
    descriptions_file.close()
    log("RUN ID: {}".format(run_id))
    log("RUN DESCRIPTION: {}".format(run_desc))
    source_file = open(join(exp_path, run_id, "source.py"), "w")
    source_file.write(inspect.getsource(sys.modules[__name__]))
    source_file.close()
    log("--------------------------------------------------------")


def load_dataset(train_batch_size=3, eval_batch_size=3, train_ratio=0.8):
    """Loads the dataset

    Args:
        train_batch_size (int, optional): Defaults to 40. Batch size for training
        eval_batch_size (int, optional): Defaults to 100. Batch size for validation
        train_ratio (float, optional): Defaults to 0.8. Ratio for the train-test split

    Returns:
        tuple of MusicDatasetNoConcat: The train and validation datasets
    """

    data = np.load(join(data_path, music_file+'_data.npy'))
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]
    return MusicDatasetNoConcat(train_data, train_batch_size), MusicDatasetNoConcat(val_data, eval_batch_size)


def load_dictionaries():
    """Loads the dictionaries

    Returns:
        list of int: the vocab sizes of pitch, offset, duration in that order
        int_to
    """

    dictionaries = np.load(data_path+music_file+'_dicts.npy')
    index_to_pitch = dictionaries[0]
    index_to_offset = dictionaries[1]
    index_to_duration = dictionaries[2]
    vocab_sizes = [len(index_to_pitch),len(index_to_offset),len(index_to_duration)]
    return vocab_sizes, index_to_pitch, index_to_offset, index_to_duration	

def save_state(tags, model=None, optimizer=None):
    """Saves the model and the optimizer state

    Args:
        tags (tuple): tags for the saved model
        model (nn.Module, optional): Defaults to None. The model to be saved
        optimizer (optim.Optimizer, optional): Defaults to None. The optimizer to be saved
    """

    log("Saving model...")
    name = ".".join([str(i) for i in tags]) + ".pt"
    state = {}
    if model:
        state["model_state"] = model.state_dict()
    if optimizer:
        state["optim_state"] = optimizer.state_dict()
    path = join(exp_path, run_id, "models", name)
    torch.save(state, path)
    log("Saved to {}".format(path))
    log("--------------------------------------------------------")


def load_state(run_id, tags, model=None, optimizer=None, strict=False):
    """Loads the model from a particular run ID

    Args:
        run_id (int): The ID of the run to load the model from
        tags (tuple): The tags identifying the model
        model (nn.Module, optional): Defaults to None. The target model for loading the state
        optimizer (optim.Optimizer, optional): Defaults to None. The target optimizer for loading the state
        strict (bool, optional): Defaults to False. Loading type
    """

    name = ".".join([str(i) for i in tags]) + ".pt"
    log("Loading model {} from run_id {}...".format(name, run_id))
    run_id = str(run_id)
    path = join(exp_path, run_id, "models", name)
    state = torch.load(path)
    if model is not None:
        model.load_state_dict(state["model_state"], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(state["optim_state"])
    log("Loaded.")
    log("--------------------------------------------------------")


def validate(model, val_dataset, criterion=CELoss(), make_batch_fn=make_batch, val_batch_fn=validate_batch_perplexity):
    """Validate the model
    
    Args:
        model (nn.Module): The model to be validated
        val_dataset (MusicDatasetNoConcat): The validation dataset
        criterion (nn.Module, optional): Defaults to CELoss(). The criterion to be used
        make_batch_fn (function, optional): Defaults to make_batch. The function to make a batch from the dataset yield
        val_batch_fn (function, optional): Defaults to validate_batch_perplexity. The function for validating a batch
    
    Returns:
        float: Mean validation score across batches
    """

    log("Validating model...")
    scores = []
    bar = tqdm(val_dataset, desc="Current validation score: NaN", file=sys.stdout)
    for batch in bar:
        score = val_batch_fn(model, batch, make_batch_fn, criterion)
        scores.append(score)
        bar.set_description("Current validation score: {}".format(score))
    log("Mean validation score: {}".format(np.mean(scores)))
    log("--------------------------------------------------------")
    return np.mean(scores)


def train(model, train_dataset, val_dataset, criterion, optimizer, iters, train_batch_fn=train_batch,
          val_batch_fn=validate_batch_perplexity, make_batch_fn=make_batch, save_every=20, validate_every=1,
          name="checkpoint"):
    """Trains the model
          
    Args:
        model (nn.Module): The model to be trained
        train_dataset (MusicDatasetNoConcat): The dataset for training
        val_dataset (MusicDatasetNoConcat): The dataset for validation
        criterion (nn.Module): The loss function
        optimizer (optim.Optimizer): The optimizer to be used
        iters (int): The number of iterations
        train_batch_fn (function, optional): Defaults to train_batch. Function for training a batch
        val_batch_fn (function, optional): Defaults to validate_batch_perplexity. Function for validating a batch
        make_batch_fn (function, optional): Defaults to make_batch. Function to make a batch from the dataset
        save_every (int, optional): Defaults to 20. Saves the model every save_every iters
        validate_every (int, optional): Defaults to 1. Validates the model every validate_every iters
        name (str, optional): Defaults to "checkpoint". Name for the saved checkpoints
    """

    for i in range(1, iters + 1):
        losses = []
        log("Iteration {}/{}:".format(i, iters))
        bar = tqdm(train_dataset, desc="Current training loss: NaN", file=sys.stdout)
        for batch in bar:
            loss = train_batch_fn(model, optimizer, batch, make_batch_fn, criterion)
            losses.append(loss)
            bar.set_description("Current training loss: {}".format(loss))
        log("Mean loss for the iteration: {}".format(np.mean(losses)))
        log("--------------------------------------------------------")
        if not i % save_every:
            save_state((name, i), model, optimizer)
        if not i % validate_every and val_batch_fn:
            validate(model, val_dataset, criterion, make_batch_fn, val_batch_fn)


def main():
    init_run()
    model = model1.Baseline(3, vocab_sizes).to(device)
    optimizer = optim.Adam(model.parameters())
    train_ds, val_ds = load_dataset()
    train(model, train_ds, val_ds, CELoss(), optimizer, 200)


if __name__ == "__main__":
    main()
