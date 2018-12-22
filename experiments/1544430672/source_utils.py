import numpy as np
import torch
from tqdm import tqdm
from os.path import join, isdir
from os import mkdir
import inspect
import sys
import time


def train_batch(model, optimizer, batch, criterion):
    """Trains the model on one batchc

    Args:
        model (nn.Modul): The model to be trained
        optimizer (optim.Optimizer): The optimizer to be used
        batch (ndarray): A batch yielded by MusicDatasetNoConcat
        criterion (nn.Module): The cross entropy criterion

    Returns:
        float: The loss for the batch
    """

    model.train()
    optimizer.zero_grad()
    inp, targets, inv_order = batch
    outputs = model(inp)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

class Runner:

    def __init__(self, train_batch_fn=train_batch, val_fn=None, exp_path="../experiments", run_desc="", log=True):
        self.exp_path = exp_path
        if not run_desc:
            self.run_desc = " ".join(sys.argv[1:])
        else:
            self.run_desc = run_desc
        if not isdir(exp_path):
            mkdir(exp_path)
        self.run_id = str(int(time.time()))
        mkdir(join(exp_path, self.run_id))
        mkdir(join(exp_path, self.run_id, "models"))
        self.log_file = open(join(exp_path, self.run_id, "log.txt"), "w") if log else None
        descriptions_file = open(join(exp_path, "descriptions.txt"), "a")
        descriptions_file.write("{}: {}\n".format(self.run_id, self.run_desc))
        descriptions_file.close()
        self.log("RUN ID: {}".format(self.run_id))
        self.log("RUN DESCRIPTION: {}".format(self.run_desc))
        source_file = open(join(exp_path, self.run_id, "source_main.py"), "w")
        source_file.write(inspect.getsource(sys.modules["__main__"]))
        source_file.close()
        utils_src_file = open(join(exp_path, self.run_id, "source_utils.py"), "w")
        utils_src_file.write(inspect.getsource(sys.modules[__name__]))
        utils_src_file.close()
        self.train_batch_fn = train_batch
        self.val_fn = val_fn
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.log("--------------------------------------------------------")
    
    def log(self, x):
        """Logs a string to the log file and stdout
        """
        tqdm.write(x)
        if self.log_file is not None:
            self.log_file.write(str(x) + "\n")
            self.log_file.flush()

    @staticmethod
    def save_state(exp_path, run_id, tags, model=None, optimizer=None):
        """Saves the model and optimizer states
        
        Args:
            exp_path (str): The experiments path
            run_id (int): The current run ID
            tags (tuple): Tags for the model to be saved
            model (nn.Module, optional): Defaults to None. The model to be saved
            optimizer (optim.Optimizer, optional): Defaults to None. The optimizer to be saved
        """


        name = ".".join([str(i) for i in tags]) + ".pt"
        state = {}
        if model:
            state["model_state"] = model.state_dict()
        if optimizer:
            state["optim_state"] = optimizer.state_dict()
        path = join(exp_path, run_id, "models", name)
        torch.save(state, path)

    @staticmethod
    def load_state(exp_path, run_id, tags, device, model=None, optimizer=None, strict=False):
        """Load a saved model
        
        Args:
            exp_path (str): The experiment path
            run_id (int): The run ID of the model to be loaded
            tags (tuple): Tags for the model to be loaded
            device (torch.device): The device to which the model is to be loaded
            model (nn.Module, optional): Defaults to None. The target model for loadin the state
            optimizer (optim.Optimizer, optional): Defaults to None. The target optimizer for loading the state
            strict (bool, optional): Defaults to False. Strict or lenient loading
        """


        name = ".".join([str(i) for i in tags]) + ".pt"        
        run_id = str(run_id)
        path = join(exp_path, run_id, "models", name)
        state = torch.load(path, map_location=device)
        if model is not None:
            model.load_state_dict(state["model_state"], strict=strict)
        if optimizer is not None:
            optimizer.load_state_dict(state["optim_state"])
        

    def train(self, model, train_dataset, val_dataset, criterion, optimizer, iters, save_every=5, validate_every=1,
            name="checkpoint", preload=None, load_optimizer=False):
        """Trains a model
        
        Args:
            model (nn.Module): The model to be trained
            train_dataset (iterable): A dataset that yields batches
            val_dataset (iterable): A dataset that yields batches
            criterion (nn.Module): The loss function
            optimizer (optim.Optimizer): The optimizer to use
            iters (int): The number of iterations
            save_every (int, optional): Defaults to 5. Save every k iterations
            validate_every (int, optional): Defaults to 1. Validate every k iterations
            name (str, optional): Defaults to "checkpoint". Name of the saved checkpoints
            preload (tuple, optional): Defaults to None. tuple of (run_id, tags) for preloading
            load_optimizer (bool, optional): Defaults to False. If preload is not None, whether to load the optimizer or not
        """

        if preload is not None:
            if load_optimizer:
                self.log("Loading model and optimizer {} from run_id {}...".format(str(preload[1]), preload[0]))
                Runner.load_state(self.exp_path, preload[0], preload[1], self.device, model, optimizer)
            else:
                self.log("Loading model {} from run_id {}...".format(str(preload[1]), preload[0]))
                Runner.load_state(self.exp_path, preload[0], preload[1], self.device, model)
            self.log("Loaded.")
            self.log("--------------------------------------------------------")
        for i in range(1, iters + 1):
            losses = []
            self.log("Iteration {}/{}:".format(i, iters))
            bar = tqdm(train_dataset, desc="Current training loss: NaN", file=sys.stdout)
            for batch in bar:
                loss = self.train_batch_fn(model, optimizer, batch, criterion)
                losses.append(loss)
                bar.set_description("Current training loss: {}".format(loss))
            self.log("Mean loss for the iteration: {}".format(np.mean(losses)))
            self.log("--------------------------------------------------------")
            if not i % save_every:
                self.log("Saving model...")
                Runner.save_state(self.exp_path, self.run_id, (name, i), model, optimizer)
                self.log("Saved model {}".format(str((name, i))))
                self.log("--------------------------------------------------------")
            if not i % validate_every and self.val_fn:
                self.log("Validating model...")
                score = self.val_fn(model, val_dataset, criterion)
                self.log("Validation score: {}".format(score))
                self.log("--------------------------------------------------------")

    
