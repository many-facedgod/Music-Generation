import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.generation import *
from utils.reading import *
from utils.training import *
# ^ load these together moving forward

# INPUT_REGEXP = "../bach/aof/*.mid"
INPUT_REGEXP = "../bach/aof/can1.mid"
NOTES_FILE = "../data/notes"
OUTPUT_FILE = "../outputs/test_baseline_v0.mid"

NUM_EPOCHS = 20
BATCH_SIZE = 80
WEIGHT_DECAY = 1e-6

OUTPUT_LEN = 400
SEED_LEN = 10

def training_phase():
    # Read in training data (including encoding dicts)
    midi_to_notes_file(INPUT_REGEXP, NOTES_FILE)
    data, note_to_int, int_to_note = notes_file_to_input_data(NOTES_FILE)
    vocab_size = len(note_to_int)
    train_dataset = NotesDataset(data)

    # Do we actually need collate????????????????????????
    def collate(seq_list):
        inputs = torch.cat([s[0].unsqueeze(1) for s in seq_list],dim=1)
        targets = torch.cat([s[1].unsqueeze(1) for s in seq_list],dim=1)
        return inputs,targets

    # Training setup
    model = MusicModel(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    print("CUDA IS AVAILABLE: ", "YES" if torch.cuda.is_available() else "NO")

    # Train model
    train(model, optimizer, criterion, train_dataloader, num_epochs=NUM_EPOCHS, device="cuda" if torch.cuda.is_available() else "cpu")

    # Save model
    torch.save(model.state_dict(), "models/baseline_model.pt")

    return vocab_size, data, int_to_note, model

def generation_phase(vocab_size, data, int_to_note, model):
    # Load model
    model = MusicModel(vocab_size)
    model.load_state_dict(torch.load("models/baseline_model.pt"))

    # Generate music! 
    generate_music(data, int_to_note, model, OUTPUT_LEN, SEED_LEN, OUTPUT_FILE)

if __name__ == "__main__":
    vocab_size, data, int_to_note, model = training_phase()
    generation_phase(vocab_size, data, int_to_note, model)
