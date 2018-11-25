import torch
import torch.nn as nn
from torch.utils.data import DataLoader

INPUT_REGEXP = "../bach/aof/*.mid"
NOTES_FILE = "../data/notes"
OUTPUT_FILE = "../outputs/test_output.mid"

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

	# Train model
	train(model, train_dataloader)

	# Save model


	return data, model

def generation_phase():
	# Load model

	# Generate music! 
	generate_music(model, OUTPUT_FILE)

if __name__ == "__main__":
	data, model = training_phase()
	generation_phase(data, model, OUTPUT_LEN, SEED_LEN)