from train import *
from utils.generation import *

# For now, before we do further refactoring, just change the log() statements
# in load_state() to print() statements before running.
#
# Roughly follows main() in src/train.py; the primary difference
# is that we use the dictionaries to complete decoding and then call
# output_pitch_offset_duration_as_midi_file() to write to a MIDI file.
def main():
    vocab_sizes, index_to_pitch, index_to_offset, index_to_duration = load_dictionaries()
    model = model1.Baseline(3, vocab_sizes).to(device)
    optimizer = optim.Adam(model.parameters())
    load_state('1543553172', ('checkpoint',40),model,optimizer)
    generated_outputs = model.decode()
    decoded_outputs = np.zeros(generated_outputs.shape)
    for i in range(len(generated_outputs)):
    	event = generated_outputs[i]
    	decoded_outputs[i] = [index_to_pitch[event[0]], index_to_offset[event[1]], index_to_duration[event[2]]]
    output_pitch_offset_duration_as_midi_file(decoded_outputs, "../outputs/generation_v0.mid")

if __name__ == "__main__":
    main()