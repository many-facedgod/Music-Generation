###############################################################################################
# Utils for generating MIDI files from a trained model.
#
# Example Usage:
# <add here>
#
# Future improvements:
# - replace argmax with sampling
# - adjust the temperature
# - possibly threshold to top k choices
###############################################################################################
from music21 import chord, instrument, note, stream
import numpy as np
import torch

def generate_music(data, int_to_note, model, output_len, seed_len, output_file):
    idx = np.random.randint(len(data) - seed_len)  # selects a random index in the dataset
    seed = np.array(data[idx:idx+10])
    print("seed.shape: ", seed.shape)
    gen_index = generate_notes(model, output_len, seed)
    gen_notes = [int_to_note[i] for i in gen_index]
    print(gen_notes)
    create_midi(gen_notes, output_file)

def create_midi(prediction_output, output_file):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)

def generate_notes(model, output_len, seed):
    #model.test()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    seed = torch.from_numpy(seed).unsqueeze(1)  # L x 1
    seed = seed.to(device).long()
    with torch.no_grad():
        generated_notes = []
        output, hidden = model.forward(seed, None, True)
        out = output[-1]
        _, curr_note = torch.max(out, dim=1)  # 1
        print("seed: ", type(seed), seed)
        print("curr_note: ", type(curr_note), curr_note)
        generated_notes.append(curr_note)

        i = 1
        while(i < output_len):
            curr_note = curr_note.unsqueeze(0)  # L=1 x N=1
            output, hidden = model.forward(curr_note, hidden, True)
            out = output[-1]
            _, curr_note = torch.max(out, dim=1)
            generated_notes.append(curr_note)
            i += 1
        g = torch.cat(generated_notes, dim=0)
        #g = torch.transpose(g,0,1)
        g = g.tolist()
        return g
