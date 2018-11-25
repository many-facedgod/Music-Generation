###############################################################################################
# Utils for reading in MIDI files and converting into various formats for model training.
#
# Example Usage:
# item_to_index, index_to_item = build_dictionary()
# encoded_files = encode_files(item_to_index)  # list of int lists
# data = concat_files(encoded_files)
###############################################################################################
import glob
import pickle
from music21 import chord, converter, instrument, midi, note


################# Just Pitch #################
# Baseline version of reading in MIDI files.
# Note: Only handles pitch.

# Read in all MIDI files matching file_regexp and write output to notes_file.
def midi_to_notes_file(input_regexp, notes_file):
    notes=[]
    for fname in glob.glob("../bach/aof/*.mid"):
        print("processing")
        midi = converter.parse(fname)
        notes_to_parse = None
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element,note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element,chord.Chord):
                notes.append('.'.join (str(n) for n in element.normalOrder) )
        with open('../data/notes', 'wb') as filepath:
                pickle.dump(notes, filepath)
    print (len(notes))
    return notes

# Read in notes_file and build dataset (plus dicts to map to and from int encoding) for training.
def notes_file_to_input_data(notes_file):
    notes = np.load(notes_file, encoding = 'bytes')  # list
    pitchnames = sorted(set(items for items in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    print(note_to_int)
    print("Vocab size : ",len(note_to_int))
    data = [ note_to_int[note] for note in notes]
    print("total number of notes in dataset:", len(data))
    return data, note_to_int, int_to_note


################# Pitch + Duration #################

# Returns <filename> as a list of (pitch, duration) tuples
# Note: Consider transposing based on key signature?
def read_file_as_pitch_duration(filename):
    processed = []
    score = midi.translate.midiFilePathToStream(filename, quarterLengthDivisors=(32,))
    for elt in score.flat:
        if isinstance(elt, note.Rest) or isinstance(elt, note.Note):  # for now, ignoring chord.Chord, meter.TimeSignature, tempo.MetronomeMark
            pitch = 0 if isinstance(elt, note.Rest) else elt.pitch.midi
            duration = elt.quarterLength
            processed.append((pitch, duration))
    return processed


################# Can be made general later #################

# Returns all unique (pitch, duration) pairs in the files matching file_regexp
def get_unique_items(file_regexp):
    global_set = set([])
    for filename in glob.glob(file_regexp):
        global_set = global_set.union(set(read_file_as_pitch_duration(filename)))
    return sorted(global_set)

# Returns 2 dictionaries:
# 1. (pitch, duration) pairs --> indices
# 2. indices --> (pitch, duration) pairs
# based on the files matching file_regexp
def build_dictionary(file_regexp):
    global_set = get_unique_items(file_regexp)  # defined by pitch and duration, for now
    item_to_index = {item: i for i, item in enumerate(global_set)}
    index_to_item = {i: item for i, item in enumerate(global_set)}
    return item_to_index, index_to_item

# Uses the built dictionaries to encode all files matching file_regexp as int-encoded vectors
# Note: nn.Embedding handles int-encoded vectors (doesn't need 1-hot vectors)
def encode_files(item_to_index, file_regexp):
    files = glob.glob(file_regexp)
    num_files = len(files)
    vocab_size = len(item_to_index)
    print(num_files)
    processed_files = []
    for file in files:
        processed_files.append(read_file_as_pitch_duration(file))
    int_encoded_files = []
    for file in processed_files:
        int_encoded_files.append([item_to_index[item] for item in file])
    return int_encoded_files

# Concates a list of processed files into one dataset.
# Note: When training on a such a dataset, original boundaries between
# files are ignored. This is probably not ideal, but it's reasonable for now.
def concat_files(files_list):
    base = []
    for file in files_list:
        base.extend(file)
    return base