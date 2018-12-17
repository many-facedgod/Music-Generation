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
import numpy as np


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


################# Pitch + Duration + Offset #################

# Returns <filename> as a list of (pitch, relative offset, duration) tuples
# Note: Consider transposing based on key signature?
def read_file_as_pitch_offset_duration(filename):
    score = midi.translate.midiFilePathToStream(filename, quarterLengthDivisors=(32,))
    events = score.flat
    processed = []
    print("processing ", filename, "...")
    for i in range(len(events)):  # flat converts relative offsets into absolute offsets!
        elt = events[i]
        if isinstance(elt, chord.Chord):
            offset = elt.offset
            duration = elt.quarterLength
            for note_in_chord in elt:
                pitch = note_in_chord.pitch.midi
                processed.append((pitch, offset, duration))
        if isinstance(elt, note.Rest) or isinstance(elt, note.Note):  # for now, ignoring chord.Chord, meter.TimeSignature, tempo.MetronomeMark
            pitch = 0 if isinstance(elt, note.Rest) else elt.pitch.midi
            offset = elt.offset
            duration = elt.quarterLength
            processed.append((pitch, offset, duration))
    processed.sort(key = lambda x: (x[1], x[0]))
    prev_abs_offset = 0
    for i in range(len(processed)):
        curr_abs_offset = processed[i][1]
        processed[i] = (processed[i][0], curr_abs_offset - prev_abs_offset, processed[i][2])
        prev_abs_offset = curr_abs_offset
    return processed

# Takes in a filename, builds all of the dictionaries we want for encoding and decoding, and encodes the files. Later, possibly split into functions for reuse.
def process(file_regexp):
    # Read in all files matching file_regexp.
    files_list = glob.glob(file_regexp)
    processed_files = []
    error_files=[] # can use this if you want to purge the data sometime
    c=0
    for filename in files_list:
        c=c+1
        try:
            processed = read_file_as_pitch_offset_duration(filename)
            processed_files.append(processed)
        except Exception:
            print('Error in this file:',filename)
            print('File no: ',c)
            error_files.append(filename)
    
    # Build dictionaries (val_to_index, index_to_val) using global_list
    triples = np.array(concat_files(processed_files))
    pitches, offsets, durations = triples[:,0], triples[:,1], triples[:,2]
    (pitch_to_index, index_to_pitch), (offset_to_index, index_to_offset), (duration_to_index, index_to_duration) = build_dictionaries(pitches), build_dictionaries(offsets), build_dictionaries(durations)
    #########
    ###EMERGENCY CODE DELETE LATER
    dictionaries = np.load('../data/piano_testandtrain_encode_dicts.npy')
    pitch_to_index = dictionaries[0]
    offset_to_index = dictionaries[1]
    duration_to_index = dictionaries[2]
    ##########
    # Encode the files
    encoded_files = np.empty(len(processed_files), dtype=object)
    print("encoded_files shape = ", encoded_files.shape)
    for i in range(len(processed_files)):
        file = np.array(processed_files[i])
        encoded_file = np.zeros((len(file),3))
        encoded_file[:,0] = np.array([pitch_to_index[pitch] for pitch in file[:,0]])
        encoded_file[:,1] = np.array([offset_to_index[offset] for offset in file[:,1]])
        encoded_file[:,2] = np.array([duration_to_index[duration] for duration in file[:,2]])
        encoded_files[i] = encoded_file
    #########
    ###EMERGENCY CODE DELETE LATER
    '''filename2 = 'piano_testandtrain_encode'
    filepath2 = '../data/'
    dictionaries = np.array((pitch_to_index,offset_to_index,duration_to_index),dtype=object)
    np.save(filepath2+filename2+'_dicts.npy',dictionaries)'''
    ##########
    return encoded_files, index_to_pitch, index_to_offset, index_to_duration

# Takes in a rank-1 numpy array of values and returns 2 dictionaries - one mapping values to indices ("codes"), and one mapping indices back to values
def build_dictionaries(arr):
    unique_vals = np.unique(arr)
    val_to_index = {item: i for i, item in enumerate(unique_vals)}
    index_to_val = {i: item for i, item in enumerate(unique_vals)}
    return val_to_index, index_to_val


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