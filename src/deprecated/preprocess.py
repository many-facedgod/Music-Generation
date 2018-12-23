import numpy as np 
import music21
from music21 import converter, instrument, note, chord, midi
import glob
import pickle

FILE_REGEXP = "../bach/aof/can*.mid"
SEQ_LEN = 50

def get_notes():
    notes=[]
    chords=[]
    for fname in glob.glob("../bach/aof/*.mid"):  # fname is a string: <class 'str'>
        midi = converter.parse(fname)  # midi is type: <class 'music21.stream.Score'>
        # print("type(midi) = ", type(midi))
        notes_to_parse = None
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            # print("type(s2) = ", type(s2))
            # print("type(s2.parts) = ", type(s2.parts))  # parts is a <class 'music21.stream.iterator.StreamIterator'>, parts[0] is a <class 'music21.stream.Part'>
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        #print (s2.show('text'))
        #print (len(s2.parts))
        # print("type(notes_to_parse) = ", notes_to_parse)
        for element in notes_to_parse:
            #print(element)
            if isinstance(element,note.Note):
                # print("note: ", str(element.pitch))
                notes.append(str(element.pitch))
                #print(element.pitch,element.octave,element.offset)
            elif isinstance(element,chord.Chord):
                #print("chord",element)
                #print(element.normalOrder)
                print("chord: ", element.normalOrder)
                notes.append('.'.join (str(n) for n in element.normalOrder) )
                chords.append('.'.join (str(n) for n in element.normalOrder) )
        # with open('../data/notes', 'wb') as filepath:
        #         pickle.dump(notes, filepath)
    print ("len(notes) = ", len(notes))
    print(chords)
    return notes

def get_notes_mine():
    notes = []
    for filename in glob.glob("../bach/aof/*.mid"):
        midi = converter.parse(filename)
        parts = instrument.partitionByInstrument(midi)
        if parts == None:
            # print("no parts for ", filename)
            # print("instead, midi.flat.notes = ", midi.flat.notes, "\n")
            # for el in midi.flat.notes:
            #     print(type(el), el)
            continue
        else:
            print("parts for ", filename, " = ")
            for part in parts:
                print(part)
                for el in part:
                    print(el)
            return
            print()

def testing():
    midi = converter.parse("../bach/aof/aria.mid")
    # midi.show('text')
    print(len(midi))
    # streams = instrument.partitionByInstrument(midi)
    # output=[]
    # if streams != None:
    #     for part in streams:
    #         print(part)
    #         for item in part:
    #             print(item)
    # else:
    #     print("no separate streams")
    #     notes = midi.flat.notes
    #     for n in notes:
    #         if isinstance(n,note.Note):
    #             output.append(str(n.pitch))
    #         elif isinstance(n,chord.Chord):
    #             output.append('.'.join (str(elt) for elt in n.normalOrder))
    # print(output)


################################

# MIGHT NEED TO TRANSPOSE BASED ON KEY SIGNATURE, FOR BETTER-QUALITY OUTPUT
# input: filename
# output: list of (pitch, duration) tuples
def read_file(filename):
    processed = []
    score = midi.translate.midiFilePathToStream(filename, quarterLengthDivisors=(32,))
    for elt in score.flat:
        if isinstance(elt, note.Rest) or isinstance(elt, note.Note):  # for now, ignoring chord.Chord, meter.TimeSignature, tempo.MetronomeMark
            pitch = 0 if isinstance(elt, note.Rest) else elt.pitch.midi
            duration = elt.quarterLength
            processed.append((pitch, duration))
    return processed

# Returns all unique (pitch, duration) pairs in the FILE_REGEXP set of files
def get_unique_items():
    global_set = set([])
    for filename in glob.glob(FILE_REGEXP):
        global_set = global_set.union(set(read_file(filename)))
    return sorted(global_set)

# Returns 2 dictionaries:
# 1. (pitch, duration) pairs --> indices
# 2. indices --> (pitch, duration) pairs
def build_dictionary():
    global_set = get_unique_items()  # defined by pitch and duration, for now
    item_to_index = {item: i for i, item in enumerate(global_set)}
    index_to_item = {i: item for i, item in enumerate(global_set)}
    return item_to_index, index_to_item

# Uses the built dictionaries to encode all files in the FILE_REGEXP set of files as int-encoded vectors
# Note: nn.Embedding handles int-encoded vectors (doesn't need 1-hot vectors)
def encode_files(item_to_index):
    files = glob.glob(FILE_REGEXP)
    num_files = len(files)
    vocab_size = len(item_to_index)
    print(num_files)
    processed_files = []
    for file in files:
        processed_files.append(read_file(file))
    int_encoded_files = []
    for file in processed_files:
        int_encoded_files.append([item_to_index[item] for item in file])
    return int_encoded_files

def concat_files(files_list):
    base = []
    for file in files_list:
        base.extend(file)
    return base

item_to_index, index_to_item = build_dictionary()
encoded_files = encode_files(item_to_index)  # list of int lists
data = concat_files(encoded_files)
print(len(data))
