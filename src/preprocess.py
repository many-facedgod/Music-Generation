import numpy as np 
import music21
from music21 import converter, instrument, note, chord, midi
import glob
import pickle

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

# MIGHT NEED TO TRANSPOSE BASED ON KEY SIGNATURE, FOR BETTER-QUALITY OUTPUT
def read_file(filename):
    processed = []
    score = midi.translate.midiFilePathToStream(filename, quarterLengthDivisors=(32,))
    for elt in score.flat:
        if isinstance(elt, note.Rest) or isinstance(elt, note.Note):  # for now, ignoring chord.Chord, meter.TimeSignature, tempo.MetronomeMark
            pitch = 0 if isinstance(elt, note.Rest) else elt.pitch.midi
            duration = elt.quarterLength
            processed.append((pitch, duration))
    return set(processed)

def get_unique_items(filepaths):
    global_set = set([])
    for filename in glob.glob(filepaths):
        global_set = global_set.union(read_file(filename))
    return sorted(global_set)

def build_dictionary():
    global_set = get_unique_items("../bach/aof/can*.mid")  # defined by pitch and duration, for now
    item_to_index = {item: i for i, item in enumerate(global_set)}
    index_to_item = {i: item for i, item in enumerate(global_set)}
    print(item_to_index)
    print(index_to_item)

build_dictionary()
encode_files()
