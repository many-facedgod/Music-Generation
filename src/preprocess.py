import numpy as np 
import music21
from music21 import converter, instrument, note, chord
import glob
import pickle
def get_notes():

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
        #print (s2.show('text'))
        #print (len(s2.parts))
        for element in notes_to_parse:
            #print(element)
            if isinstance(element,note.Note):
                notes.append(str(element.pitch))
                #print(element.pitch,element.octave,element.offset)
            elif isinstance(element,chord.Chord):
                #print("chord",element)
                #print(element.normalOrder)
                notes.append('.'.join (str(n) for n in element.normalOrder) )

        with open('../data/notes', 'wb') as filepath:
                pickle.dump(notes, filepath)
    print (len(notes))
    return notes

get_notes()
    