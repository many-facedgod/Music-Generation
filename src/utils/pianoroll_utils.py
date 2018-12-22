import pypianoroll
import numpy as np

def piano_matrix_to_midi(matrix, filename, base=21, beat_resolution=2):
    length = len(matrix)
    pianoroll = np.zeros((length, 128), dtype=np.int64)
    pianoroll[:, base : base + 88] = matrix[:, :88]
    track = pypianoroll.Track(pianoroll=pianoroll)
    multi_track = pypianoroll.Multitrack(tracks=[track], beat_resolution=beat_resolution)
    pypianoroll.write(multi_track, filename)