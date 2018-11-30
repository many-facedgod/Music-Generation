from generation import *
from reading import *

import numpy as np


arr = np.array(read_file_as_pitch_offset_duration("../../bach/aof/aria.mid"))
# print(arr[:10])
output_pitch_offset_duration_as_midi_file(arr, "testing.mid")