from utils.reading import *
import numpy as np
import time

start=time.time()

INPUT_FILE = "../lakh_midi/files_200/*.mid"
encoded_files, _, _, _ = process(INPUT_FILE)
print("encoded_files.shape = ", encoded_files.shape)
for f in encoded_files:
	print(f.shape)
np.save('../data/music_lmd.npy',encoded_files)

end=time.time()

print('Time taken for processing:', end-start)

# ADD TRAINING CODE HERE, LIKE IN BASELINE.PY