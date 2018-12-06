from utils.reading import *
import numpy as np
import time

def save_new_data(encode_files, index_to_pitch, index_to_offset, index_to_duration):
	filename = 'music_classical'
	filepath = '../data/'

	np.save(filepath+filename+'_data.npy',encoded_files)
	dictionaries = np.array((index_to_pitch,index_to_offset,index_to_duration),dtype=object)
	np.save(filepath+filename+'_dicts.npy',dictionaries)

start=time.time()

INPUT_FILE = "../datasets/bach/aof/*.mid"
encoded_files, index_to_pitch, index_to_offset, index_to_duration = process(INPUT_FILE)
print("encoded_files.shape = ", encoded_files.shape)
for f in encoded_files:
	print(f.shape)

save_new_data(encode_files, index_to_pitch, index_to_offset, index_to_duration)




end=time.time()
print('Time taken for processing:', end-start)

# ADD TRAINING CODE HERE, LIKE IN BASELINE.PY