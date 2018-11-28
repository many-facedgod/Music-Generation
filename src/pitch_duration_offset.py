from utils.reading import *

INPUT_FILE = "../bach/aof/*.mid"
encoded_files, _, _, _ = process(INPUT_FILE)
print("encoded_files.shape = ", encoded_files.shape)
for f in encoded_files:
	print(f.shape)

# ADD TRAINING CODE HERE, LIKE IN BASELINE.PY