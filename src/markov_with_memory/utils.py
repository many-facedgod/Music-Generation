#### Utils ####

# Takes in a regexp representing all files to read.
# Outputs a numpy.ndarray of rank-1 numpy.ndarrays.
#
# OUTPUT format:
# Outer array length is # of songs. Inner array lengths are song lengths.
# Each song is a numpy.ndarray of tuples (1-tuples for notes and k-tuples for chords).
def process_melodies_as_offsets(filepath = FILEPATH):
    # Identify which songs are valid (= 2 parts, without any chords in first part)... adjust size based on that
    original_files = list(glob.iglob(filepath, recursive = True))
    valid_files = []
    for file in original_files:
        success, _ = read_melody_as_pitch_offset_duration(file)
        if success:
            valid_files.append(file)

    # Process songs
    num_files = len(valid_files)
    songs_original, songs_as_pitches, songs_as_offsets = np.empty(num_files, dtype=object), np.empty(num_files, dtype=object), np.empty(num_files, dtype=object)
    first_notes = np.empty(num_files)
    for i in range(num_files):
        # First, read in each song as np-array of (pitch, offset, duration) tuples.
        _, melody = read_melody_as_pitch_offset_duration(valid_files[i])  # only added successfully-parsed files to valid_files
        songs_original[i] = np.array(melody)
        # Then, convert the song into pitch-only-tuples (1-tuple for note, k-tuple for chord).
        # We call _concat_chords to compress all notes with the same offset AND duration into chords.
        songs_as_pitches[i] = np.array([tuple(p) for p in _concat_chords(songs_original[i])]).flatten()  # processing as tuples for consistency but expecting all ints! should crash below if not ints
        song = songs_as_pitches[i]
        offsets = np.empty(song.shape[0]-1)
        for j in range(len(song)):
            if j == 0:
                first_notes[i] = song[j]
            else:
                offsets[j-1] = song[j] - song[j-1]
        songs_as_offsets[i] = np.array(offsets)
    print(songs_as_pitches[0])
    return first_notes, songs_as_offsets

def _concat_chords(arr):
    input_len = arr.shape[0]
    pitch, offset, duration = arr[:,0], arr[:,1], arr[:,2]
    pitch = pitch.astype(np.int32).tolist()  # must convert to native int type (see final case in Chord's _add_core_or_init)

    output = []
    # key idea: for every offset, maintain a dict that maps: duration --> list of pitches
    # each element of the map is a chord (or note, if only 1)
    i = 0
    output_notes = []
    while i < input_len:
        dd = defaultdict(list)
        if pitch[i] != 0:
            dd[duration[i]].append(pitch[i])
        while i+1 < input_len and offset[i+1] == 0:  # increment i and add next values to the map
            i += 1
            if pitch[i] != 0:
                dd[duration[i]].append(pitch[i])
        for pitches in dd.values():
            output.append(pitches)
        i += 1

    return output