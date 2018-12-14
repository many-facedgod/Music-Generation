##### Dictionary-based implementation of a kth-order Markov chain with an external autoassociative memory #####

from utils import *
import numpy as np
import time
import trie

DATASET = "nottingham"
TRAIN_FILEPATH = "../../datasets/" + DATASET + "/train/*.mid"
VAL_FILEPATH = "../../datasets/" + DATASET + "/valid/*.mid"
TEST_FILEPATH = "../../datasets/" + DATASET + "/test/*.mid"

def _print_nodes(nodes_list, message=""):
	print(message)
	for n in nodes_list:
		if n is not None:
			n.print_string_pretty()
		else:
			print("None")
	print()

class MarkovChain():
	def __init__(self, k = 1, frac = 1, alpha = 1):
		self.k = k
		self.i = {}
		self.t = {}
		self.num_states = 0
		self.int_to_pitch = {}
		self.pitch_to_int = {}
		self.alpha = alpha
		self.frac = frac

	def _encode_songs(self, songs_orig_rep):
		songs = np.empty(len(songs_orig_rep), dtype=object)
		for i in range(len(songs_orig_rep)):
			print("type: ", type(songs_orig_rep[i]), "\n", "song: ", songs_orig_rep[i], "\n\n")
			song = np.array([self.pitch_to_int[p] if p in self.pitch_to_int else len(self.pitch_to_int) for p in songs_orig_rep[i]])  # creates dummy value if pitch wasn't seen during training
			songs[i] = song
		return songs

	def _decode_songs(songs_as_ints):
		return NotImplemented

	def _evaluate(self, songs, depth = 10):
		total_ll = 0
		total_length = 0

		for song in songs:
			############### processing first k notes ###############
			print("song: ", song, "\n")
			# initialize trie over first k notes
			nodes_to_update = np.empty(depth, dtype=object)
			root = trie.Node(name="root")
			for i in range(self.k):
				curr_offset = song[i]
				matches = []

				###################### step all matches forward using curr_offset
				# update our matches using this note!
				_print_nodes(matches, "matching paths *before* processing index {} (curr_offset = {}):".format(i, curr_offset))
				old_matches = matches
				old_matches.append(root)  # root is always a match... we can always start over at the curr note!
				matches = []
				for match in old_matches:
					next = match.follow(curr_offset)
					if next is not None:
						matches.append(next)
				_print_nodes(matches, "matching paths *after* processing index {} (curr_offset = {}):".format(i, curr_offset))
				###################### end of match updates
				# updating the trie comes AFTER updating matches, so that we don't follow a path in the trie that we *just* created using curr_offset (we need a path with curr_offset to already exist)
				idx_to_replace = i % depth
				print("idx_to_replace = ", idx_to_replace)
				nodes_to_update[idx_to_replace] = root  # replace just one at a time
				_print_nodes(nodes_to_update, "state of 'nodes_to_update' after replacing the node that had already reached max-depth (couldn't be updated anymore... preparing for update)")
				for j in range(depth):
					if nodes_to_update[j] is not None:  # update all of the nodes we're tracking that aren't None
						nodes_to_update[j] = nodes_to_update[j].add(curr_offset)
				_print_nodes(nodes_to_update, "end of iteration {} (just saw {} --> nodes_to_update is now".format(i, curr_offset))
				print("\n\n")

			print("dump of trie after examining first k notes: \n")
			root.print_string_pretty()

			# add LL for first k notes (assume the first k notes are effectively seeded, at least wrt the trie)
			key = tuple(song[:self.k])
			count = 0 if not key in self.i else self.i[key]
			count += self.alpha  # add pseudocount = laplace smoothing
			sum_ = self.i['sum_'] + self.alpha * self.num_states**self.k  # moving forward, consider possible overflow for large k
			prob = count / sum_  # float division in python3
			total_ll += np.log(prob)
			############### end of first k notes processing ###############

			############### start of main processing (k+1 onward) ###############
			_print_nodes(nodes_to_update, "state of 'nodes_to_update' after examining first k notes: ")

			for i in range(self.k, len(song)):  # Iterate over song
				curr_offset = song[i]  # to be used throughout, below
				print("---------- Starting index {} (curr_offset = {}) ----------".format(i, curr_offset))
				print("Overall trie is now:")
				root.print_string_pretty()

				########################################
				# calculate overall probability of curr_offset using either: (1) trie or (2) transition table
				max_depth = 0
				best_node = None
				_print_nodes(matches, "...and matches is")
				for match in matches:
					if match.depth > max_depth:
						max_depth = match.depth
						best_node = match
				print("best match for sequence of notes immediately prior to curr_offset (other than those notes themselves!!!) is...")
				if best_node is None:
					print("None :(")
				else:
					best_node.print_string_pretty()
				prob_uses_trie = 0
				prob_given_trie = 0
				prob_given_matrix = 0
				if best_node is not None:
					curr_match_depth = best_node.depth
					if len(best_node.d) > 0:  # has at least one next path, o/w can't follow trie
						prob_uses_trie = 1 - self.frac**curr_match_depth
						prob_given_trie = 0 if not curr_offset in best_node.d else 1 / len(best_node.d)  # uniform random over all children
					else:
						print("best match has no next node... using transition matrix instead")

				# calculate probability of using transition matrix and choosing curr_offset with matrix
				key1 = tuple(song[i-self.k:i])  # k preceding ints
				key2 = song[i]
				if not key1 in self.t:
					count, sum_ = 0, 0
				elif not key2 in self.t[key1]:
					count, sum_ = 0, self.t[key1]['sum_']
				else:
					count = self.t[key1][key2]
					sum_ = self.t[key1]['sum_']
				count += self.alpha
				sum_ += self.alpha * self.num_states
				prob_given_matrix = count / sum_  # float division in python3

				print("prob_uses_trie: ", prob_uses_trie)
				print("prob_given_trie: ", prob_given_trie)
				print("prob_given_matrix: ", prob_given_matrix)
				prob_of_curr_offset = prob_uses_trie * prob_given_trie + (1-prob_uses_trie) * prob_given_matrix
				print("prob_of_curr_offset: ", prob_of_curr_offset)
				total_ll += np.log(prob_of_curr_offset)
				########################################

				###################### step all matches forward using curr_offset
				# update our matches using this note!
				_print_nodes(matches, "matching paths *before* processing index {} (curr_offset = {}):".format(i, curr_offset))
				old_matches = matches
				old_matches.append(root)  # root is always a match... we can always start over at the curr note!
				matches = []
				for match in old_matches:
					next = match.follow(curr_offset)
					if next is not None:
						matches.append(next)
				_print_nodes(matches, "matching paths *after* processing index {} (curr_offset = {}):".format(i, curr_offset))
				###################### end of match updates
				###################### then update the trie (even if we're using the trie... it shouldn't harm anything) - we maintain that invariant that ALL n-grams are stored in the trie (n = depth?)
				# updating the trie comes AFTER updating matches, so that we don't follow a path in the trie that we *just* created using curr_offset (we need a path with curr_offset to already exist)
				idx_to_replace = i % depth
				print("idx_to_replace = ", idx_to_replace)
				nodes_to_update[idx_to_replace] = root  # replace just one at a time
				_print_nodes(nodes_to_update, "state of 'nodes_to_update' after replacing the node that had already reached max-depth (couldn't be updated anymore... preparing for update)")
				for j in range(depth):
					if nodes_to_update[j] is not None:  # update all of the nodes we're tracking that aren't None
						nodes_to_update[j] = nodes_to_update[j].add(curr_offset)
				_print_nodes(nodes_to_update, "state of 'nodes_to_update' after updating all nodes with curr_offset")
				###################### end of trie update

			total_length += len(song)
		return total_ll / total_length

	# returns avg NLL per time step
	def train(self, songs):
		songs_concatenated = np.hstack(songs)  # makes one big song of pitch-tuples
		unique_pitches = set(songs_concatenated)
		self.num_states = len(unique_pitches)
		for song in songs:
			key = tuple(song[:self.k])  # first k (for order-k chain)
			if not key in self.i:
				self.i[key] = 0
			self.i[key] += 1
			for i in range(self.k, len(song)):
				key1 = tuple(song[i-self.k:i])  # k preceding ints
				key2 = song[i]
				if not key1 in self.t:
					self.t[key1] = {}  # self.t[key1] is now at least {}
				if not key2 in self.t[key1]:
					self.t[key1][key2] = 0
				self.t[key1][key2] += 1
		# done building self.i and self.t!

		# add sum_ keys to prevent iteration over each dict more than once
		sum_ = 0
		for key in self.i:
			sum_ += self.i[key]
		self.i['sum_'] = sum_  # every valid self.i has a 'sum_' key

		for key1 in self.t:
			sum_ = 0
			for key2 in self.t[key1]:
				sum_ += self.t[key1][key2]
			self.t[key1]['sum_'] = sum_
		# done adding sum_s to self.i and self.t!

		# compute avg NLL per time step over training set w/ newly-trained dicts!
		avg_ll = self._evaluate(songs)
		return avg_ll

	def test(self, songs):
		avg_ll = self._evaluate(songs)
		return avg_ll

for dataset in ["nottingham"]:  # ["nottingham", "jsb_chorales", "piano_midi", "musedata"]
	TRAIN_FILEPATH = "../../datasets/" + dataset + "/train/*_1.mid"
	TEST_FILEPATH = "../../datasets/" + dataset + "/test/ashover*.mid"

	train_songs_orig_rep = process_melodies_as_offsets(TRAIN_FILEPATH)  # np-array of np-arrays of [whatever the rep is - pitch-tuples, offsets, etc.]
	train_first_notes, train_songs_as_offsets = train_songs_orig_rep[0], train_songs_orig_rep[1]

	test_songs_orig_rep = process_melodies_as_offsets(TEST_FILEPATH)  # np-array of np-arrays of [whatever the rep is - pitch-tuples, offsets, etc.]
	test_first_notes, test_songs_as_offsets = test_songs_orig_rep[0], test_songs_orig_rep[1]

	f = open("avg_ll_with_dict_memory_v0.txt", "w")
	for k in range(1,21):
		for frac in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
			for alpha in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
				print("k: ", k)
				mc = MarkovChain(k, frac, alpha)
				start_train = time.time()
				train_avg_ll = mc.train(train_songs_as_offsets)
				end_train = time.time()
				start_test = time.time()
				test_avg_ll = mc.test(test_songs_as_offsets)
				end_test = time.time()
				print("train_avg_ll: ", train_avg_ll)
				print("test_avg_ll: ", test_avg_ll)
				f.write(dataset + " order " + str(k) + ", frac " + str(frac) + ", alpha " + str(alpha) + ": train avg_ll: " + str(train_avg_ll) + "\tin time " + str(end_train - start_train) + "\n")
				f.write(dataset + " order " + str(k) + ", frac " + str(frac) + ", alpha " + str(alpha) + ": test avg_ll: " + str(test_avg_ll) + "\tin time " + str(end_test - start_test) + "\n")
				f.write(dataset + " order " + str(k) + ", frac " + str(frac) + ", alpha " + str(alpha) + ": len of initials dict = " + str(len(mc.i)) + "\n")
				f.write(dataset + " order " + str(k) + ", frac " + str(frac) + ", alpha " + str(alpha) + ": len of transitions dict = " + str(len(mc.t)) + "\n")
				total = 0
				for j in mc.t:
					total += len(mc.t[j])
				f.write(dataset + " order " + str(k) + ", frac " + str(frac) + ", alpha " + str(alpha) + ": total entries in transitions dict = " + str(total) + "\n\n")
f.close()