import numpy as np
import pptree  # note: requires some minor modifications to pptree source code

class Node():
	def __init__(self, depth=0, name=""):
		self.d = {}
		self.depth = depth
		self.name = "----------- " + name + " [" + str(self.depth) + "]" + " -----------"

	def follow(self, key):
		if key in self.d:
			return self.d[key]
		else:
			return None

	def add(self, key):
		if not key in self.d:
			self.d[key] = Node(depth=self.depth+1, name=str(key))
		return self.d[key]

	def print_string_pretty(self):
		return
		# pptree.print_tree(self, childattr="d")

	def total_nodes(self):
		total = 1
		for child in self.d:
			if self.d[child] is not None:
				total += self.d[child].total_nodes()
		return total