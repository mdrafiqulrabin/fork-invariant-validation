import os
import sys
import json
import gzip
import time
import re
from copy import deepcopy

import util

class Node():
	def __init__(self, index, label, type=None, subtokens=None):
		self.index = index
		self.label = label
		self.type = type
		if subtokens: self.subtokens = subtokens
		else: self.subtokens = util.split_subtokens(label)
		if not isinstance(self.subtokens, tuple): self.subtokens = (label,)
	
	def __str__(self):
		return self.label + ("" if self.type == None else " (" + self.type + ")") + " @ " + str(self.index) + " " + str(self.subtokens)

class Edge():
	def __init__(self, type, source_ix, target_ix):
		self.type = type
		self.source_ix = source_ix
		self.target_ix = target_ix
	
	def __repr__(self):
		return str(self)
	
	def __str__(self):
		return self.type + ": " + str(self.source_ix) + " -> " + str(self.target_ix)

class Graph():
	def __init__(self, nodes, edges):
		self.nodes = [nodes[ix] for ix in sorted(nodes.keys())]
		self.type_edges = {} # Just a list of edges by type
		self.in_edges = {} # Indexes edges based on their inbound node -- useful for inference or attention
		self.out_edges = {} # Indexes edges based on their outbound nodes -- useful for propagation
		for edge in edges: self.add_edge(edge)
		# Index own variables
		self.variables = self.index_variables()
	
	def index_variables(self):
		if "Child" not in self.type_edges: return None
		vars = {}
		for ix, n in enumerate(self.nodes):
			if not util.is_id(n.label): continue
			if ix not in self.out_edges["Child"] or len(self.out_edges["Child"][ix]) == 0:
				if n.subtokens not in vars:
					vars[n.subtokens] = []
				vars[n.subtokens].append(ix)
			else:
				n.subtokens = [n.label]
		return vars
	
	def size(self):
		return len(self.nodes)
	
	def add_edge(self, edge):
		if edge.source_ix == edge.target_ix: return
		if not edge.type in self.in_edges:
			self.type_edges[edge.type] = []
			self.in_edges[edge.type] = {}
			self.out_edges[edge.type] = {}
		self.type_edges[edge.type].append(edge)
		if not edge.target_ix in self.in_edges[edge.type]:
			self.in_edges[edge.type][edge.target_ix] = []
		if not edge.source_ix in self.out_edges[edge.type]:
			self.out_edges[edge.type][edge.source_ix] = []
		self.in_edges[edge.type][edge.target_ix].append(edge)
		self.out_edges[edge.type][edge.source_ix].append(edge)
	
	def remove_edge(self, edge):
		self.type_edges[edge.type].remove(edge)
		self.in_edges[edge.type][edge.target_ix].remove(edge)
		self.out_edges[edge.type][edge.source_ix].remove(edge)
	
	# Some lookup convenience functions
	def get_edges_for_node(self, node, outbound=True):
		if outbound: return {type: self.out_edges[type][node.index] for type in self.out_edges.keys() if node.index in self.out_edges[type]}
		else: return {type: self.in_edges[type][node.index] for type in self.in_edges.keys() if node.index in self.in_edges[type]}
	
	# AST specific lookup
	def has_children(self, node_ix):
		return "Child" in self.out_edges and node_ix in self.out_edges["Child"]
	
	def get_children(self, node_ix):
		return [] if not self.has_children(node_ix) else [e.target_ix for e in self.out_edges["Child"][node_ix]]
	
	def get_parent(self, node_ix):
		parent_edges = self.out_edges["Parent"]
		return None if node_ix not in parent_edges else parent_edges[node_ix][0].target_ix
	
	# Tree traversal
	def generated_expression(self):
		return " ".join(self.dfs(0))
	
	def dfs(self, node_ix=None):
		# If no node-ix, find the root of the tree
		if node_ix == None:
			node_ix = 0 # Start at arbitrary node (<HOLE> node in this case)
			while True:
				parent_ix = self.get_parent(node_ix)
				if parent_ix == None: return self.dfs(node_ix)
				else: node_ix = parent_ix
		
		children = self.get_children(node_ix)
		if not children or len(children) == 0:
			label = self.nodes[node_ix].label
			if label.endswith(" </s>"): label = label[:-5]
			return [label]
		else:
			res = []
			for n in children: res.extend(self.dfs(n))
			return res
