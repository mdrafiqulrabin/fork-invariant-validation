import os
import json
import gzip
import random
random.seed(42)

import tensorflow as tf
from multiprocessing import Pool

from graph import *

class GraphMetaData():
	def __init__(self, graph, validity, productions):
		self.graph = graph
		self.validity = validity
		self.productions = productions
		self.expansion_locations = [0]
	
	def copy_with_state(self):
		return GraphMetaData(deepcopy(self.graph), self.validity, self.productions, tf.identity(self.state))
	
	def expand(self, location, expansion):
		if location not in self.expansion_locations:
			raise ValueError("Location is not a candidate for expansion!", str(location))
		indices = []
		# Add new nodes and parent-child connections
		ixes = []
		label = self.graph.nodes[location].label
		if label != "#Expression" and label != "<HOLE>":
			index = len(self.graph.nodes)
			indices.append(index)
			self.graph.nodes.append(Node(index, " ".join(expansion), subtokens=expansion))
			self.graph.add_edge(Edge("Child", location, index))
			self.graph.add_edge(Edge("Parent", index, location))
		else:
			for e in expansion:
				index = len(self.graph.nodes)
				indices.append(index)
				self.graph.nodes.append(Node(index, "".join(e), subtokens=e))
				if e in ["#Expression", "#Variable", "#IntLiteral", "#StringLiteral"]: ixes.append(index)
				self.graph.add_edge(Edge("Child", location, index))
				self.graph.add_edge(Edge("Parent", index, location))
		
		# Rewire parent's prev/next token edges
		if "NextToken" in self.graph.type_edges:
			if location in self.graph.in_edges["NextToken"]:
				prev_to_this = self.graph.in_edges["NextToken"][location][0]
				self.graph.remove_edge(prev_to_this)
				self.graph.remove_edge(self.graph.out_edges["PrevToken"][location][0])
				self.graph.add_edge(Edge("NextToken", prev_to_this.source_ix, indices[0]))
				self.graph.add_edge(Edge("PrevToken", indices[0], prev_to_this.source_ix))
			if location in self.graph.out_edges["NextToken"]:
				this_to_next = self.graph.out_edges["NextToken"][location][0]
				self.graph.remove_edge(this_to_next)
				self.graph.remove_edge(self.graph.in_edges["PrevToken"][location][0])
				self.graph.add_edge(Edge("NextToken", indices[-1], this_to_next.target_ix))
				self.graph.add_edge(Edge("PrevToken", this_to_next.target_ix, indices[-1]))
		# Connect siblings
		for ix in range(1, len(indices)):
			self.graph.add_edge(Edge("NextToken", indices[ix - 1], indices[ix]))
			self.graph.add_edge(Edge("PrevToken", indices[ix], indices[ix - 1]))
		
		# Connect last-use edges
		if expansion in self.graph.variables:
			first_next = self.graph.variables[expansion][0] # Start at any variable with the same name; traverse its last-use edges until none exists. Insert this node there
			visited = set([first_next])
			while True:
				if "LastUse" in self.graph.out_edges and first_next in self.graph.out_edges["LastUse"]:
					first_next = self.graph.out_edges["LastUse"][first_next][0].target_ix
					if first_next in visited: # Don't get stuck in a cycle
						first_next = min(visited)
						break
					visited.add(first_next)
				else: break
			if first_next >= 0:
				if "LastUse" in self.graph.out_edges and first_next in self.graph.out_edges["LastUse"]: # Break any existing cycles
					self.graph.remove_edge(self.graph.out_edges["LastUse"][first_next][0])
					self.graph.remove_edge(self.graph.in_edges["NextUse"][first_next][0])
				self.graph.add_edge(Edge("LastUse", first_next, indices[0]))
				self.graph.add_edge(Edge("NextUse", indices[0], first_next))
		
		# Remove this location as a candidate for expansion and return new node indices for convenience
		self.expansion_locations.remove(location)
		self.expansion_locations.extend(list(reversed(ixes)))
		return indices

class JSONGraphLoader():
	edge_types = ["Child", "Parent", "NextToken", "PrevToken",	"LastUse", "NextUse"]#, "FormalArgName", "???", "LastWrite", "NextWrite", "LastLexicalUse", "NextLexicalUse","ComputedFrom", "ComputedBy", "GuardedBy", "GuardedByNegation", "ReturnsTo", "ReturnsFrom", "BindsToSymbol", "BoundBySymbol", "GuardedByNegation", "???-2"] # TODO: temporarily only use small subset of edge types!

	def __init__(self, graph_root, config, test_root=None):
		self.grammar = {}
		self.config = config
		self.max_graphs = config["loading"]["max_graphs"]
		self.max_nodes = config["loading"]["max_nodes"]
		self.max_expansions = config["training"]["max_expansions"]
		self.max_expansion_length = config["training"]["max_word_length"]
		
		print("Loading training")
		self.graph_data = self.load_graphs(graph_root, test_root=test_root, mode="training")
		keys = list(self.graph_data.keys())
		random.shuffle(keys)
		if test_root == None:
			self.train_keys = keys[:int(0.85*len(keys))]
			self.valid_keys = keys[int(0.85*len(keys)):int(0.9*len(keys))]
			self.test_keys = keys[int(0.9*len(keys)):]
		else:
			self.train_keys = keys
			print("Loading testing")
			test_data = self.load_graphs(test_root, mode="testing")
			for k, v in test_data.items():
				self.graph_data[k] = v
			self.test_keys = list(test_data.keys())
			random.shuffle(self.test_keys)
			self.valid_keys = self.test_keys[int(0.9*len(self.test_keys)):]
			self.test_keys = self.test_keys[:int(0.9*len(self.test_keys))]
		self.make_vocab(config["loading"]["vocab_cutoff"])
	
		
	# The invariant is actually stored as a set of productions to apply; we simply expand these into a sub-tree in the graph before running the discriminator
	def expand_graphs(self, batch):
		for i in range(self.config["training"]["max_expansions"]):
			# Filter graphs that still need completion
			to_complete = [data for data in batch if len(data.expansion_locations) > 0]
			if len(to_complete) == 0: break # Stop when no more locations need expanding
			
			# Extract supervision and inject invariant production step
			expansion_locs = [data.expansion_locations[-1] for data in to_complete]
			supervision = [data.productions[i][1][:-1] for data in to_complete] # get real productions and remove end-of-line token
			for gix, data in enumerate(to_complete):
				data.expand(data.expansion_locations[-1], supervision[gix]) # Add the expansion to the data object
		
			# Some logic for appending the actual indices to graph objects
			prod_exps = list(gix for gix in range(len(to_complete)) if to_complete[gix].graph.nodes[expansion_locs[gix]].label in ["<HOLE>", "#Expression"])		
			for gix, data in enumerate(to_complete):
				data = to_complete[gix]
				prod = supervision[gix]
				data.graph.indices.extend([self.vocab_key(w) for w in supervision[gix]])
				if gix in prod_exps:
					for _ in prod:
						loc = data.graph.locs[-1] + 1
						data.graph.locs.append(loc)
				else:
					loc = data.graph.locs[-1] + 1
					data.graph.locs.extend(len(prod)*[loc])
					
	def get_expansion_options(self, graph_data):
		expansion_options = {}
		for loc in graph_data.expansion_locations:
			key = graph_data.graph.nodes[loc].label
			if key == "#Variable":
				expansion_options[loc] = set()
				for var in graph_data.graph.variables.keys():
					expansion_options[loc].add(tuple([sub if sub in self.vocabulary else "<unk>" for sub in var] + ["</s>"]))
				# Temp: adds all production variables
				for key, val in graph_data.productions:
					if key == "#Variable" or key == "#Token": expansion_options[loc].add(tuple([sub if sub in self.vocabulary else "<unk>" for sub in val]))
			elif key == "#IntLiteral":
				expansion_options[loc] = set([str(x) for x in range(10)] + ["</s>"])
			else:
				if key == "<HOLE>": key = "#Expression"
				expansion_options[loc] = self.grammar[key]
		return expansion_options
	
	def batcher(self, max_batch_size, mode="training"):
		keys = self.train_keys if mode == "training" else self.test_keys if mode == "testing" else self.valid_keys
		options = [sample for k in keys for sample in self.graph_data[k]]
		if mode == "training": random.shuffle(options)
		batch = []
		batch_size = 0
		for sample in options:
			graph = sample.graph
			if batch_size + graph.size() > max_batch_size:
				yield batch
				batch = []
				batch_size = 0
			batch.append(sample)
			batch_size += graph.size()
		if batch_size > 0:
			yield batch
	
	def make_vocab(self, vocab_cutoff):
		word_counts = {}
		for key in self.train_keys:
			for data in self.graph_data[key]:
				for node in data.graph.nodes:
					for sub in node.subtokens:
						if sub in word_counts: word_counts[sub] += 1
						else: word_counts[sub] = 1
		
		# Store all sub-tokens used in grammar
		for k in self.grammar.keys():
			for v in self.grammar[k]:
				for sub in v:
					if sub in word_counts: word_counts[sub] += 1
					else: word_counts[sub] = 1
		
		# Get top words and make sure all production internal nodes are in the vocabulary
		top_words = set(w for w, c in word_counts.items() if c >= vocab_cutoff)
		for k in self.grammar.keys():
			if k not in top_words: top_words.add(k)
			for v in self.grammar[k]:
				for s in v:
					if not util.is_id(s):
						top_words.add(s)
		
		# Build the vocabulary
		self.vocabulary = {w:i for i, w in enumerate(top_words)}
		if "<unk>" not in self.vocabulary: self.vocabulary["<unk>"] = len(self.vocabulary)
		if "<s>" not in self.vocabulary: self.vocabulary["<s>"] = len(self.vocabulary)
		self.i2w = {i:w for w, i in self.vocabulary.items()}
		self.vocab_size = len(self.vocabulary)
		self.vocab_key = lambda w: self.vocabulary[w] if w in self.vocabulary else self.vocabulary["<unk>"] # Convenience function
		print("Vocab size:", self.vocab_size)
		
		for datas in self.graph_data.values():
			for data in datas:
				indices = []
				locs = []
				loc = 0
				for node in data.graph.nodes:
					sub_indices = [self.vocab_key(sub) for sub in node.subtokens]
					if len(sub_indices) > self.max_expansion_length: sub_indices = sub_indices[:self.max_expansion_length]
					locs.extend(len(sub_indices)*[loc]) # Aggregate each sub-token to one location
					loc += 1
					indices.extend(sub_indices)
				data.graph.indices = indices
				data.graph.locs = locs
	
	def load_graphs(self, graph_root, test_root=None, mode="training"):
		graph_data = {}
		if os.path.isfile(graph_root):
			if "-graph" in graph_root:
				graph_data = self.load_graph_file(graph_root, mode)
		else:
			for child in os.listdir(graph_root):
				f = os.path.join(graph_root, child)
				if test_root != None and f.lower().startswith(test_root.lower()): continue
				res = self.load_graphs(f, test_root)
				for key in res:
					if key in graph_data: graph_data[key].extend(res[key])
					else: graph_data[key] = res[key]
			count = sum([len(v) for k, v in graph_data.items()])
			print("Total:", count, "graphs")
		return graph_data
	
	def load_graph_file(self, graph_file, mode="training"):
		print("Loading", graph_file)
		if graph_file.endswith(".gz"):
			with gzip.open(graph_file, "rb") as f:
				raw_data = json.loads(f.read().decode())
		else:
			with open(graph_file, "r", encoding="utf-8") as f:
				raw_data = json.loads(f.read())
		random.shuffle(raw_data)
		
		graph_datas = {}
		no_graphs = 0
		for graph in raw_data:
			data = self.load_graph(graph, mode)
			if data == None: continue
			data.file = graph["Filename"]
			data.method = graph["MethodName"]
			key = data.file + ":" + data.method
			if key not in graph_datas: graph_datas[key] = []
			graph_datas[key].append(data)
			no_graphs += 1
			if no_graphs >= self.max_graphs: break
		print("Retrieved", no_graphs, "graphs")
		self.merge_grammars([data for values in graph_datas.values() for data in values])
		return graph_datas

	def load_graph(self, graph, mode="training"):
		# Load misc.
		validity = float(graph["Validity"])
		positive = int(graph["Positive"])
		negative = int(graph["Negative"])
		if mode == "training" and positive + negative < 10: return None
		
		# Load nodes
		node_labels = graph["ContextGraph"]["NodeLabels"]
		if len(node_labels) > self.max_nodes: return None
		node_types = None if "NodeTypes" not in graph["ContextGraph"] else graph["ContextGraph"]["NodeTypes"]
		nodes = {}
		for k, v in node_labels.items():
			type = None if node_types == None else node_types[k] if k in node_types else None
			node = Node(int(k), v, type)
			nodes[int(k)] = node
		
		# Load edges
		edge_categories = graph["ContextGraph"]["Edges"]
		edges = list() # Just store as list; Graph class will index these
		for edge_type in edge_categories.keys():
			for edge in edge_categories[edge_type]:
				if edge_type not in self.edge_types: continue
				if edge[0] not in nodes or edge[1] not in nodes: continue
				ix = self.edge_types.index(edge_type)
				reverse_type = self.edge_types[ix + 1]
				edges.append(Edge(edge_type, edge[0], edge[1]))
				edges.append(Edge(reverse_type, edge[1], edge[0]))
		
		# Store production rules
		ordered_prods = self.crawl_productions('1', graph["Productions"], graph["SymbolKinds"], graph["SymbolLabels"])
		if len(ordered_prods) >= self.max_expansions:
			ordered_prods = ordered_prods[:self.max_expansions]
		return GraphMetaData(Graph(nodes, edges), (validity, positive, negative), ordered_prods)
	
	def crawl_productions(self, key, productions, symbol_kinds, symbol_labels):
		name = "#" + symbol_kinds[key]
		if key not in productions:
			label = symbol_labels[key]
			if label.startswith("\""): label = "\"\""
			split = util.split_subtokens(label)
			if not isinstance(split, tuple): split = (label,)
			if len(split) > self.max_expansion_length - 1:
				split = tuple(list(split)[:self.max_expansion_length - 1])
			split += ("</s>",)
			if len(split) == 1: split = ("this", "</s>") # Temporary patch; bad tracer sometimes outputs variables as ".variable", creating empty tokens
			return [(name, split)]
		else:
			prod = productions[key]
			prod = prod[:self.max_expansion_length - 1] # Cut short if needed (should be very rare)
			real_prod = []
			for p in prod:
				if symbol_kinds[str(p)] == "Token":
					if not util.is_id(symbol_labels[str(p)]):
						real_prod.append(symbol_labels[str(p)])
					else: # Convert this to a variable
						real_prod.append("#Variable")
				else:
					real_prod.append("#" + symbol_kinds[str(p)])
			real_prod.append("</s>")
			res = [(name, tuple(real_prod))]
			for ix, p in enumerate(prod):
				if real_prod[ix].startswith("#"):
					res.extend(self.crawl_productions(str(p), productions, symbol_kinds, symbol_labels))
			return res
	
	def merge_grammars(self, graph_datas):
		for data in graph_datas:
			productions = data.productions
			for (k, ps) in productions:
				if k in self.grammar: self.grammar[k].add(ps)
				else: self.grammar[k] = set([ps])
