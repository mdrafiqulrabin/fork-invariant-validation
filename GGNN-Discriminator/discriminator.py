import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

from ggnn import GraphModel
from graph import *
import util

class Discriminator(tf.keras.Model):
	def __init__(self, config, graph_loader):
		super(Discriminator, self).__init__()
		self.ggnn = GraphModel(config, graph_loader.vocabulary, graph_loader.edge_types)
		self.linears = [tf.keras.layers.Dense(config["model"]["hidden_dim"]), tf.keras.layers.Dense(1)]
		self.optimizer = tf.train.AdamOptimizer(config["training"]["learning_rate"])
	
	def call(self, batch, eval=False, verbose=True):
		# Get predicitons and compute loss
		train_time = -time.time()
		with tf.GradientTape() as tape:
			d_exp = self.predict(tape, [data.graph for data in batch])
			true_validities = tf.floor(tf.stack([graph.validity[0] for graph in batch]))
			d_loss = true_validities*tf.log(1e-5 + d_exp) + (1 - true_validities)*tf.log(1 - d_exp + 1e-5) # Binary cross-entropy loss
			d_loss = -tf.reduce_mean(d_loss)
		train_time += time.time()
		
		# Calculate and apply gradients if not evaluating
		grad_time = -time.time()
		if not eval:
			grads = tape.gradient(d_loss, self.variables)
			grads, _ = tf.clip_by_global_norm(grads, 0.25)	
			self.optimizer.apply_gradients(zip(grads, self.variables))
		grad_time += time.time()
		
		if verbose:
			print("{0} graphs processed in {1:.3f} + {2:.3f}s, loss: {3:.3f}, some samples:".format(len(batch), train_time, grad_time, d_loss))
			true_validities = tf.floor(tf.stack([data.validity[0] for data in batch]))
			for i in range(min(3, len(batch))):
				sample = "Real: {0:.3f}\t Discriminator: {1:.3f}, Invariant: {2}".format(true_validities.numpy()[i], d_exp.numpy()[i], batch[i].graph.generated_expression())
				print(sample)
		
		return d_exp
	
	def predict(self, tape, graphs):
		with tape.stop_recording():
			graph_lens = [graph.size() for graph in graphs]
			sum_sizes = util.prefix_sum(graph_lens)
			hole_locs = []
			origins = []
			for gix, graph in enumerate(graphs):
				leaf_locs = self.get_leafs(graph, 0)
				hole_locs.extend([sum_sizes[gix] + l for l in leaf_locs])
				origins.extend([gix]*len(leaf_locs))
		
		# Extract per-tokens tates from the GGNN, then aggregate the states at the leafs (tokens) of the invariant and judge these through a two-layer perceptron
		states = self.ggnn(tape, graphs)
		states = tf.stack([states[ix] for ix in hole_locs])
		for layer in self.linears:
			states = layer(states)
		states = tf.segment_mean(states, origins)
		preds = tf.sigmoid(states)
		return tf.squeeze(preds, -1)
	
	def get_leafs(self, graph, ix):
		children = graph.get_children(ix)
		indices = [ix] if len(children) == 0 else []
		for child in children:
			indices.extend(self.get_leafs(graph, child))
		return indices
