import sys
import os
import yaml
import argparse
import math

# Import Tensorflow and enable Eager execution
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
tf.enable_eager_execution()

from json_graph_loader import *
from discriminator import Discriminator

# Train the model, evaluating validation and test performance every epoch. Writes to log_file if available
def train(graph_loader, log_file=None):
	model = Discriminator(config, graph_loader) # Create the discriminator; we pass it the graph_loader for metadata, such as vocabulary size
	for i in range(config["training"]["num_epochs"]):
		epoch = i + 1
		log(log_file, "Epoch {0}".format(epoch))
		
		print("Epoch:", epoch)
		for batch in graph_loader.batcher(config["training"]["max_batch_size"], mode="training"):
			run_batch(graph_loader, batch, model)
		
		print("Evaluating:")
		eval(graph_loader, model, epoch, mode="validation", log_file=log_file)
		eval(graph_loader, model, epoch, mode="testing", log_file=log_file)

def eval(graph_loader, model, epoch, mode="testing", log_file=None):
	total = tp = tn = fp = fn = 0
	for batch in graph_loader.batcher(config["training"]["max_batch_size"], mode=mode):
		disc = run_batch(graph_loader, batch, model, eval=True).numpy()
		# Collect the predictions and some intermediate statistics (e.g. acc, prec, rec)
		res = []
		for ix, data in enumerate(batch):
			res.append("{0}\t{1}\t{2:.6f}\t{3}\t{4}\t{5:.6f}".format(data.file, data.method, data.validity[0], data.validity[1], data.validity[2], disc[ix]))
			total += 1
			if math.floor(data.validity[0]) == 1.0:
				if round(disc[ix]) == 1.0: tp += 1
				else: fn += 1
			else:
				if round(disc[ix]) == 1.0: fp += 1
				else: tn += 1
		log(log_file, res) # Log every prediction if a log file exists
	# Compute summary statistics and print/log
	prec = 0.0 if tp + fp == 0 else tp / (tp + fp) # Treat specially to avoid division by zero
	rec = tp / (tp + fn)
	neg_acc = tn / (tn + fp)
	print("{0} on {1} samples: discriminator: acc: {2:.3f}, prec: {3:.3f}, rec: {4:.3f}".format(mode, total, (tp + tn) / total, prec, rec))
	log(log_file, "{0} on {1} samples, acc: {2:.3f}, prec: {3:.3f}, rec: {4:.3f}, -acc: {5:.3f}".format(mode, total, (tp + tn) / total, prec, rec, neg_acc))

# Clones the batch, adds the invariant into each method graph and sets up index tensors for the discriminator; then, runs the model
def run_batch(graph_loader, batch, model, eval=False, verbose=True):
	batch = [deepcopy(data) for data in batch]
	graph_loader.expand_graphs(batch)
	for data in batch:
		data.graph.indices = tf.one_hot(data.graph.indices, len(graph_loader.vocabulary), dtype=tf.float32)
	return model(batch, eval)

def log(log_file, data):
	if log_file == None: return
	with open(log_file, "a") as f:
		if isinstance(data, list):
			for line in data:
				f.write(line)
				f.write('\n')
		else:
			f.write(data)
			f.write('\n')

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("train", help="Path to training data, file or directory.")
	ap.add_argument("--test", required=False, help="Path to test data. If not supplied, training data is randomly split into train/valid/test.")
	ap.add_argument("--output", required=False, help="The output file to write logs to")
	args = ap.parse_args()
	with open('config.yaml', 'r') as f:
		config = yaml.load(f)
	print(config)
	out_file = args.output
	if out_file != None:
		with open(out_file, "w") as f: pass
	train(JSONGraphLoader(args.train, config, args.test), out_file)
