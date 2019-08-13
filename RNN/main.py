import os
import re
import math
import random
import string
import argparse
import yaml
with open('config.yaml', 'r') as f:
	config = yaml.load(f)

import cntk as C
import numpy as np

random.seed(31)

def main():
	# Set up some global variables to reduce parameter passing
	global train_data, valid_data, test_data
	global vocab, i2w, translator
	global method, invariant, validity
	global model, prediction
	
	# Extract arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("data_root", help="Path to data root directory, must contain project with test name. All other projects are treated as training data")
	ap.add_argument("test_project", help="Name of test project within root")
	ap.add_argument("--output", required=False, help="The output file to write logs to")
	args = ap.parse_args()
	log_file = args.output
	if log_file != None: # Empty the log file
		with open(log_file, "w") as f: pass
	print(config)
	
	# Load the datasets and split into train/valid/test, build according vocabulary
	data = load(args.data_root)
	train_data = [sample for key, vals in data.items() for sample in vals if key != args.test_project and sample[3] >= 10] # Drop training samples with insuffficient support
	test_data = data[args.test_project]
	valid_data = test_data[int(.9*len(test_data)):] # 10% held-out
	vocab, i2w, translator = make_vocab(train_data)
	vocab_dim = len(vocab)
	print("Finished loading, got {0} samples with {1} tokens, Vocabulary: {2}".format(len(train_data), sum([len(p[0]) + len(p[1]) for p in train_data]), vocab_dim))
	
	# Setup input placeholders
	method_axis = C.Axis('inputAxis')
	invariant_axis = C.Axis('labelAxis')
	method = C.sequence.input_variable((), sequence_axis=method_axis, name="method_inp")
	invariant = C.sequence.input_variable((), sequence_axis=invariant_axis, name="invariant_inp")
	validity = C.input_variable(1, name="validity_inp")

	# Create model and instantiate its output node, then train
	model = create_model(config["model"]["embed_dim"], config["model"]["hidden_dim"], config["model"]["num_layers"])
	prediction = model(C.one_hot(method, vocab_dim), C.one_hot(invariant, vocab_dim))
	train(log_file)

""" Training and testing """
def validate(metric):
	count = 0
	acc_sum = 0
	for mb in batcher(valid_data, is_training=False):
		acc = metric.eval({method: mb[2], invariant: mb[3], validity: mb[4]})
		for acc_list in acc:
			acc_sum += np.sum(acc_list)
			count += len(acc_list)
	print("Validation complete, count: {0}, accuracy: {1:.2%}".format(count, acc_sum/count))

def test():
	vals = []
	for mb in batcher(test_data, is_training=False):
		preds = prediction.eval({method: mb[2], invariant: mb[3]})
		for ix, pred in enumerate([p for pr in preds for p in pr]):
			vals.append((mb[0][ix], mb[1][ix], mb[-1][ix][0], pred))
	return vals

def train(log_file=None):
	C.logging.log_number_of_parameters(prediction)
	num_epochs = config['training']['num_epochs']
	lr = config["training"]["learning_rate"]
	max_lr_epochs = config['training']['max_lr_epochs']
	lr_schedule = C.learning_parameter_schedule_per_sample([lr]*(max_lr_epochs - 1) + [lr*(0.5**i) for i in range(num_epochs - max_lr_epochs + 1)], epoch_size=len(train_data))
	loss = C.losses.binary_cross_entropy(prediction, validity)
	metric = C.ops.equal(C.ops.round(prediction), validity)
	trainer = C.Trainer(None,
		(loss, metric),
		C.adam(prediction.parameters, lr = lr_schedule, momentum = 0.9, gradient_clipping_threshold_per_sample=5, gradient_clipping_with_truncation=True),
		C.logging.ProgressPrinter(tag='Training', num_epochs=num_epochs, freq=config['training']['print_freq']))
	
	for epoch in range(num_epochs):
		print("Epoch", (epoch + 1))
		for mb in batcher(train_data, is_training=True):
			trainer.train_minibatch({method: mb[2], invariant: mb[3], validity: mb[4]})
		validate(metric)
		preds = test()
		if log_file != None:
			with open(log_file, "a") as f:
				for p in preds:
					f.write("{0}\t{1}\t{2}\t{3:.6f}\t{4:.6f}\n".format(epoch, p[0], p[1], p[2], p[3]))
				f.flush()

""" Modeling setup """
def BiRecurrence(dim):
	F = C.layers.Recurrence(C.layers.GRU(dim//2))
	G = C.layers.Recurrence(C.layers.GRU(dim//2), go_backwards=True)
	x = C.placeholder()
	return C.splice(F(x), G(x))

def create_model(embed_dim, hidden_dim, num_layers):
	embed = C.layers.Embedding(embed_dim) # Embed vocabulary
	method_rnn = C.layers.For(range(num_layers), lambda: BiRecurrence(hidden_dim))
	invariant_rnn = C.layers.For(range(num_layers), lambda: BiRecurrence(hidden_dim))
	method_bn = C.layers.BatchNormalization()
	invariant_bn = C.layers.BatchNormalization()
	discriminator = C.layers.Dense(2*hidden_dim, activation=C.ops.relu) >> C.layers.Dense(1, activation=C.ops.sigmoid)
	
	@C.Function
	def reduce_mean_seq(s):
		return C.sequence.reduce_sum(s)/C.sequence.reduce_sum(0*s + 1)
	
	@C.Function
	def model(m, i):
		method_encoding = reduce_mean_seq(method_rnn(embed(m))) >> method_bn
		invariant_encoding = reduce_mean_seq(invariant_rnn(embed(i))) >> invariant_bn
		return discriminator(C.ops.splice(method_encoding, invariant_encoding))
	
	return model


""" Simple Data Utils """
# Identifier check function
c_sharp_keywords = ["abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked", "class", "const", "continue", "decimal", "default", "delegate", "do", "double", "else", "enum", "event", "explicit", "extern", "false", "finally", "fixed", "float", "for", "foreach", "goto", "if", "implicit", "in", "int", "interface", "internal", "is", "lock", "long", "namespace", "new", "null", "object", "operator", "out", "override", "params", "private", "protected", "public", "readonly", "ref", "return", "sbyte", "sealed", "short", "sizeof", "stackalloc", "static", "string", "struct", "switch", "this", "throw", "true", "try", "typeof", "uint", "ulong", "unchecked", "unsafe", "ushort", "using", "using", "static", "virtual", "void", "volatile", "while", "add", "alias", "ascending", "async", "await", "descending", "dynamic", "from", "get", "global", "group", "into", "join", "let", "nameof", "orderby", "partial", "partial", "remove", "select", "set", "value", "var", "when", "where", "where", "yield"]
id_regex = "[a-zA-Z$_][a-zA-Z0-9$_]*"
def is_id(name):
	return re.match(id_regex, name) and name not in c_sharp_keywords

regex = re.compile('[%s]' % re.escape(string.punctuation))
def split_subtokens(token):
	def camel_case_split(identifier):
		if len(identifier) == 0: return []
		if not is_id(identifier):
			if identifier[0] == "\"": return ["\"str\""]
			elif identifier[0] == "'": return ["\'chr'"]
			else: return [c for c in identifier]
		if "_" in identifier:
			parts = identifier.split("_")
			return [m for p in parts for m in camel_case_split(p)]
		
		matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
		return [m.group(0).lower() for m in matches]
	
	if not is_id(token):
		if "\"" in token or "'" in token: return token
		elif token in c_sharp_keywords: return token
		elif re.match(regex, token): return token
		elif token == "<HOLE>": return token
		else: return tuple(token)
	parts = [p for t in token.split("_") for p in camel_case_split(t)]
	parts = [p.lower() for p in parts if len(p) > 0]
	if len(parts) == 0:
		parts = [token]
	return tuple(parts)

def post_process(sequence):
	cleaned = []
	for word in sequence:
		if '"' in word: cleaned.append('""')
		elif "'" in word: cleaned.append("''")
		else:
			split = split_subtokens(word)
			if not isinstance(split, tuple): cleaned.append(word)
			else:
				for part in split: cleaned.append(part)
	return cleaned

def load_file(file):
	samples = []
	with open(file, encoding="utf-8") as f:
		for line in f:
			parts = line.rstrip().split("\t")
			if len(parts) < 8: continue
			invariant = parts[7]
			method = parts[6]
			method = method.replace("if ( " + invariant + " ) ;", "", 1)
			while True:
				s = re.search('"[^"]+[^\\\\]"', method)
				if s == None: break
				method = method.replace(s.group(0), '""')
			method = post_process(method.split(" "))
			invariant = post_process(invariant.split(" "))
			count = int(parts[4]) + int(parts[5])
			validity = math.floor(float(parts[4]) / float(count))
			samples.append((parts[1], method, invariant, count, validity))
	return samples

def load(dir):
	data = {}
	for child in os.listdir(dir):
		project = os.path.join(dir, child)
		for file in os.listdir(project):
			if not file.endswith(".gz"):
				print("Loading:", file)
				res = load_file(os.path.join(project, file))
				data[child.lower()] = res
	return data

def make_vocab(train_data):
	counts = {}
	for sample in train_data:
		for token in sample[1] + sample[2]:
			if token not in counts: counts[token] = 1
			else: counts[token] += 1
	
	vocab = [token for token, count in counts.items() if count >= config["words"]["vocab_cutoff"]]
	vocab.append("<unk>")
	vocab = {token:ix for ix, token in enumerate(vocab)}
	i2w = {i:w for i, w in vocab.items()}
	translator = lambda w: vocab[w] if w in vocab else vocab["<unk>"]
	return vocab, i2w, translator

def batcher(data, is_training=True):
	def finish(batch):
		batch[-1] = np.array(batch[-1], dtype=np.float32)
		return batch
	if is_training: random.shuffle(data)
	batch = [[], [], [], [], []]
	batch_size = 0
	for (method_name, method, invariant, count, validity) in data:
		samples = len(method) + len(invariant)
		if batch_size + samples > config["training"]["batch_size"]:
			yield finish(batch)
			batch = [[], [], [], [], []]
			batch_size = 0
		batch[0].append(method_name)
		batch[1].append("".join(invariant))
		batch[2].append(np.array([translator(w) for w in method], dtype=np.float32))
		batch[3].append(np.array([translator(w) for w in invariant], dtype=np.float32))
		batch[4].append([validity])
		batch_size += samples
	if len(batch) > 0: yield finish(batch)

if __name__ == '__main__':
	main()
