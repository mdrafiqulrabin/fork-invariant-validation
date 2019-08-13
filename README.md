# [ This repository has been forked from https://doi.org/10.5281/zenodo.2574189 ]


## Invariant validation: data and code
This repository contains a rudimentary replication package for our paper "Are My Invariants Valid? A Learning Approach".
In the interest of open science, we anonymously release our (processed) data, code and some output logs; this document describes how to use these.

## Data files
We release graph data files belonging to each of our eight studied projects, with pre- and post-condition files stored separately. These can be found in the Graphs-Pre and Graphs-Post directories. Every project has its own directory with one or more g-zipped JSON graph files ending in "-graph.*index*.gz", where the index is used when graphs were split over multiple files when a data limit was exceeded, which is only the case for Lucene in our dataset.

Every project folder in our Graphs data also contains a plain-text file with the same data, but stored without edge information; this file is used by our RNNs (and makes for easier reading). The data is tab-separated, one method-invariant pair per line; its format is: File-path (which is machine-specific but not used by our code), Method-name (fully qualified with parameters), invariant, validity (score in [0, 1]), number of supporting trace files, number of non-supporting trace files, tokenized method (tokens split by spaces; this actually includes the invariant as an if-statement, which we remove in the modeling code), tokenized invariant (same tokenization).

We also release our manually annotated invariants in the file "ManualAnnotations.tsv". These contain, tab-separated, our annotations for each method-invariant pair. The format is given in the header row: Consensus, Rater 1's assessment, Rater 2's assessment, Method URL (at the correct revision), Project, Method (fully qualified with parameters), Invariant, Comments (may be empty). The comments occasionally include explanations for the more complex cases, as well as annotations where a valid invariant was not actually perceived as useful, where an irrelevant invariant may be valid, and some insight into places where the two raters deliberated about a judgement.

## Code
Our Gated-Graph Neural Discriminator is included in the directory GGNN-Discriminator. It can be run on any of our Graphs files and directories by invoking main.py. Invoking it without parameters (or with -h) will print its usage information; generally it can be run with a train-directory, an optional test-file or directory (which may be inside the train-directory, in which case it is excluded from training) and an optional log-file to write the results to. The parameters in config.yaml are the ones used in our work and may be changed e.g. to reduce the batch size on smaller GPUs. The code is written in Python 3 and requires the following libraries to run: tensorflow 1.12+, numpy (typically included with TF), argparse and yaml, which can all be installed using "pip".

We also include our RNN model, used in our ablation study, in the RNN directory. It can be run similarly to our GGNN; since it was only used in cross-project evaluation, the syntax here is to pass it a data-root and test-project (name) which should be inside the data-root. An optional log-file can again be specified. The config.yaml file can again be altered for different modeling and training parameters.

## Output logs
We also release the logs with the output of running our code in the Logs-Pre & -Post directories. Here, each folder stores results for the full GGNN (in folders "Full-Intra", for intra-project models, and "Full-Cross", for cross-project models), the No-Context model (only cross-project results) and the RNN (also only cross-project results). Each results folder contains a .txt file for each project; the exact format depends on the tool that produced that file, but each file always contains the results for all epochs (epoch number either listed at the start of each row or as a delimiter line between results), and each line in each epoch's results contains (in some order):
- the method name (fully qualified),
- the invariant,
- the invariant's actual validity (and possibly counts of supporting and non-supporting trace files),
- the probability assigned to the invariant by the model.

These results (especially typically those at epoch 3) can be used in a straightforward way to reproduce all our plots and results; we will include the corresponding Python/R scripts for alignment with the full replication package.
