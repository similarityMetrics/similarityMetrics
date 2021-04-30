# Semantic Similarity Metrics for Evaluating Source Code Summarization
This repository contains code and data for calculating semantic similarity metrics for evaluating source code summarization. This project is aimed to demonstrate the empirical and statistical evidence that bleu score does not correlate between similarity between reference and autmatically generated comments. It also provides a script for computing cosine similarity score using universal sentence encoder on the comment pairs.

## Dependencies
We assume Ubuntu 18.04, Python 3.6.7, Keras 2.4.3, numpy 1.19.5, Tensorflow 2.4.1, javalang 0.13.0, nltk 3.6.1, pandas 1.1.5, py-rouge 1.1, pytorch 1.8.1, sentence-transformers 1.1.0, matplotlib 3.3.4

## Similarity Metrics
### Step 1: Obtain Dataset
We use the dataset of 2.1m Java methods and method comments, already cleaned and separated into train/val/test sets by LeClair et al.

(Their raw data was downloaded from: http://leclair.tech/data/funcom/)

### Step 2: Train Attendgru
We organize and train the attendgru model using the reccomendations also by LeClair et. al.

(Their code was downloaded from: https://github.com/mcmillco/funcom)

### Step 3: Run Similarity Metrics
```console
research@server:~/dev/similarityMetrics$ time python3 -W ignore simmetrics.py --help
```
This will output the list of input arguments that can be passed via the command line to figure out what information needs to be included to run the simmetrics.py file.
