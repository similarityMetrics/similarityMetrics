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

To run the similarity using attendgru embedding, the trained model needs to be downloaded in the similarityMetrics directory.
The model can be downloaded from the following link:
https://drive.google.com/file/d/1tDiv6kRRwydhYi8wY3Wgv_cL7Jkur-Pd/view?usp=sharing

To run the similarity using inferSent encoding, follow the instructions in their official repository to download the Glove embedding and the pretrained model. The instructions can be found in the following repository:
https://github.com/facebookresearch/InferSent

## Human Study Data
We also include the 210 function dataset that we used in the human study in the comsdata/ directory in both pickle format and txt format.
The raw data obtained from the human study is also made available in this repository in the final_megafile.csv file.
The Spearman Rho and Kendall Tau correlations we compute from the raw data is also made available in this repository.

## Run Universal Sentence Encoder on your own reference and baseline generated comments
```console
research@server:~/dev/similarityMetrics$ time python3 -W ignore use_score_v.py --help
```
This will output the list of input arguments that can be passed via the command line to figure out what information needs to be included to run the use_score_v.py file.

The use_score_v.py file can be used to get the cosine similarity of the embedding obtained using the universal sentence encoder(large) model (This is refered to as the USE+c score).
The following command can be run to obtain the USE+c score on the 210 comment set used in the human study for our paper:
```console
research@server:~/dev/similarityMetrics$ time python -W ignore use_score_v.py comsdata/attendgru_coms.txt --coms-filename=comsdata/refcoms.txt --batchsize=20000 --gpu=0
```
You can adjust the batch-size depending on the gpu memory and change the generated-input-filename and the coms-filename to compute the USE+c score on your own reference and generated output.
