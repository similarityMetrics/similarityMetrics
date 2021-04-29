"""
Functions for calculating the statistical significant differences between two dependent or independent correlation
coefficients.
The Fisher and Steiger method is adopted from the R package http://personality-project.org/r/html/paired.r.html
and is described in detail in the book 'Statistical Methods for Psychology'
The Zou method is adopted from http://seriousstats.wordpress.com/2012/02/05/comparing-correlations/
Credit goes to the authors of above mentioned packages!
"""

from __future__ import division

import sys
import argparse
import numpy as np
import pandas
import scipy.stats as ss
from scipy.stats import t, norm
from math import atanh, pow
from numpy import tanh
from sklearn.metrics import cohen_kappa_score

def simU(metric):
    p = pandas.read_csv("final_megafile.csv")
    print("Similarity Correlations for each user")

    user_ids = [1, 3, 4, 6, 10, 11]

    a = []
    d = []
    for u1, u2, u3, u4, u5, u6, m in zip(list(p.loc[p["user_id"]==1]["similarity"]), list(p.loc[p["user_id"]==3]["similarity"]), list(p.loc[p["user_id"]==4]["similarity"]), list(p.loc[p["user_id"]==6]["similarity"]), list(p.loc[p["user_id"]==10]["similarity"]), list(p.loc[p["user_id"]==11]["similarity"]), list(p.loc[p["user_id"]==1][metric])):
        u = (u1+u2+u3+u4+u5+u6)/6
        if u <=2:
           d.append(m)
        elif u>=3:
           a.append(m)
        # if u1 <=2 and u2 <=2 and u3 <=2 and u4<=2 and u5 <=2 and u6<=2:
        #     d.append(m) 
        # elif u1 >2 and u2 >2 and u3 >2 and u4 >2 and u5 <=2 and u6<=2:
        #     a.append(m) 

    print(metric, ss.mannwhitneyu(a,d))


def otherU(category, metric):
    if category == 'similarity' or category == 'similar':
        return simU(metric)
    if category == 'complete':
        category = 'adequate'
    p = pandas.read_csv("final_megafile.csv")
    # category = "adequate"

    user_ids = [1, 3, 4, 6, 10, 11]

    a = []
    d = []

    count = 0
    for u1, u2, u3, u4, u5, u6, t1, t2, t3, t4, t5, t6, m in zip(list(p.loc[p["user_id"]==1][category]), list(p.loc[p["user_id"]==3][category]), list(p.loc[p["user_id"]==4][category]), list(p.loc[p["user_id"]==6][category]), list(p.loc[p["user_id"]==10][category]), list(p.loc[p["user_id"]==11][category]), list(p.loc[p["user_id"]==1]["source"]), list(p.loc[p["user_id"]==3]["source"]), list(p.loc[p["user_id"]==4]["source"]), list(p.loc[p["user_id"]==6]["source"]), list(p.loc[p["user_id"]==10]["source"]), list(p.loc[p["user_id"]==11]["source"]), list(p.loc[p["user_id"]==1][metric])):
        ref = []
        pred = []
        if t1 == 'reference':
            ref.append(u1)
        else:
            pred.append(u1)
        if t2 == 'reference':
            ref.append(u2)
        else:
            pred.append(u2)
        if t3 == 'reference':
            ref.append(u3)
        else:
            pred.append(u3)
        if t4 == 'reference':
            ref.append(u4)
        else:
            pred.append(u4)
        if t5 == 'reference':
            ref.append(u5)
        else:
            pred.append(u5)
        if t6 == 'reference':
            ref.append(u6)
        else:
            pred.append(u6)
        avgr = sum(ref)/len(ref)
        avgp = sum(pred)/len(pred)

        if avgr<=2:
            d.append(m)
        elif avgr>=3:
            a.append(m)

        if avgp<=2:
            d.append(m)
        elif avgp>=3:
            a.append(m)
        # if r[0]<=2 and r[1]<=2:
        #     d.append(m)
        # elif r[0]>=3 and r[1]>=3:
        #     a.append(m)

        # if p[0]<=2 and p[1]<=2:
        #     d.append(m)
        # elif p[0]>=3 and p[1]>=3:
        #     a.append(m)

    # print('>=3:', len(sorted(a)))
    # # print(sorted(a))
    # print('<=2', len(sorted(d)))
    # print(sorted(d))
    # sys.exit()
    print(metric, ss.mannwhitneyu(a,d))

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--metric', dest='metric', type=str, default="bleu-average")
    parser.add_argument('--category', dest='category', type=str, default="similarity")
    args = parser.parse_args()
    category = args.category
    #pandas.set_option("display.max_rows", None, "display.max_columns", None)
    ms = ['bleu-1gram', 'bleu-average', 'rouge-l', 'rouge-w', 'meteor_score', 'jaccard_similarity_score', 'f1bert', 'tfidf_cosine', 'tfidf_euclidean', 'iS_cosine', 'iS_euclidean', 'use_cosine_dict', 'use_euclidean_dict', 'sb_cosine', 'sb_euclidean', 'attendgru_flatgru_cosine', 'attendgru_flatgru_euclidean']
    for m in ms:
        otherU(category, m)
