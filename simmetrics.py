import sys
import pickle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import argparse

datapath = '.'
outpath='.'

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def p_cosine_greedy(cosine_matrix):
    _, pred_len = cosine_matrix.shape
    pnum = list()
    for i in range(pred_len):
        pnum.append(max((cosine_matrix[:, i])))
    p = sum(pnum)/len(pnum)
    return p

def r_cosine_greedy(cosine_matrix):
    rnum = list()
    for ref_cos in cosine_matrix:
        rnum.append(max(ref_cos))
    r = sum(rnum)/len(rnum)
    return r

def f1_score(p, r):
    return ((2*p*r)/(p+r))

def p_euclidean_greedy(euclidean_matrix, rescale=False):
    _, pred_len = euclidean_matrix.shape
    pnum = list()
    for i in range(pred_len):
        minval = min((euclidean_matrix[:, i]))
        if rescale:
            minval = math.exp(-minval)
        pnum.append(minval)
    p = sum(pnum)/len(pnum)
    return p

def r_euclidean_greedy(euclidean_matrix, rescale=False):
    rnum = list()
    for ref_eucd in euclidean_matrix:
        minval = min(ref_eucd)
        if rescale:
            minval = math.exp(-minval)
        rnum.append(minval)
    r = sum(rnum)/len(rnum)
    return r

def f1_euclidean_greedy(euclidean_matrix, rescale=False):
    p = p_euclidean_greedy(euclidean_matrix)
    r = r_euclidean_greedy(euclidean_matrix)
    f1 = (2*p*r)/(p+r)
    if rescale:
        return (math.exp(-f1))
    return f1


def jaccard_similarity_score(fidlist, ref, pred, average=None):
    score_list = list()
    score_dict = dict()
    for i in range(len(ref)):
        refwords = set(ref[i])
        predwords = set(pred[i])
        fid = fidlist[i]
        intersection = refwords.intersection(predwords)
        union = refwords.union(predwords)
        score_dict[fid] = float(len(intersection)/len(union))
    pickle.dump(score_dict, open(outpath+'/jaccard_similarity_score.pkl', 'wb'))
    if average is None:
        return list(score_dict.values())
    elif average=='unweighted':
        score_list = list(score_dict.values())
        js = sum(score_list)/len(score_list)
        js = round(js*100, 2)
        ret = ('for %s functions\n' %(len(pred)))
        ret+= ('Jaccard Score %s\n' %(js))
        return ret

def all_bleu_score(reflist, preds):
    refs = list()
    for ref in reflist:
        refs.append([ref])
    ba = corpus_bleu(refs, preds)
    b1 = corpus_bleu(refs, preds, weights=(1,0,0,0))
    b2 = corpus_bleu(refs, preds, weights=(0,1,0,0))
    b3 = corpus_bleu(refs, preds, weights=(0,0,1,0))
    b4 = corpus_bleu(refs, preds, weights=(0,0,0,1))

    ret = ('for %s functions\n' %(len(preds)))
    ret+= ('Ba %s\n' % (round(ba*100, 2)))
    ret+= ('B1 %s\n' % (round(b1*100, 2)))
    ret+= ('B2 %s\n' % (round(b2*100, 2)))
    ret+= ('B3 %s\n' % (round(b3*100, 2)))
    ret+= ('B4 %s\n' % (round(b4*100, 2)))
    return ret

def indv_bleu_score(fidlist, reflist, predlist):
    badict = dict()
    b1dict = dict()
    b2dict = dict()
    b3dict = dict()
    b4dict = dict()
    for fid, ref, pred in zip(fidlist, reflist, predlist):
        badict[fid] = sentence_bleu([ref], pred)
        b1dict[fid] = sentence_bleu([ref], pred, weights=(1,0,0,0))
        b2dict[fid] = sentence_bleu([ref], pred, weights=(0,1,0,0))
        b3dict[fid] = sentence_bleu([ref], pred, weights=(0,0,1,0))
        b4dict[fid] = sentence_bleu([ref], pred, weights=(0,0,0,1))
    pickle.dump(badict, open(outpath+'/bleu-average.pkl', 'wb'))
    pickle.dump(b1dict, open(outpath+'/bleu-1gram.pkl', 'wb'))
    pickle.dump(b2dict, open(outpath+'/bleu-2gram.pkl', 'wb'))
    pickle.dump(b3dict, open(outpath+'/bleu-3gram.pkl', 'wb'))
    pickle.dump(b4dict, open(outpath+'/bleu-4gram.pkl', 'wb'))

    balist = list(badict.values())
    b1list = list(b1dict.values())
    b2list = list(b2dict.values())
    b3list = list(b3dict.values())
    b4list = list(b4dict.values())

    avg_ba = sum(balist)/len(balist)
    avg_b1 = sum(b1list)/len(b1list)
    avg_b2 = sum(b2list)/len(b2list)
    avg_b3 = sum(b3list)/len(b3list)
    avg_b4 = sum(b4list)/len(b4list)

    ret = ('for %s functions\n' %(len(predlist)))
    ret+= ('Ba %s\n' % (round(avg_ba*100, 2)))
    ret+= ('B1 %s\n' % (round(avg_b1*100, 2)))
    ret+= ('B2 %s\n' % (round(avg_b2*100, 2)))
    ret+= ('B3 %s\n' % (round(avg_b3*100, 2)))
    ret+= ('B4 %s\n' % (round(avg_b4*100, 2)))
    return ret

def prepare_rouge_results(p, r, f, metric):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}\n'.format(metric, 'P', 100.0*p, 'R', 100.0*r, 'F1', 100.0*f)

def all_rouge_score(reflist, predlist):
    import rouge
    refs = list()
    preds = list()
    for ref in reflist:
        refs.append(' '.join(ref).strip())
    for pred in predlist:
        preds.append(' '.join(pred).strip())
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'], max_n=4, limit_length=True,
                            length_limit=100, length_limit_type='words', alpha=.5, weight_factor=1.2)
    scores = evaluator.get_scores(refs, preds)

    ret = ('for %s functions\n' %(len(preds))) 
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        ret+= prepare_rouge_results(results['p'], results['r'], results['f'], metric)

    return ret

def indv_rouge_score(fidlist, reflist, predlist):
    import rouge
    rougedict = dict()
    for fid, ref, pred in zip(fidlist, reflist, predlist):
        ref = ' '.join(ref).strip()
        pred = ' '.join(pred).strip()
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'], max_n=4, limit_length=True,
                            length_limit=100, length_limit_type='words', alpha=.5, weight_factor=1.2)
        score = evaluator.get_scores(ref, pred)
        score = sorted(score.items(), key=lambda x: x[0])
        rougedict[fid] = score
    pickle.dump(rougedict, open(outpath+'/rouge_score.pkl', 'wb'))
    return f_rouge_score(reflist, predlist)

def all_meteor_score(reflist, predlist):
    from nltk.translate.meteor_score import single_meteor_score
    mslist = list()
    for ref, pred in zip(reflist, predlist):
        ref = ' '.join(ref).strip()
        pred = ' '.join(pred).strip()
        if pred == '':
            mslist.append(0)
            continue
        ms = single_meteor_score(ref, pred)
        mslist.append(ms)
    avg_ms = sum(mslist)/len(mslist)
    ret = ('for %s functions\n' %(len(predlist)))
    ret += ('Meteor score %s\n' %(round(avg_ms*100, 2)))
    return ret

def indv_meteor_score(fidlist, reflist, predlist):
    from nltk.translate.meteor_score import single_meteor_score
    msdict = dict()
    for fid, ref, pred in zip(fidlist, reflist, predlist):
        ref = ' '.join(ref).strip()
        pred = ' '.join(pred).strip()
        if pred == '':
            msdict[fid] = 0
            continue
        ms = single_meteor_score(ref, pred)
        msdict[fid] = ms
    pickle.dump(msdict, open(outpath+'/meteor_score.pkl', 'wb'))
    mslist = list(msdict.values())
    avg_ms = sum(mslist)/len(mslist)
    ret = ('for %s functions\n' %(len(predlist)))
    ret += ('Meteor score %s\n' %(round(avg_ms*100, 2)))
    return ret

def tfidf_vectorizer(fidlist, reflist, predlist):
    from sklearn.feature_extraction.text import TfidfVectorizer
    cosine_score_dict = dict()
    euclidean_distance_dict = dict()
    for fid, ref, pred in zip(fidlist, reflist, predlist):
        ref = ' '.join(ref).strip()
        pred = ' '.join(pred).strip()
        if pred == '':
            cosine_score_dict[fid] = 0
            continue
        data = [ref, pred]
        vect = TfidfVectorizer()
        vector_matrix = vect.fit_transform(data)
        css = cosine_similarity_score(vector_matrix[0].todense(), vector_matrix[1].todense())[0][0]
        ess = euclidean_distance_score(vector_matrix[0].todense(), vector_matrix[1].todense())[0][0]
        cosine_score_dict[fid] = css
        euclidean_distance_dict[fid] = ess
    pickle.dump(cosine_score_dict, open(outpath+'/tfidf_cosine.pkl', 'wb'))
    pickle.dump(euclidean_distance_dict, open(outpath+'/tfidf_euclidean.pkl', 'wb'))
    cosine_score_list = list(cosine_score_dict.values())
    euclidean_distance_list = list(euclidean_distance_dict.values())
    avg_css = sum(cosine_score_list)/len(cosine_score_list)
    avg_ess = sum(euclidean_distance_list)/len(euclidean_distance_list)
    ret = ('for %s functions\n' % (len(predlist)))
    ret+= 'Cosine similarity score with tfidf vectorizer is %s\n' % (round(avg_css*100, 2))
    ret+= 'Euclidean distance score with tfidf vectorizer is %s\n' % (round(avg_ess*100, 2))
    return ret


def indv_universal_sentence_encoder_dict(fidlist, reflist, predlist):
    import tensorflow_hub as tfhub

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = tfhub.load(module_url)
    cosine_score_dict = dict()
    euclidean_distance_dict = dict()

    for fid, ref, pred in zip(fidlist, reflist, predlist):
        ref = ' '.join(ref).strip()
        pred = ' '.join(pred).strip()
        if pred == '':
            cosine_score_dict[fid] = 0
            continue    
        data = [ref, pred]
        data_emb = model(data)
        data_emb = np.array(data_emb)
        
        css = cosine_similarity_score(data_emb[0].reshape(1, -1), data_emb[1].reshape(1, -1))[0][0]
        ess = euclidean_distance_score(data_emb[0].reshape(1, -1), data_emb[1].reshape(1, -1))[0][0]
        cosine_score_dict[fid] = css
        euclidean_distance_dict[fid] = ess
    pickle.dump(cosine_score_dict, open(outpath+'/use_cosine_dict.pkl', 'wb'))
    pickle.dump(euclidean_distance_dict, open(outpath+'/use_euclidean_dict.pkl', 'wb'))
    cosine_score_list = list(cosine_score_dict.values())
    euclidean_distance_list = list(euclidean_distance_dict.values())
    avg_css = sum(cosine_score_list)/len(cosine_score_list)
    avg_ess = sum(euclidean_distance_list)/len(euclidean_distance_list)

    ret = ('for %s functions\n' % (len(predlist)))
    ret+= 'Cosine similarity score with universal sentence encoder embedding is %s\n' % (round(avg_css*100, 2))
    ret+= 'Euclidean distance score with universal sentence encoder embedding is %s\n' % (round(avg_ess*100, 2))
    return ret

def all_universal_sentence_encoder_dict(fidlist, reflist, predlist):
    import tensorflow_hub as tfhub

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = tfhub.load(module_url)
    refs = list()
    preds = list()
    count = 0

    for ref, pred in zip(reflist, predlist):
        ref = ' '.join(ref).strip()
        pred = ' '.join(pred).strip()
        if pred == '':
            count+=1
            continue
        refs.append(ref)
        preds.append(pred)

    total_csd = np.zeros(count)
    ref_emb = model(refs)
    pred_emb = model(preds)
    csm = cosine_similarity_score(ref_emb, pred_emb)
    csd = csm.diagonal()
    total_csd = np.concatenate([total_csd, csd])
    avg_css = np.average(total_csd)

    esm = euclidean_distance_score(ref_emb, pred_emb)
    esd = esm.diagonal()
    avg_ess = np.average(esd)

    ret = ('for %s functions\n' % (len(predlist)))
    ret+= 'Cosine similarity score with universal sentence encoder embedding is %s\n' % (round(avg_css*100, 2))
    ret+= 'Euclidean distance score with universal sentence encoder embedding is %s\n' % (round(avg_ess*100, 2))
    return ret

def official_bert_score(fidlist, reflist, predlist):
    from bert_score import score
    refs = list()
    preds = list()
    p_bert = dict()
    r_bert = dict()
    f1_bert = dict()
    for ref, pred in zip(reflist, predlist):
        ref = ' '.join(ref).strip()
        pred = ' '.join(pred).strip()
        refs.append(ref)
        preds.append(pred)
    p, r, f1 = score(preds, refs, lang='en', rescale_with_baseline=True)
    for fid, pscore, rscore, f1score in zip(fidlist, p.numpy(), r.numpy(), f1.numpy()):
        p_bert[fid] = pscore
        r_bert[fid] = rscore
        f1_bert[fid] = f1score
    pickle.dump(p_bert, open(outpath+'/pbert.pkl', 'wb'))
    pickle.dump(r_bert, open(outpath+'/rbert.pkl', 'wb'))
    pickle.dump(f1_bert, open(outpath+'/f1bert.pkl', 'wb'))
    avg_p = sum(list(p_bert.values()))/len(p_bert)
    avg_r = sum(list(r_bert.values()))/len(r_bert)
    avg_f1 = sum(list(f1_bert.values()))/len(f1_bert)
    
    ret = ('for %s functions\n' % (len(predlist)))
    ret+= 'precision bertscore using official bert-score repo %s\n' % (round(avg_p*100, 2))
    ret+= 'recall bertscore using official bert-score repo %s\n' % (round(avg_r*100, 2))
    ret+= 'f1 bertscore using official bert-score repo %s\n' % (round(avg_f1*100, 2))
    return ret

def sentence_bert_encoding(fidlist, reflist, predlist):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('stsb-roberta-large')
    cosine_score_dict = dict()
    euclidean_distance_dict = dict()

    for fid, ref, pred in zip(fidlist, reflist, predlist):
        ref = ' '.join(ref).strip()
        pred = ' '.join(pred).strip()
        if pred == '':
            cosine_score_dict[fid] = 0
            continue
        data = [ref, pred]
        data_emb = model.encode(data)
        
        css = cosine_similarity_score(data_emb[0].reshape(1, -1), data_emb[1].reshape(1, -1))[0][0]
        ess = euclidean_distance_score(data_emb[0].reshape(1, -1), data_emb[1].reshape(1, -1))[0][0]
        cosine_score_dict[fid] = css
        euclidean_distance_dict[fid] = ess
    pickle.dump(cosine_score_dict, open(outpath+'/sb_cosine.pkl', 'wb'))
    pickle.dump(euclidean_distance_dict, open(outpath+'/sb_euclidean.pkl', 'wb'))
    cosine_score_list = list(cosine_score_dict.values())
    euclidean_distance_list = list(euclidean_distance_dict.values())
    avg_css = sum(cosine_score_list)/len(cosine_score_list)
    avg_ess = sum(euclidean_distance_list)/len(euclidean_distance_list)

    ret = ('for %s functions\n' % (len(predlist)))
    ret+= 'Cosine similarity score with sentence bert encoder embedding is %s\n' % (round(avg_css*100, 2))
    ret+= 'Euclidean distance score with sentence bert encoder embedding is %s\n' % (round(avg_ess*100, 2))
    return ret

def bert_viz_heatmap(ref, pred):
    from bert_score import plot_example
    ref = ' '.join(ref[0]).strip()
    pred = ' '.join(pred[0]).strip()
    plot_example(pred, ref, lang='en', rescale_with_baseline=True)

def attendgru_embedding(fidlist, reflist, predlist):
    import tensorflow as tf
    from tensorflow import keras
    import tokenizer

    flatgru_css_dict = dict()
    flatgru_ess_dict = dict()

    refs = list()
    preds = list()
    for ref, pred in zip(reflist, predlist):
        refs.append('<s> ' + ' '.join(ref).strip() + ' </s>')
        preds.append('<s> ' + ' '.join(pred).strip() + ' </s>')

    comstok = pickle.load(open('coms.tok', 'rb'), encoding='UTF-8')
    fmodelfname = 'attendgru_E01_1612205848.h5'
    fmodel = keras.models.load_model(fmodelfname, custom_objects={"tf":tf, "keras":keras})
    dat_input = fmodel.get_layer('input_1')
    com_input = fmodel.get_layer('input_2')
    tdats_emb = fmodel.get_layer('embedding')
    tdats_gru = fmodel.get_layer('gru')
    dec_emb = fmodel.get_layer('embedding_1')
    dec_gru = fmodel.get_layer('gru_1')
    attn_dot = fmodel.get_layer('dot')
    attn_actv = fmodel.get_layer('activation')
    attn_context = fmodel.get_layer('dot_1')

    reftok = comstok.texts_to_sequences(refs)[:, :13]
    predtok = comstok.texts_to_sequences(preds)[:, :13]
    ref_input = com_input(reftok)
    pred_input = com_input(predtok)
    ref_emb = dec_emb(ref_input)
    ref_gru = dec_gru(ref_emb)
    pred_emb = dec_emb(pred_input)
    pred_gru = dec_gru(pred_emb)
    flatp = tf.keras.layers.Flatten()(pred_gru)
    flatr = tf.keras.layers.Flatten()(ref_gru)
    
    for fid, ref, pred in zip(fidlist, flatr, flatp):
        css = cosine_similarity_score([ref], [pred])[0][0]
        ess = euclidean_distance_score([ref], [pred])[0][0]
        flatgru_css_dict[fid] = css
        flatgru_ess_dict[fid] = ess

    pickle.dump(flatgru_css_dict, open(outpath+'/attendgru_flatgru_cosine.pkl', 'wb'))
    pickle.dump(flatgru_ess_dict, open(outpath+'/attendgru_flatgru_euclidean.pkl', 'wb'))

    flatgru_css_list = list(flatgru_css_dict.values())
    flatgru_ess_list = list(flatgru_ess_dict.values())

    avg_flatgru_css = sum(flatgru_css_list)/len(flatgru_css_list)
    avg_flatgru_ess = sum(flatgru_ess_list)/len(flatgru_ess_list)
    
    ret = ('for %s functions\n' % (len(predlist)))
    ret+= 'Cosine similarity with attendgru flatgru vector %s\n' % (round(avg_flatgru_css*100, 2))
    ret+= 'Euclidean distance with attendgru flatgru vector %s\n' % (round(avg_flatgru_ess*100, 2))
    return ret

def viz_heatmap(similarity_matrix, ref=None, pred=None):
    import matplotlib.pyplot as plt
    import seaborn as sb

    ax = sb.heatmap(similarity_matrix, vmin=-1, vmax=1, linewidths=0.5, annot=True)
    ax.set_xticklabels(pred.split())
    ax.set_yticklabels(ref.split(), rotation=90, horizontalalignment='right', verticalalignment='center')
    plt.show()

def infersent_encoding(fidlist, reflist, predlist):
    import torch
    from infersent.models import InferSent
    model_version = 1
    model_path = "infersent/encoder/infersent%s.pkl" % model_version
    params_model = {'bsize':64, 'word_emb_dim':300, 'enc_lstm_dim':2048, 'pool_type':'max',
                    'dpout_model':0.0, 'version':model_version}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(model_path))
    glove_path = 'infersent/GloVe/glove.840B.300d.txt'
    fasttext_path = 'infersent/fastText/crawl-300d-2M.vec'
    w2v_path = glove_path if model_version == 1 else fasttest_path
    model.set_w2v_path(w2v_path)
    model.build_vocab_k_words(K=100000)
    cosine_score_dict = dict()
    euclidean_distance_dict = dict()

    for fid, ref, pred in zip(fidlist, reflist, predlist):
        ref = ' '.join(ref).strip()
        pred = ' '.join(pred).strip()
        if pred == '':
            cosine_score_dict[fid] = 0
            continue    
        data = [ref, pred]
        data_emb = model.encode(data)
        
        css = cosine_similarity_score(data_emb[0].reshape(1, -1), data_emb[1].reshape(1, -1))[0][0]
        ess = euclidean_distance_score(data_emb[0].reshape(1, -1), data_emb[1].reshape(1, -1))[0][0]
        cosine_score_dict[fid] = css
        euclidean_distance_dict[fid] = ess
    pickle.dump(cosine_score_dict, open(outpath+'/iS_cosine.pkl', 'wb'))
    pickle.dump(euclidean_distance_dict, open(outpath+'/iS_euclidean.pkl', 'wb'))
    cosine_score_list = list(cosine_score_dict.values())
    euclidean_distance_list = list(euclidean_distance_dict.values())
    avg_css = sum(cosine_score_list)/len(cosine_score_list)
    avg_ess = sum(euclidean_distance_list)/len(euclidean_distance_list)

    ret = ('for %s functions\n' % (len(predlist)))
    ret+= 'Cosine similarity score with sentence infersent encoder embedding is %s\n' % (round(avg_css*100, 2))
    ret+= 'Euclidean distance score with sentence infersent encoder embedding is %s\n' % (round(avg_ess*100, 2))
    return ret    

def cosine_similarity_score(x, y):
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_similarity_matrix = cosine_similarity(x, y)
    return cosine_similarity_matrix

def euclidean_distance_score(x, y):
    from sklearn.metrics.pairwise import euclidean_distances
    euclidean_distance_matrix = euclidean_distances(x, y)
    return euclidean_distance_matrix

def prep_dataset2():
    refdict = pickle.load(open(datapath+'/comsdata/ref210.pkl', 'rb'))
    preddict = pickle.load(open(datapath+'/comsdata/attendgru-pred210.pkl', 'rb'))
    refcoms = list()
    predcoms = list()
    fidlist = list()
    preds = dict()
    for fid, com in preddict.items():
        fid = int(fid)
        com = com.split()
        com = fil(com)
        preds[fid] = com

    for fid, com in refdict.items():
        fid = int(fid)
        com = com.split()
        com = fil(com)
        if len(com) < 1:
            continue
        try:
            predcoms.append(preds[fid])
        except:
            continue
        refcoms.append(com)
        fidlist.append(fid)
    return fidlist, refcoms, predcoms

if __name__ == '__main__':
    fidlist, refcoms, predcoms = prep_dataset2()
    parser = argparse.ArgumentParser(description='allowed metrics are: jaccard, bleu, rouge, meteor, bertscore, tfidf, use, sbert, infersent, attendgru')
    parser.add_argument('--metric', dest='metric', type=str, default='use')
    args = parser.parse_args()
    metric = args.metric

    metric_list = ['jaccard', 'bleu', 'rouge', 'meteor', 'bertscore', 'tfidf', 'use', 'sbert', 'infersent', 'attendgru']
    if metric not in metric_list:
        print('{} not an available metric yet. Please use one of the following metrics: {}'.format(metric, [m for m in metric_list]))

    if metric=='jaccard':
        # Jaccard Similarity
        js = jaccard_similarity_score(fidlist, refcoms, predcoms, average='unweighted')
        print(js)
    elif metric=='bleu':
        # Bleu Score
        bs = all_bleu_score(refcoms, predcoms)
        print(bs)
    elif metric=='rouge':
        # Rouge Score
        rs = all_rouge_score(refcoms, predcoms)
        print(rs)
    elif metric=='meteor':
        # Meteor Score
        ms = all_meteor_score(refcoms, predcoms)
        print(ms)
    elif metric=='bertscore':
        # Bert Score    
        officialBertScore = official_bert_score(fidlist, refcoms, predcoms)
        print(officialBertScore)
    elif metric=='tfidf':
        # Tf-IDF
        tfidf_css = tfidf_vectorizer(fidlist, refcoms, predcoms)
        print(tfidf_css)
    elif metric=='use':
        # Universal Sentence Encoder
        use_css = all_universal_sentence_encoder_dict(fidlist, refcoms, predcoms)
        print(use_css)
    elif metric=='sbert':
        # Sentence Bert
        sent_bert_css = sentence_bert_encoding(fidlist, refcoms, predcoms)
        print(sent_bert_css)
    elif metric=='infersent':
        # InferSent
        infersent_css = infersent_encoding(fidlist, refcoms, predcoms)
        print(infersent_css)
    elif metric=='attendgru':
        # Attendgru
        attendgru_tdats = attendgru_embedding(fidlist, refcoms, predcoms)
        print(attendgru_tdats)