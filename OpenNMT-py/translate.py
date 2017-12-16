#!/usr/bin/env python
# python OpenNMT-py/translate.py -model models/autoencoder.pt -src data/test_nl -output pred_autoencoder.txt -replace_unk -verbose
# python OpenNMT-py/translate.py -model models/translator-en-nl.pt -src data/test_en -output pred_translator.txt -replace_unk -verbose

from __future__ import division
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from builtins import bytes
import os
import argparse
import math
import codecs
import torch

import onmt
import onmt.IO
import opts
from itertools import takewhile, count
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()


def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total/words_total)))


def get_src_words(src_indices, index2str):
    words = []
    raw_words = (index2str[i] for i in src_indices)
    words = takewhile(lambda w: w != onmt.IO.PAD_WORD, raw_words)
    return " ".join(words)


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    print(opt.model)
    translator = onmt.Translator(opt, dummy_opt.__dict__)

    ##### CUSTOM CODE
    opt.model = 'models/autoencoder.pt'
    autoencoder = onmt.Translator(opt, dummy_opt.__dict__)

    out_file = codecs.open(opt.output, 'w', 'utf-8')
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()
        autoencoder.initBeamAccum()
    data = onmt.IO.ONMTDataset(
        opt.src, opt.tgt, translator.fields,
        use_filter_pred=False)

    test_data = onmt.IO.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        shuffle=False)

    counter = count(1)

    for batch in test_data:
        pred_batch, gold_batch, pred_scores, gold_scores, attn, src \
            = translator.translate(batch, data)

        pred_score_total += sum(score[0] for score in pred_scores)
        pred_words_total += sum(len(x[0]) for x in pred_batch)
        if opt.tgt:
            gold_score_total += sum(gold_scores)
            gold_words_total += sum(len(x) for x in batch.tgt[1:])

        # z_batch: an iterator over the predictions, their scores,
        # the gold sentence, its score, and the source sentence for each
        # sentence in the batch. It has to be zip_longest instead of
        # plain-old zip because the gold_batch has length 0 if the target
        # is not included.
        z_batch = zip_longest(
                pred_batch, gold_batch,
                pred_scores, gold_scores,
                (sent.squeeze(1) for sent in src.split(1, dim=1)))

        for pred_sents, gold_sent, pred_score, gold_score, src_sent in z_batch:
            n_best_preds = [" ".join(pred) for pred in pred_sents[:opt.n_best]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

            if opt.verbose:
                sent_number = next(counter)
                words = get_src_words(
                    src_sent, translator.fields["src"].vocab.itos)

                os.write(1, bytes('\nSENT %d: %s\n' %
                                  (sent_number, words), 'UTF-8'))

                best_pred = n_best_preds[0]
                best_score = pred_score[0]
                os.write(1, bytes('PRED %d: %s\n' %
                                  (sent_number, best_pred), 'UTF-8'))
                print("PRED SCORE: %.4f" % best_score)

                if opt.tgt:
                    tgt_sent = ' '.join(gold_sent)
                    os.write(1, bytes('GOLD %d: %s\n' %
                             (sent_number, tgt_sent), 'UTF-8'))
                    print("GOLD SCORE: %.4f" % gold_score)

                if len(n_best_preds) > 1:
                    print('\nBEST HYP:')
                    for score, sent in zip(pred_score, n_best_preds):
                        os.write(1, bytes("[%.4f] %s\n" % (score, sent),
                                 'UTF-8'))

    report_score('PRED', pred_score_total, pred_words_total)
    if opt.tgt:
        report_score('GOLD', gold_score_total, gold_words_total)

    if opt.dump_beam:
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))

def in_main_for_words():
    # Define list of words
    # NL-EN Cross-Model Embeddings
    nl_en_same_nl = ['toespraak', 'urine', 'maar', 'missie', 'plant', 'boot', 'sport', 'lamp', 'wereld', 'water']
    nl_en_same_en = ['speech', 'urine', 'but', 'mission', 'plant', 'boat', 'sport', 'lamp', 'world', 'water']

    nl_en_diff_nl = ['vis', 'uur', 'het', 'wetenschap', 'mensen', 'primaat', 'richting', 'slaap', 'kleingeld', 'inbox']
    nl_en_diff_en = ['bee', 'blind', 'day', 'homeless', 'street', 'fish', 'hour', 'it', 'science', 'people']

    nl_en_similar_nl = ['hond','auto','kinderen','neef','arts','laptop','toerist','muziek','wiskundige','spijkerbroek']
    nl_en_similar_en = ['cat','bus','grandchildren','niece','surgeon','computer','tourism','jazz','physicist','sweater']

    # NL-NL Autoencoder Embeddings
    nl_nl_diff_1 = ['soortgelijks', 'spotgoedkoop', 'nazaten', 'marktrevolutie', 'variatie', 'fouten', 'röntgenfoto', 'experts', 'schaalvoordeel', 'verbazingwekkend']
    nl_nl_diff_2 = ['diamanten', 'epicentrum', 'weergaven', 'Keulse', 'wervelkolom', 'katrol', 'gekroond', 'Heiligheid', 'Schotse', 'Waarvan']

    nl_nl_same_1 = ['nodig', 'gelijk', 'afval', 'aanhouden', 'subject', 'designer', 'nerveus', 'sekse', 'steen', 'overgeven']
    nl_nl_same_2 = ['noodzakelijk', 'hetzelfde', 'vuil', 'blijven', 'onderwerp', 'ontwerper', 'gespannen', 'geslacht', 'rots', 'braken']

    nl_nl_similar_1 = ['hond','auto','kinderen','neef','arts','laptop','toerist','muziek','wiskundige','spijkerbroek']
    nl_nl_similar_2 = ['kat','bus','kleinkinderen','nicht','chirurg','computer','toerisme','jazz','natuurkundige','trui']

    # EN-EN Translator Embeddings

    en_en_diff_1 = ['fulfilled', 'schmuck', 'scrutinized', 'Foresight', 'Circle', 'interpret', 'multitask', '30', 'conspiring', 'Amerasians']
    en_en_diff_2 = ['Geraldine', 'propolis', 'geographies', 'race', 'bluff', 'Tighter', 'most', 'one-life', 'Westerner', 'X-Files']

    en_en_same_1 = ['cubicles', 'visitors', 'immediate', 'rope', 'shrub', 'outcome', 'rest', 'compute', 'draw']
    en_en_same_2 = ['offices', 'guests', 'instantaneous', 'cord', 'bush', 'result', 'relax', 'calculate', 'sketch']

    en_en_similar_1 = ['cat','bus','grandchildren','niece','surgeon','computer','tourism','jazz','physicist','sweater']
    en_en_similar_2 = ['dog','car','children','cousin','physician','laptop','tourist','music','mathematician','jeans']

    print('----- NL-NL -----')
    print('Same words')
    print_similarity(autoencoder, autoencoder, nl_nl_same_1, nl_nl_same_2)

    print('\nSimilar words')
    print_similarity(autoencoder, autoencoder, nl_nl_similar_1, nl_nl_similar_2)

    print('\nDifferent words')
    print_similarity(autoencoder, autoencoder, nl_nl_diff_1, nl_nl_diff_2)

    print('\n\n----- EN-EN -----')
    print('Same words')
    print_similarity(translator, translator, en_en_same_1, en_en_same_2)

    print('\nSimilar words')
    print_similarity(translator, translator, en_en_similar_1, en_en_similar_2)

    print('\nDifferent words')
    print_similarity(translator, translator, en_en_diff_1, en_en_diff_2)

    print('\n\n----- NL-EN -----')
    print('Same words')
    print_similarity(autoencoder, translator, nl_en_same_nl, nl_en_same_en)

    print('\nSimilar words')
    print_similarity(autoencoder, translator, nl_en_similar_nl, nl_en_similar_en)

    print('\nDifferent words')
    print_similarity(autoencoder, translator, nl_en_diff_nl, nl_en_diff_en)


def print_similarity(model1, model2, words1, words2):
    # Transform list of words to list of indices
    words1_idx = [model1.fields['src'].vocab.stoi[word] for word in words1]
    words2_idx = [model2.fields['src'].vocab.stoi[word] for word in words2]
    if 0 in words1_idx or 0 in words2_idx:
        print(words1_idx)
        print(words2_idx)
        raise ValueError('Een van de woorden niet in vocab!')

    # Fetch corresponding word embeddings
    model1_embeddings = model1.model.encoder.state_dict()['embeddings.make_embedding.emb_luts.0.weight'].numpy()
    model2_embeddings = model2.model.encoder.state_dict()['embeddings.make_embedding.emb_luts.0.weight'].numpy()

    i = 0
    sims = []
    for w1_idx, w2_idx in zip(words1_idx, words2_idx):
        w1_embedding = model1_embeddings[w1_idx,:].reshape(1,-1)
        w2_embedding = model2_embeddings[w2_idx,:].reshape(1,-1)
        sim = cosine_similarity(w1_embedding, w2_embedding)[0][0]
        print(' - ({},{}) = {}'.format(words1[i], words2[i], str(sim)[0:5]))
        i += 1
        sims.append(sim)
    print(' -> AVR: {}, STD: {}'.format(str(np.mean(sims)),str(np.std(sims))[0:4]))

if __name__ == "__main__":
    main()
