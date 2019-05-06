# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import io
from logging import getLogger
import numpy as np
import torch
from scipy.stats import spearmanr


MONOLINGUAL_EVAL_PATH = 'data/monolingual'
SEMEVAL17_EVAL_PATH = 'data/crosslingual/wordsim'


logger = getLogger()


def get_word_pairs(path, lower=True):
    """
    Return a list of (word1, word2, score) tuples from a word similarity file.
    """
    assert os.path.isfile(path) and type(lower) is bool
    word_pairs = []
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            line = line.lower() if lower else line
            line = line.split()
            # ignore phrases, only consider words
            if len(line) != 3:
                assert len(line) > 3
                assert 'SEMEVAL17' in os.path.basename(path) or 'EN-IT_MWS353' in path
                continue
            word_pairs.append((line[0], line[1], float(line[2])))
    return word_pairs


def get_word_id(word, word2id, lower):
    """
    Get a word ID.
    If the model does not use lowercase and the evaluation file is lowercased,
    we might be able to find an associated word.
    """
    assert type(lower) is bool
    word_id = word2id.get(word)
    if word_id is None and not lower:
        word_id = word2id.get(word.capitalize())
    if word_id is None and not lower:
        word_id = word2id.get(word.title())
    return word_id


def get_spearman_rho(word2id1, embeddings1, path, lower,
                     word2id2=None, embeddings2=None):
    """
    Compute monolingual or cross-lingual word similarity score.
    """
    assert not ((word2id2 is None) ^ (embeddings2 is None))
    word2id2 = word2id1 if word2id2 is None else word2id2
    embeddings2 = embeddings1 if embeddings2 is None else embeddings2
    assert len(word2id1) == embeddings1.shape[0]
    assert len(word2id2) == embeddings2.shape[0]
    assert type(lower) is bool
    word_pairs = get_word_pairs(path)
    not_found = 0
    pred = []
    gold = []
    for word1, word2, similarity in word_pairs:
        id1 = get_word_id(word1, word2id1, lower)
        id2 = get_word_id(word2, word2id2, lower)
        if id1 is None or id2 is None:
            not_found += 1
            continue
        u = embeddings1[id1]
        v = embeddings2[id2]
        score = u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))
        gold.append(similarity)
        pred.append(score)
    return spearmanr(gold, pred).correlation, len(gold), not_found


def get_wordsim_scores(language, word2id, embeddings, lower=True):
    """
    Return monolingual word similarity scores.
    """
    dirpath = os.path.join(MONOLINGUAL_EVAL_PATH, language)
    if not os.path.isdir(dirpath):
        return None

    scores = {}
    separator = "=" * (30 + 1 + 10 + 1 + 13 + 1 + 12)
    pattern = "%30s %10s %13s %12s"
    logger.info(separator)
    logger.info(pattern % ("Dataset", "Found", "Not found", "Rho"))
    logger.info(separator)

    for filename in list(os.listdir(dirpath)):
        if filename.startswith('%s_' % (language.upper())):
            filepath = os.path.join(dirpath, filename)
            coeff, found, not_found = get_spearman_rho(word2id, embeddings, filepath, lower)
            logger.info(pattern % (filename[:-4], str(found), str(not_found), "%.4f" % coeff))
            scores[filename[:-4]] = coeff
    logger.info(separator)

    return scores


def get_wordanalogy_scores(language, word2id, embeddings, lower=True):
    """
    Return monolingual word analogy scores.
    """
    filepath = os.path.join(MONOLINGUAL_EVAL_PATH, language, 'questions-words.txt')
    if not os.path.exists(filepath):
        return None

    # normalize word embeddings
    embeddings = embeddings / np.sqrt((embeddings ** 2).sum(1))[:, None]

    # scores by category
    scores = {}

    word_ids = {}
    queries = {}

    with io.open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # new line
            line = line.rstrip()
            if lower:
                line = line.lower()

            # new category
            if ":" in line:
                assert line[1] == ' '
                category = line[2:]
                assert category not in scores
                scores[category] = {'n_found': 0, 'n_not_found': 0, 'n_correct': 0}
                word_ids[category] = []
                queries[category] = []
                continue

            # get word IDs
            assert len(line.split()) == 4, line
            word1, word2, word3, word4 = line.split()
            word_id1 = get_word_id(word1, word2id, lower)
            word_id2 = get_word_id(word2, word2id, lower)
            word_id3 = get_word_id(word3, word2id, lower)
            word_id4 = get_word_id(word4, word2id, lower)

            # if at least one word is not found
            if any(x is None for x in [word_id1, word_id2, word_id3, word_id4]):
                scores[category]['n_not_found'] += 1
                continue
            else:
                scores[category]['n_found'] += 1
                word_ids[category].append([word_id1, word_id2, word_id3, word_id4])
                # generate query vector and get nearest neighbors
                query = embeddings[word_id1] - embeddings[word_id2] + embeddings[word_id4]
                query = query / np.linalg.norm(query)

                queries[category].append(query)

    # Compute score for each category
    for cat in queries:
        qs = torch.from_numpy(np.vstack(queries[cat]))
        keys = torch.from_numpy(embeddings.T)
        values = qs.mm(keys).cpu().numpy()

    # be sure we do not select input words
        for i, ws in enumerate(word_ids[cat]):
            for wid in [ws[0], ws[1], ws[3]]:
                values[i, wid] = -1e9
        scores[cat]['n_correct'] = np.sum(values.argmax(axis=1) == [ws[2] for ws in word_ids[cat]])

    # pretty print
    separator = "=" * (30 + 1 + 10 + 1 + 13 + 1 + 12)
    pattern = "%30s %10s %13s %12s"
    logger.info(separator)
    logger.info(pattern % ("Category", "Found", "Not found", "Accuracy"))
    logger.info(separator)

    # compute and log accuracies
    accuracies = {}
    for k in sorted(scores.keys()):
        v = scores[k]
        accuracies[k] = float(v['n_correct']) / max(v['n_found'], 1)
        logger.info(pattern % (k, str(v['n_found']), str(v['n_not_found']), "%.4f" % accuracies[k]))
    logger.info(separator)

    return accuracies


def get_crosslingual_wordsim_scores(lang1, word2id1, embeddings1,
                                    lang2, word2id2, embeddings2, lower=True):
    """
    Return cross-lingual word similarity scores.
    """
    f1 = os.path.join(SEMEVAL17_EVAL_PATH, '%s-%s-SEMEVAL17.txt' % (lang1, lang2))
    f2 = os.path.join(SEMEVAL17_EVAL_PATH, '%s-%s-SEMEVAL17.txt' % (lang2, lang1))
    if not (os.path.exists(f1) or os.path.exists(f2)):
        return None

    if os.path.exists(f1):
        coeff, found, not_found = get_spearman_rho(
            word2id1, embeddings1, f1,
            lower, word2id2, embeddings2
        )
    elif os.path.exists(f2):
        coeff, found, not_found = get_spearman_rho(
            word2id2, embeddings2, f2,
            lower, word2id1, embeddings1
        )

    scores = {}
    separator = "=" * (30 + 1 + 10 + 1 + 13 + 1 + 12)
    pattern = "%30s %10s %13s %12s"
    logger.info(separator)
    logger.info(pattern % ("Dataset", "Found", "Not found", "Rho"))
    logger.info(separator)

    task_name = '%s_%s_SEMEVAL17' % (lang1.upper(), lang2.upper())
    logger.info(pattern % (task_name, str(found), str(not_found), "%.4f" % coeff))
    scores[task_name] = coeff
    if not scores:
        return None
    logger.info(separator)

    return scores
