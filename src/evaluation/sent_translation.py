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

from src.utils import bow_idf, get_nn_avg_dist


EUROPARL_DIR = 'data/crosslingual/europarl'


logger = getLogger()


def load_europarl_data(lg1, lg2, n_max=1e10, lower=True):
    """
    Load data parallel sentences
    """
    if not (os.path.isfile(os.path.join(EUROPARL_DIR, 'europarl-v7.%s-%s.%s' % (lg1, lg2, lg1))) or
            os.path.isfile(os.path.join(EUROPARL_DIR, 'europarl-v7.%s-%s.%s' % (lg2, lg1, lg1)))):
        return None

    if os.path.isfile(os.path.join(EUROPARL_DIR, 'europarl-v7.%s-%s.%s' % (lg2, lg1, lg1))):
        lg1, lg2 = lg2, lg1

    # load sentences
    data = {lg1: [], lg2: []}
    for lg in [lg1, lg2]:
        fname = os.path.join(EUROPARL_DIR, 'europarl-v7.%s-%s.%s' % (lg1, lg2, lg))

        with io.open(fname, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n_max:
                    break
                line = line.lower() if lower else line
                data[lg].append(line.rstrip().split())

    # get only unique sentences for each language
    assert len(data[lg1]) == len(data[lg2])
    data[lg1] = np.array(data[lg1])
    data[lg2] = np.array(data[lg2])
    data[lg1], indices = np.unique(data[lg1], return_index=True)
    data[lg2] = data[lg2][indices]
    data[lg2], indices = np.unique(data[lg2], return_index=True)
    data[lg1] = data[lg1][indices]

    # shuffle sentences
    rng = np.random.RandomState(1234)
    perm = rng.permutation(len(data[lg1]))
    data[lg1] = data[lg1][perm]
    data[lg2] = data[lg2][perm]

    logger.info("Loaded europarl %s-%s (%i sentences)." % (lg1, lg2, len(data[lg1])))
    return data


def get_sent_translation_accuracy(data, lg1, word2id1, emb1, lg2, word2id2, emb2,
                                  n_keys, n_queries, method, idf):

    """
    Given parallel sentences from Europarl, evaluate the
    sentence translation accuracy using the precision@k.
    """
    # get word vectors dictionaries
    emb1 = emb1.cpu().numpy()
    emb2 = emb2.cpu().numpy()
    word_vec1 = dict([(w, emb1[word2id1[w]]) for w in word2id1])
    word_vec2 = dict([(w, emb2[word2id2[w]]) for w in word2id2])
    word_vect = {lg1: word_vec1, lg2: word_vec2}
    lg_keys = lg2
    lg_query = lg1

    # get n_keys pairs of sentences
    keys = data[lg_keys][:n_keys]
    keys = bow_idf(keys, word_vect[lg_keys], idf_dict=idf[lg_keys])

    # get n_queries query pairs from these n_keys pairs
    rng = np.random.RandomState(1234)
    idx_query = rng.choice(range(n_keys), size=n_queries, replace=False)
    queries = data[lg_query][idx_query]
    queries = bow_idf(queries, word_vect[lg_query], idf_dict=idf[lg_query])

    # normalize embeddings
    queries = torch.from_numpy(queries).float()
    queries = queries / queries.norm(2, 1, keepdim=True).expand_as(queries)
    keys = torch.from_numpy(keys).float()
    keys = keys / keys.norm(2, 1, keepdim=True).expand_as(keys)

    # nearest neighbors
    if method == 'nn':
        scores = keys.mm(queries.transpose(0, 1)).transpose(0, 1)
        scores = scores.cpu()

    # inverted softmax
    elif method.startswith('invsm_beta_'):
        beta = float(method[len('invsm_beta_'):])
        scores = keys.mm(queries.transpose(0, 1)).transpose(0, 1)
        scores.mul_(beta).exp_()
        scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
        scores = scores.cpu()

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist_keys = torch.from_numpy(get_nn_avg_dist(queries, keys, knn))
        average_dist_queries = torch.from_numpy(get_nn_avg_dist(keys, queries, knn))
        # scores
        scores = keys.mm(queries.transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist_queries[:, None].float() + average_dist_keys[None, :].float())
        scores = scores.cpu()

    results = []
    top_matches = scores.topk(10, 1, True)[1]
    for k in [1, 5, 10]:
        top_k_matches = (top_matches[:, :k] == torch.from_numpy(idx_query)[:, None]).sum(1)
        precision_at_k = 100 * top_k_matches.float().numpy().mean()
        logger.info("%i queries (%s) - %s - Precision at k = %i: %f" %
                    (len(top_k_matches), lg_query.upper(), method, k, precision_at_k))
        results.append(('sent-precision_at_%i' % k, precision_at_k))

    return results
