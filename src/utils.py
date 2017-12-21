# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import sys
import pickle
import random
import inspect
import argparse
import subprocess
import numpy as np
import torch
from torch import optim
from logging import getLogger

from .logger import create_logger
from .dictionary import Dictionary


MAIN_DUMP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dumped')


logger = getLogger()


# load Faiss if available (dramatically accelerates the nearest neighbor search)
try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False


def initialize_exp(params):
    """
    Initialize experiment.
    """
    # initialization
    if getattr(params, 'seed', -1) >= 0:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        if params.cuda:
            torch.cuda.manual_seed(params.seed)

    # dump parameters
    params.exp_path = get_exp_path(params) if not params.exp_path else params.exp_path
    pickle.dump(params, open(os.path.join(params.exp_path, 'params.pkl'), 'wb'))

    # create logger
    logger = create_logger(os.path.join(params.exp_path, 'train.log'), vb=params.verbose)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s' % params.exp_path)
    return logger


def bow(sentences, word_vec, normalize=False):
    """
    Get sentence representations using average bag-of-words.
    """
    embeddings = []
    for sent in sentences:
        sentvec = [word_vec[w] for w in sent if w in word_vec]
        if normalize:
            sentvec = [v / np.linalg.norm(v) for v in sentvec]
        if len(sentvec) == 0:
            sentvec = [word_vec[list(word_vec.keys())[0]]]
        embeddings.append(np.mean(sentvec, axis=0))
    return np.vstack(embeddings)


def bow_idf(sentences, word_vec, idf_dict=None):
    """
    Get sentence representations using weigthed IDF bag-of-words.
    """
    embeddings = []
    for sent in sentences:
        sent = set(sent)
        list_words = [w for w in sent if w in word_vec and w in idf_dict]
        if len(list_words) > 0:
            sentvec = [word_vec[w] * idf_dict[w] for w in list_words]
            sentvec = sentvec / np.sum([idf_dict[w] for w in list_words])
        else:
            sentvec = [word_vec[list(word_vec.keys())[0]]]
        embeddings.append(np.sum(sentvec, axis=0))
    return np.vstack(embeddings)


def get_idf(europarl, src_lg, tgt_lg, n_idf):
    """
    Compute IDF values.
    """
    idf = {src_lg: {}, tgt_lg: {}}
    k = 0
    for lg in idf:
        start_idx = 200000 + k * n_idf
        end_idx = 200000 + (k + 1) * n_idf
        for sent in europarl[lg][start_idx:end_idx]:
            for word in set(sent):
                idf[lg][word] = idf[lg].get(word, 0) + 1
        n_doc = len(europarl[lg][start_idx:end_idx])
        for word in idf[lg]:
            idf[lg][word] = max(1, np.log10(n_doc / (idf[lg][word])))
        k += 1
    return idf


def read_embeddings(path, dim=None, n_max=1e9):
    """
    Read all words from a word embedding file, and optionally filter them.
    """
    word2id = {}
    embeddings = []
    with open(path, 'r') as f:
        line = f.readline()
        dim = int(line.split(' ', 1)[1])
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                logger.warning('Word "%s" has several embeddings!' % word)
                continue
            word2id[word] = len(word2id)
            embeddings.append(np.fromstring(vec, sep=' '))
            if len(word2id) == n_max:
                break
    embeddings = np.array(embeddings, dtype=np.float32)
    embeddings = embeddings / np.sqrt((embeddings ** 2).sum(1))[:, None]
    logger.info("Found %s word vectors of size %s" % (len(word2id), dim))
    return word2id, embeddings


def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(query, knn)
        return distances.mean(1)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.numpy()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


def get_exp_path(params):
    """
    Create a directory to store the experiment.
    """
    # create the main dump path if it does not exist
    if not os.path.exists(MAIN_DUMP_PATH):
        subprocess.Popen("mkdir %s" % MAIN_DUMP_PATH, shell=True).wait()
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    while True:
        exp_name = ''.join(random.choice(chars) for _ in range(10))
        exp_path = os.path.join(MAIN_DUMP_PATH, exp_name)
        if not os.path.isdir(exp_path):
            break
    # create the dump folder
    if not os.path.isdir(exp_path):
        subprocess.Popen("mkdir %s" % exp_path, shell=True).wait()
    return exp_path


def clip_parameters(model, clip):
    """
    Clip model weights.
    """
    if clip > 0:
        for x in model.parameters():
            x.data.clamp_(-clip, clip)


def load_external_embeddings(params, source):
    """
    Reload pretrained embeddings from a text file.
    """
    assert type(source) is bool
    word2id = {}
    vectors = []

    # load pretrained embeddings
    lang = params.src_lang if source else params.tgt_lang
    emb_path = params.src_emb if source else params.tgt_emb
    _emb_dim_file = params.emb_dim
    with open(emb_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert _emb_dim_file == int(split[1])
            else:
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                assert word not in word2id
                assert vect.shape == (_emb_dim_file,), i
                word2id[word] = len(word2id)
                vectors.append(vect[None])
            if params.max_vocab > 0 and i >= params.max_vocab:
                break

    logger.info("Loaded %i pre-trained word embeddings" % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)
    embeddings = np.concatenate(vectors, 0)
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cuda() if params.cuda else embeddings
    assert embeddings.size() == (len(word2id), params.emb_dim), ((len(word2id), params.emb_dim, embeddings.size()))

    return dico, embeddings


def normalize_embeddings(emb, types):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            emb.sub_(emb.mean(1, keepdim=True).expand_as(emb))
        elif t == 'renorm':
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))
        else:
            raise Exception('Unknown normalization type: "%s"' % t)


def export_embeddings(src_emb, tgt_emb, params):
    """
    Export embeddings to a text file.
    """
    src_id2word = params.src_dico.id2word
    tgt_id2word = params.tgt_dico.id2word
    n_src = len(src_id2word)
    n_tgt = len(tgt_id2word)
    dim = src_emb.shape[1]
    src_path = os.path.join(params.exp_path, 'vectors-%s.txt' % params.src_lang)
    tgt_path = os.path.join(params.exp_path, 'vectors-%s.txt' % params.tgt_lang)
    # source embeddings
    logger.info('Writing source embeddings to %s ...' % src_path)
    with open(src_path, 'w') as f:
        f.write("%i %i\n" % (n_src, dim))
        for i in range(len(src_id2word)):
            f.write("%s %s\n" % (src_id2word[i], " ".join(str(x) for x in src_emb[i])))
            # target embeddings
    logger.info('Writing target embeddings to %s ...' % tgt_path)
    with open(tgt_path, 'w') as f:
        f.write("%i %i\n" % (n_tgt, dim))
        for i in range(len(tgt_id2word)):
            f.write("%s %s\n" % (tgt_id2word[i], " ".join(str(x) for x in tgt_emb[i])))
