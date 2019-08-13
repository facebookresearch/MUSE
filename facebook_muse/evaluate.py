# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# python evaluate.py --crosslingual --src_lang en --tgt_lang es --src_emb data/wiki.en-es.en.vec --tgt_emb data/wiki.en-es.es.vec

import os
import argparse
from collections import OrderedDict

from facebook_muse.utils import bool_flag, initialize_exp
from facebook_muse.models import build_model
from facebook_muse.trainer import Trainer
from facebook_muse.evaluation import Evaluator

# main
def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
    # data
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
    parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
    # reload pre-trained embeddings
    parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
    parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
    parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")


    # parse parameters
    params = parser.parse_args()

    # check parameters
    assert params.src_lang, "source language undefined"
    assert os.path.isfile(params.src_emb)
    assert not params.tgt_lang or os.path.isfile(params.tgt_emb)
    assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)

    # build logger / model / trainer / evaluator
    logger = initialize_exp(params)
    src_emb, tgt_emb, mapping, _ = build_model(params, False)
    trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
    evaluator = Evaluator(trainer)

    # run evaluations
    to_log = OrderedDict({'n_iter': 0})
    evaluator.monolingual_wordsim(to_log)
    # evaluator.monolingual_wordanalogy(to_log)
    if params.tgt_lang:
        evaluator.crosslingual_wordsim(to_log)
        evaluator.word_translation(to_log)
        evaluator.sent_translation(to_log)
        # evaluator.dist_mean_cosine(to_log)

if __name__ == "__main__":
    main()
