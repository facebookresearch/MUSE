# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

en_analogy='https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip'
aws_path='https://s3.amazonaws.com/arrival'
semeval_2017='http://alt.qcri.org/semeval2017/task2/data/uploads'
europarl='http://www.statmt.org/europarl/v7'

declare -A wordsim_lg
wordsim_lg=(["en"]="EN_MC-30.txt EN_MTurk-287.txt EN_RG-65.txt EN_VERB-143.txt EN_WS-353-REL.txt EN_YP-130.txt EN_MEN-TR-3k.txt EN_MTurk-771.txt EN_RW-STANFORD.txt EN_SIMLEX-999.txt EN_WS-353-ALL.txt EN_WS-353-SIM.txt" ["es"]="ES_MC-30.txt ES_RG-65.txt ES_WS-353.txt" ["de"]="DE_GUR350.txt DE_GUR65.txt DE_SIMLEX-999.txt DE_WS-353.txt DE_ZG222.txt" ["fr"]="FR_RG-65.txt" ["it"]="IT_SIMLEX-999.txt IT_WS-353.txt")

mkdir monolingual crosslingual

## English word analogy task
curl -Lo source-archive.zip $en_analogy
mkdir -p monolingual/en/
unzip -p source-archive.zip word2vec/trunk/questions-words.txt > monolingual/en/questions-words.txt
rm source-archive.zip


## Downloading en-{} or {}-en dictionaries
lgs="af ar bg bn bs ca cs da de el en es et fa fi fr he hi hr hu id it ja ko lt lv mk ms nl no pl pt ro ru sk sl sq sv ta th tl tr uk vi zh"
mkdir -p crosslingual/dictionaries/
for lg in ${lgs}
do
  for suffix in .txt .0-5000.txt .5000-6500.txt
  do
    fname=en-$lg$suffix
    curl -Lo crosslingual/dictionaries/$fname $aws_path/dictionaries/$fname
    fname=$lg-en$suffix
    curl -Lo crosslingual/dictionaries/$fname $aws_path/dictionaries/$fname
  done
done

## Download European dictionaries
for src_lg in de es fr it pt
do
  for tgt_lg in de es fr it pt
  do
    if [ $src_lg != $tgt_lg ]
    then
      for suffix in .txt .0-5000.txt .5000-6500.txt
      do
        fname=$src_lg-$tgt_lg$suffix
        curl -Lo crosslingual/dictionaries/$fname $aws_path/dictionaries/european/$fname
      done
    fi
  done
done

## Download Dinu et al. dictionaries
for fname in OPUS_en_it_europarl_train_5K.txt OPUS_en_it_europarl_test.txt
do
    echo $fname
    curl -Lo crosslingual/dictionaries/$fname $aws_path/dictionaries/$fname
done

## Monolingual wordsim tasks
for lang in "${!wordsim_lg[@]}"
do
  echo $lang
  mkdir monolingual/$lang
  for wsim in ${wordsim_lg[$lang]}
  do
    echo $wsim
    curl -Lo monolingual/$lang/$wsim $aws_path/$lang/$wsim
  done
done

## SemEval 2017 monolingual and cross-lingual wordsim tasks
# 1) Task1: monolingual
curl -Lo semeval2017-task2.zip $semeval_2017/semeval2017-task2.zip
unzip semeval2017-task2.zip

fdir='SemEval17-Task2/test/subtask1-monolingual'
for lang in en es de fa it
do
  mkdir -p monolingual/$lang
  uplang=`echo $lang | awk '{print toupper($0)}'`
  paste $fdir/data/$lang.test.data.txt $fdir/keys/$lang.test.gold.txt > monolingual/$lang/${uplang}_SEMEVAL17.txt
done

# 2) Task2: cross-lingual
mkdir -p crosslingual/wordsim
fdir='SemEval17-Task2/test/subtask2-crosslingual'
for lg_pair in de-es de-fa de-it en-de en-es en-fa en-it es-fa es-it it-fa
do
  echo $lg_pair
  paste $fdir/data/$lg_pair.test.data.txt $fdir/keys/$lg_pair.test.gold.txt > crosslingual/wordsim/$lg_pair-SEMEVAL17.txt
done
rm semeval2017-task2.zip
rm -r SemEval17-Task2/

## Europarl for sentence retrieval
# TODO: set to true to activate download of Europarl (slow)
if false; then
  mkdir -p crosslingual/europarl
  # Tokenize EUROPARL with MOSES
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  git clone https://github.com/moses-smt/mosesdecoder.git
  SCRIPTS=mosesdecoder/scripts
  TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl

  for lg_pair in it-en  # es-en etc
  do
    curl -Lo $lg_pair.tgz $europarl/$lg_pair.tgz
    tar -xvf $lg_pair.tgz
    rm $lg_pair.tgz
    lgs=(${lg_pair//-/ })
    for lg in ${lgs[0]} ${lgs[1]}
    do
      cat europarl-v7.$lg_pair.$lg | $TOKENIZER -threads 8 -l $lg -no-escape > euro.$lg.txt
      rm europarl-v7.$lg_pair.$lg
    done

    paste euro.${lgs[0]}.txt euro.${lgs[1]}.txt | shuf > euro.paste.txt
    rm euro.${lgs[0]}.txt euro.${lgs[1]}.txt

    cut -f1 euro.paste.txt > crosslingual/europarl/europarl-v7.$lg_pair.${lgs[0]}
    cut -f2 euro.paste.txt > crosslingual/europarl/europarl-v7.$lg_pair.${lgs[1]}
    rm euro.paste.txt
  done

  rm -rf mosesdecoder
fi
