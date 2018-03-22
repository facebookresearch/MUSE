## MUSE: Multilingual Unsupervised and Supervised Embeddings
![Model](https://s3.amazonaws.com/arrival/outline_all.png)

MUSE is a Python library for *multilingual word embeddings*, whose goal is to provide the community with:
* state-of-the-art multilingual word embeddings based on [fastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
* large-scale high-quality bilingual dictionaries for training and evaluation

We include two methods, one *supervised* that uses a bilingual dictionary or identical character strings, and one *unsupervised* that does not use any parallel data (see [Word Translation without Parallel Data](https://arxiv.org/pdf/1710.04087.pdf) for more details).

## Dependencies
* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)
* [PyTorch](http://pytorch.org/)
* [Faiss](https://github.com/facebookresearch/faiss) (recommended) for fast nearest neighbor search (CPU or GPU).

MUSE is available on CPU or GPU, in Python 2 or 3. Faiss is *optional* for GPU users - though Faiss-GPU will greatly speed up nearest neighbor search - and *highly recommended* for CPU users. Faiss can be installed using "conda install faiss-cpu -c pytorch" or "conda install faiss-gpu -c pytorch".

## Get evaluation datasets
Get monolingual and cross-lingual word embeddings evaluation datasets:
* Our 110 [bilingual dictionaries](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries)
* 28 monolingual word similarity tasks for 6 languages, and the English word analogy task
* Cross-lingual word similarity tasks from [SemEval2017](http://alt.qcri.org/semeval2017/task2/)
* Sentence translation retrieval with [Europarl](http://www.statmt.org/europarl/) corpora

by simply running (in data/):

```bash
./get_evaluation.sh
```
*Note: Requires bash 4. The download of Europarl is disabled by default (slow), you can enable it [here](https://github.com/facebookresearch/MUSE/blob/master/data/get_evaluation.sh#L99-L100).*

## Get monolingual word embeddings
For pre-trained monolingual word embeddings, we highly recommend [fastText Wikipedia embeddings](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md), or using [fastText](https://github.com/facebookresearch/fastText) to train your own word embeddings from your corpus.

You can download the English (en) and Spanish (es) embeddings this way:
```bash
# English fastText Wikipedia embeddings
curl -Lo data/wiki.en.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
# Spanish fastText Wikipedia embeddings
curl -Lo data/wiki.es.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.es.vec
```

## Align monolingual word embeddings
This project includes two ways to obtain cross-lingual word embeddings:
* **Supervised**: using a train bilingual dictionary (or identical character strings as anchor points), learn a mapping from the source to the target space using (iterative) [Procrustes](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem) alignment.
* **Unsupervised**: without any parallel data or anchor point, learn a mapping from the source to the target space using adversarial training and (iterative) Procrustes refinement.

For more details on these approaches, please check [here](https://arxiv.org/pdf/1710.04087.pdf).

### The supervised way: iterative Procrustes (CPU|GPU)
To learn a mapping between the source and the target space, simply run:
```bash
python supervised.py --src_lang en --tgt_lang es --src_emb data/wiki.en.vec --tgt_emb data/wiki.es.vec --n_refinement 5 --dico_train default
```
By default, *dico_train* will point to our ground-truth dictionaries (downloaded above); when set to "identical_char" it will use identical character strings between source and target languages to form a vocabulary. Logs and embeddings will be saved in the dumped/ directory.

### The unsupervised way: adversarial training and refinement (CPU|GPU)
To learn a mapping using adversarial training and iterative Procrustes refinement, run:
```bash
python unsupervised.py --src_lang en --tgt_lang es --src_emb data/wiki.en.vec --tgt_emb data/wiki.es.vec --n_refinement 5
```
By default, the validation metric is the mean cosine of word pairs from a synthetic dictionary built with CSLS (Cross-domain similarity local scaling).

### Evaluate monolingual or cross-lingual embeddings (CPU|GPU)
We also include a simple script to evaluate the quality of monolingual or cross-lingual word embeddings on several tasks:

**Monolingual**
```bash
python evaluate.py --src_lang en --src_emb data/wiki.en.vec --max_vocab 200000
```

**Cross-lingual**
```bash
python evaluate.py --src_lang en --tgt_lang es --src_emb data/wiki.en-es.en.vec --tgt_emb data/wiki.en-es.es.vec --max_vocab 200000
```

## Download
We provide multilingual embeddings and ground-truth bilingual dictionaries.

### Multilingual word Embeddings
We release fastText Wikipedia **supervised** word embeddings for **30** languages, aligned in a **single vector space**.

| | | | | | |
|---|---|---|---|---|---|
| Arabic: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.ar.vec) | Bulgarian: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.bg.vec) | Catalan: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.ca.vec) | Croatian: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.hr.vec) | Czech: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.cs.vec) | Danish: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.da.vec)
| Dutch: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.nl.vec) | English: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.en.vec) | Estonian: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.et.vec) | Finnish: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.fi.vec) | French: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.fr.vec) | German: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.de.vec)
| Greek: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.el.vec) | Hebrew: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.he.vec) | Hungarian: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.hu.vec) | Indonesian: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.id.vec) | Italian: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.it.vec) | Macedonian: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.mk.vec)
| Norwegian: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.no.vec) | Polish: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.pl.vec) | Portuguese: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.pt.vec) | Romanian: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.ro.vec) | Russian: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.ru.vec) | Slovak: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.sk.vec)
| Slovenian: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.sl.vec) | Spanish: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.es.vec) | Swedish: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.sv.vec) | Turkish: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.tr.vec) | Ukrainian: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.uk.vec) | Vietnamese: [*text*](https://s3.amazonaws.com/arrival/embeddings/wiki.multi.vi.vec)




### Ground-truth bilingual dictionaries
We created **110 large-scale ground-truth bilingual dictionaries** using an internal translation tool. The dictionaries handle well the polysemy of words. We provide a train and test split of 5000 and 1500 unique source words, as well as a larger set of up to 100k pairs. Our goal is to *ease the development and the evaluation of cross-lingual word embeddings and multilingual NLP*.

**European languages in every direction**

|   src-tgt  | German | English | Spanish | French | Italian | Portuguese |
|:----------:|:------:|:-------:|:-------:|:------:|:-------:|:----------:|
| German | - |[full](https://s3.amazonaws.com/arrival/dictionaries/de-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/de-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/de-en.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/european/de-es.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/de-es.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/de-es.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/european/de-fr.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/de-fr.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/de-fr.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/european/de-it.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/de-it.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/de-it.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/european/de-pt.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/de-pt.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/de-pt.5000-6500.txt)|
| English |[full](https://s3.amazonaws.com/arrival/dictionaries/en-de.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-de.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-de.5000-6500.txt)| - |[full](https://s3.amazonaws.com/arrival/dictionaries/en-es.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-es.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-es.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/en-fr.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-fr.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-fr.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/en-it.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-it.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-it.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/en-pt.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-pt.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-pt.5000-6500.txt)|
| Spanish |[full](https://s3.amazonaws.com/arrival/dictionaries/european/es-de.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/es-de.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/es-de.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/es-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/es-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/es-en.5000-6500.txt)| - |[full](https://s3.amazonaws.com/arrival/dictionaries/european/es-fr.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/es-fr.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/es-fr.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/european/es-it.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/es-it.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/es-it.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/european/es-pt.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/es-pt.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/es-pt.5000-6500.txt)|
| French |[full](https://s3.amazonaws.com/arrival/dictionaries/european/fr-de.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/fr-de.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/fr-de.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/fr-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/fr-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/fr-en.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/european/fr-es.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/fr-es.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/fr-es.5000-6500.txt)| - |[full](https://s3.amazonaws.com/arrival/dictionaries/european/fr-it.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/fr-it.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/fr-it.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/european/fr-pt.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/fr-pt.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/fr-pt.5000-6500.txt)|
| Italian |[full](https://s3.amazonaws.com/arrival/dictionaries/european/it-de.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/it-de.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/it-de.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/it-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/it-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/it-en.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/european/it-es.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/it-es.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/it-es.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/european/it-fr.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/it-fr.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/it-fr.5000-6500.txt)| - |[full](https://s3.amazonaws.com/arrival/dictionaries/european/it-pt.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/it-pt.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/it-pt.5000-6500.txt)|
| Portuguese |[full](https://s3.amazonaws.com/arrival/dictionaries/european/pt-de.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/pt-de.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/pt-de.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/pt-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/pt-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/pt-en.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/european/pt-es.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/pt-es.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/pt-es.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/european/pt-fr.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/pt-fr.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/pt-fr.5000-6500.txt)|[full](https://s3.amazonaws.com/arrival/dictionaries/european/pt-it.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/european/pt-it.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/european/pt-it.5000-6500.txt)| - |


**Other languages to English (e.g. {fr,es}-en)**

|||||
|-|-|-|-|
| Afrikaans: [full](https://s3.amazonaws.com/arrival/dictionaries/af-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/af-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/af-en.5000-6500.txt) | Albanian: [full](https://s3.amazonaws.com/arrival/dictionaries/sq-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/sq-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/sq-en.5000-6500.txt) | Arabic: [full](https://s3.amazonaws.com/arrival/dictionaries/ar-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/ar-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/ar-en.5000-6500.txt) | Bengali: [full](https://s3.amazonaws.com/arrival/dictionaries/bn-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/bn-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/bn-en.5000-6500.txt)
| Bosnian: [full](https://s3.amazonaws.com/arrival/dictionaries/bs-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/bs-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/bs-en.5000-6500.txt) | Bulgarian: [full](https://s3.amazonaws.com/arrival/dictionaries/bg-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/bg-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/bg-en.5000-6500.txt) | Catalan: [full](https://s3.amazonaws.com/arrival/dictionaries/ca-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/ca-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/ca-en.5000-6500.txt) | Chinese: [full](https://s3.amazonaws.com/arrival/dictionaries/zh-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/zh-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/zh-en.5000-6500.txt)
| Croatian: [full](https://s3.amazonaws.com/arrival/dictionaries/hr-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/hr-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/hr-en.5000-6500.txt) | Czech: [full](https://s3.amazonaws.com/arrival/dictionaries/cs-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/cs-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/cs-en.5000-6500.txt) | Danish: [full](https://s3.amazonaws.com/arrival/dictionaries/da-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/da-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/da-en.5000-6500.txt) | Dutch: [full](https://s3.amazonaws.com/arrival/dictionaries/nl-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/nl-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/nl-en.5000-6500.txt)
| English: [full](https://s3.amazonaws.com/arrival/dictionaries/en-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-en.5000-6500.txt) | Estonian: [full](https://s3.amazonaws.com/arrival/dictionaries/et-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/et-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/et-en.5000-6500.txt) | Filipino: [full](https://s3.amazonaws.com/arrival/dictionaries/tl-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/tl-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/tl-en.5000-6500.txt) | Finnish: [full](https://s3.amazonaws.com/arrival/dictionaries/fi-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/fi-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/fi-en.5000-6500.txt)
| French: [full](https://s3.amazonaws.com/arrival/dictionaries/fr-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/fr-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/fr-en.5000-6500.txt) | German: [full](https://s3.amazonaws.com/arrival/dictionaries/de-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/de-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/de-en.5000-6500.txt) | Greek: [full](https://s3.amazonaws.com/arrival/dictionaries/el-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/el-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/el-en.5000-6500.txt) | Hebrew: [full](https://s3.amazonaws.com/arrival/dictionaries/he-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/he-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/he-en.5000-6500.txt)
| Hindi: [full](https://s3.amazonaws.com/arrival/dictionaries/hi-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/hi-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/hi-en.5000-6500.txt) | Hungarian: [full](https://s3.amazonaws.com/arrival/dictionaries/hu-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/hu-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/hu-en.5000-6500.txt) | Indonesian: [full](https://s3.amazonaws.com/arrival/dictionaries/id-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/id-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/id-en.5000-6500.txt) | Italian: [full](https://s3.amazonaws.com/arrival/dictionaries/it-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/it-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/it-en.5000-6500.txt)
| Japanese: [full](https://s3.amazonaws.com/arrival/dictionaries/ja-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/ja-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/ja-en.5000-6500.txt) | Korean: [full](https://s3.amazonaws.com/arrival/dictionaries/ko-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/ko-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/ko-en.5000-6500.txt) | Latvian: [full](https://s3.amazonaws.com/arrival/dictionaries/lv-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/lv-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/lv-en.5000-6500.txt) | Littuanian: [full](https://s3.amazonaws.com/arrival/dictionaries/lt-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/lt-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/lt-en.5000-6500.txt)
| Macedonian: [full](https://s3.amazonaws.com/arrival/dictionaries/mk-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/mk-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/mk-en.5000-6500.txt) | Malay: [full](https://s3.amazonaws.com/arrival/dictionaries/ms-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/ms-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/ms-en.5000-6500.txt) | Norwegian: [full](https://s3.amazonaws.com/arrival/dictionaries/no-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/no-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/no-en.5000-6500.txt) | Persian: [full](https://s3.amazonaws.com/arrival/dictionaries/fa-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/fa-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/fa-en.5000-6500.txt)
| Polish: [full](https://s3.amazonaws.com/arrival/dictionaries/pl-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/pl-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/pl-en.5000-6500.txt) | Portuguese: [full](https://s3.amazonaws.com/arrival/dictionaries/pt-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/pt-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/pt-en.5000-6500.txt) | Romanian: [full](https://s3.amazonaws.com/arrival/dictionaries/ro-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/ro-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/ro-en.5000-6500.txt) | Russian: [full](https://s3.amazonaws.com/arrival/dictionaries/ru-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/ru-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/ru-en.5000-6500.txt)
| Slovak: [full](https://s3.amazonaws.com/arrival/dictionaries/sk-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/sk-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/sk-en.5000-6500.txt) | Slovenian: [full](https://s3.amazonaws.com/arrival/dictionaries/sl-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/sl-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/sl-en.5000-6500.txt) | Spanish: [full](https://s3.amazonaws.com/arrival/dictionaries/es-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/es-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/es-en.5000-6500.txt) | Swedish: [full](https://s3.amazonaws.com/arrival/dictionaries/sv-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/sv-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/sv-en.5000-6500.txt)
| Tamil: [full](https://s3.amazonaws.com/arrival/dictionaries/ta-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/ta-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/ta-en.5000-6500.txt) | Thai: [full](https://s3.amazonaws.com/arrival/dictionaries/th-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/th-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/th-en.5000-6500.txt) | Turkish: [full](https://s3.amazonaws.com/arrival/dictionaries/tr-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/tr-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/tr-en.5000-6500.txt) | Ukrainian: [full](https://s3.amazonaws.com/arrival/dictionaries/uk-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/uk-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/uk-en.5000-6500.txt)
| Vietnamese: [full](https://s3.amazonaws.com/arrival/dictionaries/vi-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/vi-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/vi-en.5000-6500.txt)









**English to other languages (e.g. en-{fr,es})**

|||||
|-|-|-|-|
| Afrikaans: [full](https://s3.amazonaws.com/arrival/dictionaries/en-af.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-af.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-af.5000-6500.txt) | Albanian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-sq.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-sq.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-sq.5000-6500.txt) | Arabic: [full](https://s3.amazonaws.com/arrival/dictionaries/en-ar.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-ar.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-ar.5000-6500.txt) | Bengali: [full](https://s3.amazonaws.com/arrival/dictionaries/en-bn.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-bn.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-bn.5000-6500.txt)
| Bosnian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-bs.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-bs.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-bs.5000-6500.txt) | Bulgarian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-bg.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-bg.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-bg.5000-6500.txt) | Catalan: [full](https://s3.amazonaws.com/arrival/dictionaries/en-ca.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-ca.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-ca.5000-6500.txt) | Chinese: [full](https://s3.amazonaws.com/arrival/dictionaries/en-zh.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-zh.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-zh.5000-6500.txt)
| Croatian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-hr.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-hr.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-hr.5000-6500.txt) | Czech: [full](https://s3.amazonaws.com/arrival/dictionaries/en-cs.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-cs.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-cs.5000-6500.txt) | Danish: [full](https://s3.amazonaws.com/arrival/dictionaries/en-da.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-da.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-da.5000-6500.txt) | Dutch: [full](https://s3.amazonaws.com/arrival/dictionaries/en-nl.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-nl.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-nl.5000-6500.txt)
| English: [full](https://s3.amazonaws.com/arrival/dictionaries/en-en.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-en.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-en.5000-6500.txt) | Estonian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-et.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-et.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-et.5000-6500.txt) | Filipino: [full](https://s3.amazonaws.com/arrival/dictionaries/en-tl.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-tl.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-tl.5000-6500.txt) | Finnish: [full](https://s3.amazonaws.com/arrival/dictionaries/en-fi.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-fi.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-fi.5000-6500.txt)
| French: [full](https://s3.amazonaws.com/arrival/dictionaries/en-fr.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-fr.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-fr.5000-6500.txt) | German: [full](https://s3.amazonaws.com/arrival/dictionaries/en-de.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-de.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-de.5000-6500.txt) | Greek: [full](https://s3.amazonaws.com/arrival/dictionaries/en-el.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-el.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-el.5000-6500.txt) | Hebrew: [full](https://s3.amazonaws.com/arrival/dictionaries/en-he.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-he.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-he.5000-6500.txt)
| Hindi: [full](https://s3.amazonaws.com/arrival/dictionaries/en-hi.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-hi.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-hi.5000-6500.txt) | Hungarian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-hu.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-hu.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-hu.5000-6500.txt) | Indonesian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-id.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-id.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-id.5000-6500.txt) | Italian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-it.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-it.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-it.5000-6500.txt)
| Japanese: [full](https://s3.amazonaws.com/arrival/dictionaries/en-ja.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-ja.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-ja.5000-6500.txt) | Korean: [full](https://s3.amazonaws.com/arrival/dictionaries/en-ko.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-ko.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-ko.5000-6500.txt) | Latvian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-lv.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-lv.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-lv.5000-6500.txt) | Littuanian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-lt.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-lt.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-lt.5000-6500.txt)
| Macedonian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-mk.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-mk.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-mk.5000-6500.txt) | Malay: [full](https://s3.amazonaws.com/arrival/dictionaries/en-ms.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-ms.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-ms.5000-6500.txt) | Norwegian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-no.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-no.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-no.5000-6500.txt) | Persian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-fa.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-fa.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-fa.5000-6500.txt)
| Polish: [full](https://s3.amazonaws.com/arrival/dictionaries/en-pl.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-pl.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-pl.5000-6500.txt) | Portuguese: [full](https://s3.amazonaws.com/arrival/dictionaries/en-pt.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-pt.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-pt.5000-6500.txt) | Romanian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-ro.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-ro.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-ro.5000-6500.txt) | Russian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-ru.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-ru.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-ru.5000-6500.txt)
| Slovak: [full](https://s3.amazonaws.com/arrival/dictionaries/en-sk.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-sk.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-sk.5000-6500.txt) | Slovenian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-sl.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-sl.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-sl.5000-6500.txt) | Spanish: [full](https://s3.amazonaws.com/arrival/dictionaries/en-es.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-es.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-es.5000-6500.txt) | Swedish: [full](https://s3.amazonaws.com/arrival/dictionaries/en-sv.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-sv.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-sv.5000-6500.txt)
| Tamil: [full](https://s3.amazonaws.com/arrival/dictionaries/en-ta.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-ta.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-ta.5000-6500.txt) | Thai: [full](https://s3.amazonaws.com/arrival/dictionaries/en-th.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-th.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-th.5000-6500.txt) | Turkish: [full](https://s3.amazonaws.com/arrival/dictionaries/en-tr.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-tr.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-tr.5000-6500.txt) | Ukrainian: [full](https://s3.amazonaws.com/arrival/dictionaries/en-uk.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-uk.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-uk.5000-6500.txt)
| Vietnamese: [full](https://s3.amazonaws.com/arrival/dictionaries/en-vi.txt) [train](https://s3.amazonaws.com/arrival/dictionaries/en-vi.0-5000.txt) [test](https://s3.amazonaws.com/arrival/dictionaries/en-vi.5000-6500.txt)



## References
Please cite [[1]](https://arxiv.org/pdf/1710.04087.pdf) if you found the resources in this repository useful.

### Word Translation Without Parallel Data

[1] A. Conneau\*, G. Lample\*, L. Denoyer, MA. Ranzato, H. Jégou, [*Word Translation Without Parallel Data*](https://arxiv.org/pdf/1710.04087.pdf)

\* Equal contribution. Order has been determined with a coin flip.
```
@article{conneau2017word,
  title={Word Translation Without Parallel Data},
  author={Conneau, Alexis and Lample, Guillaume and Ranzato, Marc'Aurelio and Denoyer, Ludovic and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:1710.04087},
  year={2017}
}
```

MUSE is the project at the origin of the work on unsupervised machine translation with monolingual data only [[2]](https://arxiv.org/abs/1711.00043).

### Unsupervised Machine Translation With Monolingual Data Only

[2] G. Lample, L. Denoyer, MA. Ranzato [*Unsupervised Machine Translation With Monolingual Data Only*](https://arxiv.org/abs/1711.00043)

```
@article{lample2017unsupervised,
  title={Unsupervised Machine Translation Using Monolingual Corpora Only},
  author={Lample, Guillaume and Denoyer, Ludovic and Ranzato, Marc'Aurelio},
  journal={arXiv preprint arXiv:1711.00043},
  year={2017}
}
```

Contact: [gl@fb.com](mailto:gl@fb.com)  [aconneau@fb.com](mailto:aconneau@fb.com)
