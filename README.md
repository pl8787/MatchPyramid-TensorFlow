# MatchPyramid-TensorFlow
A simple version of MatchPyramid implement in TensorFlow.

Please reference paper - *Text Matching as Image Recognition* [https://arxiv.org/abs/1602.06359].

## Quick Start

### Toy Dataset Download
You can download data from [[here]](http://pan.baidu.com/s/1eSot4hO).

    $ tar xzvf data.tar.gz

### Dataset Format

#### **Word Dictionary File**   

> *(eg. word_dict.txt)*

We map each word to a uniqe number, called `wid`, and save this mapping in the word dictionary file. 

For example,

```
word   wid
machine 1232
learning 1156
```

#### **Corpus File**    

> *(eg. qid_query.txt and docid_doc.txt)*

We use a value of string identifier (`qid`/`docid`) to represent a sentence, such as a `query` or a `document`. The second number denotes the length of the sentence. The following numbers are the `wid`s of the sentence.

For example,

```
docid  sentence_length  sentence_wid_sequence
GX000-00-0000000 42 2744 1043 377 2744 1043 377 187 117961 ...
```

#### **Relation File**    

> *(eg. relation.train.fold1.txt, relation.test.fold1.txt ...)*

The relation files are used to store the relation between two sentences, such as the relevance relation between `query` and `document`.

For example,

```
relevance   qid   docid
1 3571 GX245-00-1220850
0 3571 GX004-51-0504917
0 3571 GX006-36-4612449
```

#### **Embedding File**    

> *(eg. embed_wiki-pdc_d50_norm)*

We store the word embedding into the embedding file.

For example,

```
wid   embedding
13275 -0.050766 0.081548 -0.031107 0.131772 0.172194 ... 0.165506 0.002235
```

### Training & Evaluation
Run the following commond to train & evaluate MatchPyramid model on Letor dataset:

    $ python Letor07_Train_Global.py config/letor07_mp_fold1.model


## Requirements

* Python 2.7
* TensorFlow 1.1.0


## Future Works

- [ ] Support more similarity functions
- [ ] Baseline models
- [ ] Performance comparison
