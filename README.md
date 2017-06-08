# MatchPyramid-TensorFlow
A simple version of MatchPyramid implement in TensorFlow.

Please reference paper - Text Matching as Image Recognition [https://arxiv.org/abs/1602.06359].

## Quick Start

### Toy Dataset Download
You can download data from [[here]](http://pan.baidu.com/s/1eSot4hO).

    $ tar xzvf data.tar.gz

### Training & Evaluation
Run the following commond to train & evaluate MatchPyramid model on Letor dataset:

    $ python Letor07_Train_Global.py config/letor07_mp_fold1.model


## Requirements

* Python 2.7
* TensorFlow 1.1.0
