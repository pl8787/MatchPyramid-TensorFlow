"""This is the Training Structure source code.

This module is the main function of model training.
usage:
    python Letor07_Train_Global.py [config_file]
"""

__version__ = '0.1'
__author__ = 'Liang Pang'

import sys
sys.path.insert(0, 'model/')

import json
config = json.loads( open(sys.argv[1]).read() )

import data0_utils as du
Letor07Path = config['data_dir'] 

config['fill_word'] = du._PAD_
config['embedding'] = du.embedding
config['feat_size'] = du.feat_size

pair_gen = du.PairGenerator(rel_file=Letor07Path + '/relation.train.fold%d.txt'%(config['fold']), config=config)

from importlib import import_module
mo = import_module(config['model_file'])
model = mo.Model(config)

import tensorflow as tf
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
model.init_step(sess)

import random
def eval_MAP(pred, gt):
    map_value = 0.0
    r = 0.0
    c = list(zip(pred, gt))
    random.shuffle(c)
    c = sorted(c, key = lambda x:x[0], reverse=True)
    for j, (p, g) in enumerate(c):
        if g != 0:
            r += 1
            map_value += r/(j+1.0)
    if r == 0:
        return 0.0
    else:
        return map_value/r

flog = open(config['log_file'], 'w')
model.saver.export_meta_graph('checkpoint/%s.meta'%(config['model_tag']))
for i in range(config['train_iters']):
    X1, X1_len, X2, X2_len, Y, F = pair_gen.get_batch(data1=du.query_data, data2=du.doc_data)
    feed_dict={ model.X1: X1, model.X1_len: X1_len, model.X2: X2, 
                model.X2_len: X2_len, model.Y: Y, model.F: F}
    loss = model.train_step(sess, feed_dict)
    print('[Train:%s]'%i, loss, file=flog)
    print('\r[Train:%s]'%i, loss, end='')
    flog.flush()
    
    if i == 0:
        model.saver.save(sess, 'checkpoint/%s.ckpt'%(config['model_tag']), global_step=i)
        
    if (i+1) % 200 == 0:
        print('')
        model.saver.save(sess, 'checkpoint/%s.ckpt'%(config['model_tag']), global_step=i, write_meta_graph=False)
        list_gen = du.ListGenerator(rel_file=Letor07Path + '/relation.test.fold%d.txt'%(config['fold']), config=config)
        map_v = 0.0
        map_c = 0.0
        for X1, X1_len, X2, X2_len, Y, F in list_gen.get_batch(data1=du.query_data, data2=du.doc_data):
            feed_dict={ model.X1: X1, model.X1_len: X1_len, model.X2: X2, 
                        model.X2_len: X2_len, model.Y: Y, model.F: F}
            pred = model.test_step(sess, feed_dict)
            map_o = eval_MAP(pred, Y)
            map_v += map_o
            map_c += 1.0
        map_v /= map_c

        print('[Test:%s]'%i, map_v, file=flog)
        print('[Test:%s]'%i, map_v)
        flog.flush()
flog.close()
