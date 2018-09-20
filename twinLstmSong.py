#/nfs/disk/perm/linux-x86_64/bin/python
#coding:utf-8

import os
import sys
import numpy
import theano

from collections import OrderedDict
import random
import time

from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pickle

theano.config.compute_test_value = 'off'
theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'
# theano.config.compute_test_value = 'warn'
theano.config.floatX = 'float32'

# THEANO_FLAGS=mode=FAST_COMPILE,device=gpu,floatX=float32



def load_params(path, params):
    #Load arrays or pickled objects from .npy, .npz or pickled files.
    pp = numpy.load(path)
    for kk, vv in pp.iteritems():
        if kk not in params:
            raise Warning('%s is not in the archive' % kk)
        if options['use_global']==0 and kk=='C_lstm_end_once':
            pass
        else:
            params[kk] = pp[kk]

    return params

# params to tparams(shared)
def p_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk) # theano.shared(value, name): value: 共享变量的值，name: 共享变量的名称
    return tparams


# set tparams(shared) with params
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# convert tparams(键，共享变量) to params(键，数值)
def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def _p(pp, name):
    return '%s_%s' % (pp, name)

# data to numpy array
def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


# generate a (0~n-1) random seq
def get_idx(n, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        random.shuffle(idx_list)

    return idx_list


# init radom weight and wordvec
def init_params(options, inputFile, targetFile):
    
    params = {}

    wordsFilePath = options['wordsFilePath']
    count = 2
    word2id = {}
    id2word = {}

    word2id['START'] = 0
    word2id['END'] = 1

    id2word[0] = 'START'
    id2word[1] = 'END'

    vocabList = []
    word_counts = {}

    print('building vocab')

    if not os.path.exists(wordsFilePath):

        for linet in (open(inputFile).readlines()+open(targetFile).readlines()):
            for line in linet.split('\t'):
                words = line.split()
                for word in words:
                    if word!='\n' and word!='':
                        # print word
                        word_counts[word] = word_counts.get(word, 0) + 1

        vocabList = [w for w in word_counts if word_counts[w] >= options['word_count_threshold']]
            
        for w in vocabList:
            word2id[w] = int(count)
            id2word[int(count)] = w
            # embedding = numpy.array([float(w) for w in line.split()[1:]]).astype(config.floatX)
            # params['Emb'][int(count)] = embedding
            count += 1

        word2id['UNK'] = count
        id2word[count] = 'UNK'
    
        print(str(len(open(inputFile).readlines()+open(targetFile).readlines())) + ' lines have been processed')
        print(str(len(word2id.keys())) + ' words have been initialized')

        wordMisc = {}
        wordMisc['word2id'] = word2id
        wordMisc['id2word'] = id2word
        pickle.dump(wordMisc, open(wordsFilePath, 'wb'))
    else:
        wordMisc = pickle.load(open(wordsFilePath, 'rb'))
        word2id = wordMisc['word2id']
        id2word = wordMisc['id2word']
        print('loaded word vectors')


    charac_num = len(word2id.keys())

    # charac_num = 5000

    # params['W_Z'] = (1/numpy.sqrt(2) * ( 2 * numpy.random.rand(2, options['emb_dim']) - 1)).astype(config.floatX) # C 1,2

    # embedding
    params["Emb"] = (1/numpy.sqrt(charac_num) * ( 2 * numpy.random.rand(charac_num, options['emb_dim']) - 1)).astype(config.floatX)

    # DECODER
    # params['W_Z'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['W_R'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['W'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['U_Z'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['U_R'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['U'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['C_Z'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(2 * options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['C_R'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(2 * options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['de_bias_1'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size']) - 1)).astype(config.floatX)
    # params['de_bias_2'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size']) - 1)).astype(config.floatX)
    # params['de_bias_3'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size']) - 1)).astype(config.floatX)
    # params['de_out_bias'] = (1/numpy.sqrt(charac_num) * ( 2 * numpy.random.rand(charac_num) - 1)).astype(config.floatX)
    params['C_lstm'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(2 * options['hidden_size'], 4*options['hidden_size']) - 1)).astype(config.floatX)
    
    if options['use_global']==1:
        params['C_lstm_end_once'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(2 * options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)

    # params['C'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(2 * options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)
    params['W_S'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)
    

    # MAX_OUT (now is onlt softmax)
    params['W_O'] = (1/numpy.sqrt(2 * options['maxout_size']) * ( 2 * numpy.random.rand(2 * options['maxout_size'], charac_num) - 1)).astype(config.floatX)
    params['U_O'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], 2 * options['maxout_size']) - 1)).astype(config.floatX)
    params['V_O'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], 2 * options['maxout_size']) - 1)).astype(config.floatX)
    params['C_O'] = (1/numpy.sqrt(2 * options['hidden_size']) * ( 2 * numpy.random.rand(2 * options['hidden_size'], 2 * options['maxout_size']) - 1)).astype(config.floatX)


    # # ENCODER_R
    # params['Wr'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['Wr_R'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['Wr_Z'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['Ur_Z'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['Ur_R'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['Ur'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)

    # # ENCODER_L
    # params['Wl'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['Wl_R'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['Wl_Z'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['Ul_Z'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['Ul_R'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)
    # params['Ul'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']) - 1)).astype(config.floatX)

    # # MLP
    params['W_A'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], options['mlp_hidden_size']) - 1)).astype(config.floatX)
    params['U_A'] = (1/numpy.sqrt(2 * options['hidden_size']) * ( 2 * numpy.random.rand(2 * options['hidden_size'], options['mlp_hidden_size']) - 1)).astype(config.floatX)
    params['V_A'] = (1/numpy.sqrt(options['mlp_hidden_size']) * ( 2 * numpy.random.rand(options['mlp_hidden_size'], 1) - 1)).astype(config.floatX)


    # W = numpy.random.randn(options['emb_dim'], options['hidden_size']*4)
    # params['lstm_W'] = W
    
    # U = numpy.random.randn(options['emb_dim'], options['hidden_size']*4)
    # params['lstm_U'] = U

    # b = numpy.zeros((4 * options['hidden_size'],))
    # params['lstm_bias'] = b.astype(config.floatX)

    params['lstm_W'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], options['hidden_size']*4) - 1)).astype(config.floatX)
    params['lstm_U'] = (1/numpy.sqrt(2 * options['emb_dim']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']*4) - 1)).astype(config.floatX)
    params['input_bias'] = (1/numpy.sqrt(4*options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size']*4) - 1)).astype(config.floatX)
        
    params['lstm_de_W'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], options['hidden_size']*4) - 1)).astype(config.floatX)
    params['lstm_de_U'] = (1/numpy.sqrt(2 * options['emb_dim']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']*4) - 1)).astype(config.floatX)
    params['input_de_bias'] = (1/numpy.sqrt(4*options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size']*4) - 1)).astype(config.floatX)
    params['out_bias'] = (1/numpy.sqrt(charac_num) * ( 2 * numpy.random.rand(charac_num) - 1)).astype(config.floatX)


    if os.path.exists(options['googleVec']):
        for line in open(options['googleVec']).readlines()[1:]:
            word = line.split()[0]
            if word == '</s>':
                continue

            ids = word2id[word]
            embedding = numpy.array([float(w) for w in line.split()[1:]]).astype(config.floatX)
            params['Emb'][ids] = embedding


    if os.path.exists(options['model_path']):
        load_params(options['model_path'], params)
        print('weights have been loaded')
    else:
        print('weights of the network have been initialized')

    return params, word2id, id2word


# def word2VecLayer(X, tparams):
#     results, updates = theano.scan(lambda x: tparams['Emb'][x], sequences=[X])
#     return results

# def ortho_weight(ndim, mdim):  # ortho cross matrix, input multiple this matrix is information lostless
#     W = numpy.random.randn(ndim, mdim)
#     u, s, v = numpy.linalg.svd(W)
#     return u.astype(config.floatX)


def lstm_layer(X, tparams, options, if_reverse):

    X_emb, updates = theano.scan(lambda x: tparams['Emb'][x], sequences=[X])

    def _slice(_x, n, dim):
        return _x[n * dim:(n + 1) * dim] # _x[:, n * dim:(n + 1) * dim] for 1*l matrix

    def _step(x_, h_, c_):
        preact = tensor.dot(h_, tparams['lstm_U'])
        preact += x_
        preact += tparams['input_bias']

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['hidden_size']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['hidden_size']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['hidden_size']))
        c = tensor.tanh(_slice(preact, 3, options['hidden_size']))

        c = f * c_ + i * c
        # c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        # h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    X_emb = tensor.dot(X_emb, tparams['lstm_W'])

    
    hidden_and_c, updates = theano.scan(_step,
                                sequences=[X_emb],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), options['hidden_size']),
                                              tensor.alloc(numpy_floatX(0.), options['hidden_size'])])
    if if_reverse:
        return hidden_and_c[0][::-1]
    else:
        return hidden_and_c[0]




def ENCODER_R(X, tparams, options):

    # (tensor.alloc(numpy_floatX(1.), options['hidden_size'], 1)-tensor.nnet.sigmoid(tensor.dot(tparams['Wr_Z'], xr) + tensor.dot(tparams['Ur_Z'], hr_tm1))) * hr_tm1\
    #  + tensor.nnet.sigmoid(tensor.dot(tparams['Wr_Z'], xr) + tensor.dot(tparams['Ur_Z'], hr_tm1)) * tensor.tanh(\
    #     tensor.dot(tparams['Wr'], xr) + tensor.dot(tparams['Ur'],  \
    #         (tensor.nnet.sigmoid(tensor.dot(tparams['Wr_R'], xr) + \
    #             tensor.dot(tparams['Ur_R'], hr_tm1)) * hr_tm1)\
    #         )\
    #     )

    # (tensor.alloc(numpy_floatX(1.), options['hidden_size'])-tensor.nnet.sigmoid(tensor.dot\
    # (tparams["Emb"][xr], tparams['Wr_Z']) + tensor.dot(hr_tm1, tparams['Ur_Z']))) * hr_tm1\
    #  + tensor.nnet.sigmoid(tensor.dot(tparams["Emb"][xr], tparams['Wr_Z']) + tensor.dot(hr_tm1, \
    # tparams['Ur_Z'])) *  tensor.tanh(tensor.dot(tparams["Emb"][xr], tparams['Wr']) + \
    #     tensor.dot((tensor.nnet.sigmoid(tensor.dot(tparams["Emb"][xr], tparams['Wr_R']) + tensor\
    # .dot(hr_tm1, tparams['Ur_R'])) * hr_tm1) , tparams['Ur']))\

    # tparams["Emb"][xr]
    # X_Vec = word2VecLayer(X, tparams)

    results_r, updates = theano.scan(lambda xr, hr_tm1:    (tensor.alloc(numpy_floatX(1.), options['hidden_size'])\
        -tensor.nnet.sigmoid(tensor.dot(tparams["Emb"][xr], tparams['Wr_Z']) + tensor.dot(hr_tm1, tparams['Ur_Z']))) * hr_tm1\
     + tensor.nnet.sigmoid(tensor.dot(tparams["Emb"][xr], tparams['Wr_Z']) + tensor.dot(hr_tm1, \
        tparams['Ur_Z'])) *  tensor.tanh(tensor.dot(tparams["Emb"][xr], tparams['Wr']) + \
        tensor.dot((tensor.nnet.sigmoid(tensor.dot(tparams["Emb"][xr], tparams['Wr_R']) + tensor.\
            dot(hr_tm1, tparams['Ur_R'])) * hr_tm1) , tparams['Ur']))\
     ,  sequences=[X], outputs_info=tensor.alloc(numpy_floatX(0.), options['hidden_size']))
    #initial value of the scan can only be vec

    return results_r # [hi_right]  # return[ (n,) *l ] that is [(1*n) * l]


def ENCODER_L(X_reverse, tparams, options):

    results_l, updates = theano.scan(lambda xl, hl_tm1:   (tensor.alloc(numpy_floatX(1.), options['hidden_size'])\
        -tensor.nnet.sigmoid(tensor.dot(tparams["Emb"][xl], tparams['Wl_Z']) + tensor.dot(hl_tm1, tparams['Ul_Z']))) * hl_tm1\
     + tensor.nnet.sigmoid(tensor.dot(tparams["Emb"][xl], tparams['Wl_Z']) + tensor.dot(hl_tm1, \
        tparams['Ul_Z'])) *  tensor.tanh(tensor.dot(tparams["Emb"][xl], tparams['Wl']) + \
        tensor.dot((tensor.nnet.sigmoid(tensor.dot(tparams["Emb"][xl], tparams['Wl_R']) + tensor.\
            dot(hl_tm1, tparams['Ul_R'])) * hl_tm1) , tparams['Ul']))\
    ,  sequences=[X_reverse], outputs_info=tensor.alloc(numpy_floatX(0.), options['hidden_size']))

    return results_l[::-1] # [hi_left]



def MLP(s_last, h_right, h_left, tparams): # 
    
    # a_s_h are eij(s), s_last is s_i-1
    a_s_h, updates = theano.scan(lambda hr, hl: tensor.dot(tensor.tanh(tensor.dot(s_last, \
        tparams['W_A']) + tensor.dot(tensor.concatenate([hr, hl]), tparams['U_A'])), tparams['V_A']),
    sequences=[h_right, h_left])

    aij = (tensor.exp(a_s_h)/tensor.sum(tensor.exp(a_s_h))) #.reshape((a_s_h.shape[0], 1))

    up = tensor.dot(aij[:,0].T, h_right)
    down = tensor.dot(aij[:,0].T, h_left)
    ci = tensor.concatenate([up, down])

    # ci_temp, updates = theano.scan(lambda hr, hl: aij * tensor.concatenate([hr, hl], axis = 0) ,
    #                                sequences=[h_right, h_left])

    # ci = tensor.sum(ci_temp, axis=1)

    return ci, aij[:,0].T # 1*2n



def DECODER(h_right, h_left, Y_prev, tparams):

    # Y_prev is START+yi~yn
    # s_list is s1 to sn+1
    # CI is from c1 to cn+1
    # generated y_g is generated y1~yn+END, ALL n+1 items

    s0 = tensor.tanh(tensor.dot(h_left[0], tparams['W_S']))
    # ci0 = MLP(s0, h_right, h_left, tparams)

    # (      tensor.alloc(numpy_floatX(1.), options['hidden_size'], 1)  -   tensor.nnet.sigmoid(     tensor.dot(tparams['W_Z'], y_last) + tensor.dot(tparams['U_Z'], s_last) + tensor.dot(tparams['C_Z'], ci)  )    )*s_last  +\
    # tensor.nnet.sigmoid(  tensor.dot(tparams['W_Z'], y_last) + tensor.dot(tparams['U_Z'], s_last) + tensor.dot(tparams['C_Z'], ci)   )*\
    # tensor.tanh(tensor.dot(tparams['W'], y_last) +  tensor.dot(tparams['C'], ci)  +   tensor.dot(tparams['U'], (   tensor.nnet.sigmoid(tensor.dot(tparams['W_R'], y_last) + \
    #     tensor.dot(tparams['U_R'], s_last) + tensor.dot(tparams['C_R'], ci)) * s_last  ) )  )

    # s_list, updates = theano.scan(lambda y_last, s_last: (    tensor.alloc(numpy_floatX(1.), options['hidden_size'])  -   \
    #     tensor.nnet.sigmoid(     tensor.dot(tparams["Emb"][y_last], tparams['W_Z']) + tensor.dot(s_last, tparams['U_Z']) + \
    #         tensor.dot(MLP(s_last, h_right, h_left, tparams), tparams['C_Z'])    )    )*s_last  +\
    # tensor.nnet.sigmoid(    tensor.dot(tparams["Emb"][y_last], tparams['W_Z']) + tensor.dot(s_last, tparams['U_Z']) + \
    #     tensor.dot(MLP(s_last, h_right, h_left, tparams), tparams['C_Z'])   )*\
    # tensor.tanh(    tensor.dot(tparams["Emb"][y_last], tparams['W']) +  tensor.dot(MLP(s_last, h_right, h_left, tparams), tparams['C'])  \
    #     +   tensor.dot(   (  tensor.nnet.sigmoid(  tensor.dot(tparams["Emb"][y_last], tparams['W_R']) + \
    #     tensor.dot(s_last, tparams['U_R']) + tensor.dot(MLP(s_last, h_right, h_left, tparams), tparams['C_R'])  )  * s_last ), tparams['U']   )    ),
    #                               sequences=[Y_prev], outputs_info=s0)

    # s_list, updates = theano.scan(lambda y_last, s_last: (    tensor.alloc(numpy_floatX(1.), options['hidden_size'])  -   \
    #     tensor.nnet.sigmoid(     tensor.dot(tparams["Emb"][y_last], tparams['W_Z']) + tensor.dot(s_last, tparams['U_Z']) + \
    #         tensor.dot(MLP(s_last, h_right, h_left, tparams), tparams['C_Z'])    )    )*s_last  +\
    # tensor.nnet.sigmoid(    tensor.dot(tparams["Emb"][y_last], tparams['W_Z']) + tensor.dot(s_last, tparams['U_Z']) + \
    #     tensor.dot(MLP(s_last, h_right, h_left, tparams), tparams['C_Z'])   )*\
    # tensor.tanh(    tensor.dot(tparams["Emb"][y_last], tparams['W']) +  tensor.dot(MLP(s_last, h_right, h_left, tparams), tparams['C'])  \
    #     +   tensor.dot(   (  tensor.nnet.sigmoid(  tensor.dot(tparams["Emb"][y_last], tparams['W_R']) + \
    #     tensor.dot(s_last, tparams['U_R']) + tensor.dot(MLP(s_last, h_right, h_left, tparams), tparams['C_R'])  )  * s_last ), tparams['U']   )    ),
    #                               sequences=[Y_prev], outputs_info=s0)

    s_list, updates = theano.scan(lambda y_last, s_last: (    tensor.alloc(numpy_floatX(1.), options['hidden_size'])  -   \
        tensor.nnet.sigmoid(     tensor.dot(tparams["Emb"][y_last], tparams['W_Z']) + tensor.dot(s_last, tparams['U_Z']) + \
            tensor.dot(MLP(s_last, h_right, h_left, tparams), tparams['C_Z'])+tparams['de_bias_1']    )    )*s_last  +\
    tensor.nnet.sigmoid(    tensor.dot(tparams["Emb"][y_last], tparams['W_Z']) + tensor.dot(s_last, tparams['U_Z']) + \
        tensor.dot(MLP(s_last, h_right, h_left, tparams), tparams['C_Z'])+tparams['de_bias_1']     )*\
    tensor.tanh(    tensor.dot(tparams["Emb"][y_last], tparams['W']) +  tensor.dot(MLP(s_last, h_right, h_left, tparams), tparams['C'])+tparams['de_bias_2']  \
        +   tensor.dot(   (  tensor.nnet.sigmoid(  tensor.dot(tparams["Emb"][y_last], tparams['W_R']) + \
        tensor.dot(s_last, tparams['U_R']) + tensor.dot(MLP(s_last, h_right, h_left, tparams), tparams['C_R'])+tparams['de_bias_3']   )  * s_last ), tparams['U']   )    ),
                                  sequences=[Y_prev], outputs_info=s0)

    
    s_list_temp = [s0]+s_list


    y_temp, updates_y = theano.scan(lambda s_l, y_l: tensor.dot( tensor.dot(s_l, tparams['U_O']) + \
        tensor.dot(tparams["Emb"][y_l], tparams['V_O']) + tensor.dot(MLP(s_l, h_right, h_left, tparams), tparams['C_O'] ) , tparams['W_O'])+tparams['de_out_bias'],\
                                   sequences=[s_list_temp, Y_prev])
    
    y_g = tensor.nnet.softmax(y_temp)
    # y_g, updates_g = theano.scan(lambda y: tensor.nnet.softmax(y), sequences=[y_temp])
    return y_g  # , s_list



def lstm_DECODER(h_right, h_left, Y_prev, tparams):

    # charac_num = tparams['W_O'].shape[1]
    Y_prev_emb, updates = theano.scan(lambda y: tensor.dot(tparams['Emb'][y], tparams['lstm_de_W']), sequences=[Y_prev])

    s0 = tensor.tanh(tensor.dot(h_left[0], tparams['W_S']))

    ci_temp_end = tensor.concatenate([h_right[-1], h_left[-1]])


    ci_end = tensor.switch( tensor.eq(options['use_global'], 1) , tensor.dot(ci_temp_end, tparams['C_lstm_end_once']) , tensor.alloc(numpy_floatX(0.), options['hidden_size']))


    # ci_temp = MLP(s0, h_right, h_left, tparams)
    # ci = tensor.dot(ci_temp, tparams['C'])

    def _slice(y_pre_, n, dim):
        return y_pre_[n * dim:(n + 1) * dim] # _x[:, n * dim:(n + 1) * dim] for 1*l matrix

    def _step_de(y_pre_, s_, c_):
        preact = tensor.dot(s_, tparams['lstm_de_U'])
        preact += y_pre_
        preact += tparams['input_de_bias']

        mlptemp = MLP(s_, h_right, h_left, tparams)
        ci_temp = mlptemp[0]
        aij = mlptemp[1]
        ci = tensor.dot(ci_temp, tparams['C_lstm'])

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['hidden_size']) + _slice(ci, 0, options['hidden_size']))#  + ci
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['hidden_size']) + _slice(ci, 1, options['hidden_size']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['hidden_size']) + _slice(ci, 2, options['hidden_size']))
        c = tensor.tanh(_slice(preact, 3, options['hidden_size']) + _slice(ci, 3, options['hidden_size'])) 

        c = f * c_ + i * c

        # c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        # h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h, c, aij


    s_list, updates =  theano.scan(_step_de,
                                sequences=[Y_prev_emb],
                                outputs_info=[s0, ci_end, None])   
    
    s_list_temp = s_list[0] #tensor.concatenate([s0.reshape((1, s0.shape[0])), s_list[0]])   # !!!

    y_temp, updates_y = theano.scan(lambda s_l, y_l: tensor.dot( tensor.dot(s_l, tparams['U_O']) + \
        tensor.dot(tparams["Emb"][y_l], tparams['V_O']) + tensor.dot(MLP(s_l, h_right, h_left, tparams)[0], tparams['C_O'] ) , tparams['W_O']) + tparams['out_bias'],\
                                   sequences=[s_list_temp, Y_prev])
    
    y_g = tensor.nnet.softmax(y_temp)
    # y_g, updates_g = theano.scan(lambda y: tensor.nnet.softmax(y.reshape((1, charac_num))), sequences=[y_temp])
    return y_g, s_list[2]  # , s_list




def sgd(lrate, tparams, grads, X, X_reverse, Y_prev, y, cost_sentence):
    
    # Y_prev = start+y1~yn
    # y is used in cross entropy

    gshared = [theano.shared(p.get_value() * 0., name = '%s_grad' % k)
                   for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    
    # X, X_reverse, Y_prev, y are the inputs of all three network

    f_grad_shared = theano.function([X, X_reverse, Y_prev, y], [cost_sentence], updates=gsup,
                                    name='sgd_f_grad_shared', mode='FAST_COMPILE', allow_input_downcast=True)
    pup = [(p, p - lrate * g) for p, g in zip(tparams.values(), gshared)]
    
    f_update = theano.function([lrate],[], updates=pup,
                               name='sgd_f_update', mode = 'FAST_COMPILE', allow_input_downcast=True)
    
    return f_grad_shared, f_update



def adadelta(lrate, tparams, grads, X, X_reverse, Y_prev, y, cost):

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([X, X_reverse, Y_prev, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared', allow_input_downcast=True)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lrate], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update', allow_input_downcast=True)

    return f_grad_shared, f_update


def build_model_for_training(tparams, options):
    
    trng = RandomStreams(seed=1234)
    
    # network input
    X = tensor.lvector('X')
    X_reverse = tensor.lvector('X_reverse') 
    Y_prev = tensor.lvector('Y_prev')

    y = tensor.fmatrix('Y')  # target

    h_right = lstm_layer(X, tparams, options, 0)  #  return[ (n*1) *l ]
    h_left = lstm_layer(X_reverse, tparams, options, 1)

    # h_right = ENCODER_R(X, tparams, options)  #  return[ (n*1) *l ]
    # h_left = ENCODER_L(X_reverse, tparams, options)


    y_g, aij_list = lstm_DECODER(h_right, h_left, Y_prev, tparams)

    cost = tensor.nnet.categorical_crossentropy(y_g, y)
    cost_sentence = tensor.mean(cost) 
    
    return X, X_reverse, Y_prev, y, cost_sentence



def build_model_for_prediction(tparams, options):
    trng = RandomStreams(seed=1234)
    
    # input
    X = tensor.lvector('X')
    X_reverse = tensor.lvector('X_reverse')
    Y_prev = tensor.lvector('Y_prev')

    h_right = lstm_layer(X, tparams, options, 0)  #  return[ (n*1) *l ]
    h_left = lstm_layer(X_reverse, tparams, options, 1)

    # h_right = ENCODER_R(X, tparams, options)  #  return[ (n*1) *l ]
    # h_left = ENCODER_L(X_reverse, tparams, options)

    y_g, aij_list = lstm_DECODER(h_right, h_left, Y_prev, tparams)

    
    f_predict = theano.function([X, X_reverse, Y_prev], [y_g[-1], aij_list[-1]], name='f_predict', mode = 'FAST_COMPILE', allow_input_downcast=True)
    # f_aij = theano.function([X, X_reverse], aij_list, name='f_aij', mode = 'FAST_COMPILE', allow_input_downcast=True)

    return X, X_reverse, Y_prev, f_predict


def training_process(options):
    
    print("model options", options)
    
    print('Loading data')
    
    trainInput = []
    targetOut = []
    if options['use_first_train']==0:
        
        infilelist = open(options['inputFilePath']).readlines()
        for line in infilelist[:int(options['train_percentage']*len(infilelist))]:
            if line != '\n':
                # trainInput.append(line)
                lines = line.split('\t')
                if len(lines)>1:
                    lines = lines[:len(lines)-1]
                for ll in lines:
                    trainInput.append(ll)

        targetOut = []
        infilelistt = open(options['targetFilePath']).readlines()
        for line in infilelistt[:int(options['train_percentage']*len(infilelistt))]:
            if line != '\n':
                # targetOut.append(line)  # !!!
                lines = line.split('\t')
                if len(lines)>1:
                    lines = lines[1:]
                for ll in lines:
                    targetOut.append(ll)
    else:
        infilelist = open(options['inputFilePath']).readlines()
        for line in infilelist[:int(options['train_percentage']*len(infilelist))]:
            
            l_temp = line.split('\t')[0]
            trainInput.append(l_temp)
            
            l1 = line.replace(l_temp, '')
            targetOut.append(l1.replace('\t', ' / ')[3:])

    
    print('Initialize parameters')

    params, word2id, id2word = init_params(options, options['inputFilePath'], options['targetFilePath'])
    tparams = p_tparams(params)

    charac_num = len(word2id.keys())

    
    print('Building model')
    
    X, X_reverse, Y_prev, y, cost_sentence = build_model_for_training(tparams, options)
    
    grads = tensor.grad(cost_sentence, wrt=tparams.values())
    
    lrate = tensor.fscalar(name='lrate')
    # f_grad_shared, f_update = sgd(lrate, tparams, grads, X, X_reverse, Y_prev, y, cost_sentence)
    f_grad_shared, f_update = adadelta(lrate, tparams, grads, X, X_reverse, Y_prev, y, cost_sentence)
    
    # random_seq = get_idx(options['batch_size'], True) # random choose training data
    
    start_time = time.clock()
    

    mean_of_costAll = 0. # last cost
    best_cost = 0.
    best_p = None # last step parameters
    count = 0

    lr = options['lrate']

    print('start running')
    sys.stdout.flush()

    i = 1
    
    for epoch in range(options['max_epochs']):
        costAll = []
        
        for idxx in get_idx(len(trainInput), True):

            # get a ques and a answ
            idxx = random.randint(0, len(trainInput)-1)
            ques = ['START'] + trainInput[idxx].split() + ['END']
            # answ = ['START'] + targetOut[idx] + ['END']
            temp = targetOut[idxx].split()
            answIn = ['START'] + temp
            answOut = temp + ['END']

            # convert to X, X_reverse, Y_prev, y vectors and matries
            x_in = []
            y_prev = []
            y_out = []

            for q in ques:
                # temp_q = numpy.zeros(charac_num, dtype="int32")
                # temp_q[word2id[q]] = 1
                x_in.append(word2id.get(q, charac_num-1))

            X = numpy_floatX(x_in)
            X_reverse = numpy_floatX(x_in)[::-1] # 

            for a in answIn:
                # temp_a = numpy.zeros(charac_num, dtype="int32")
                # temp_a[word2id[a]] = 1
                y_prev.append(word2id.get(a, charac_num-1))
            Y_prev = numpy_floatX(y_prev)

            for a in answOut:
                temp_a = numpy.zeros(charac_num, dtype="int32")
                temp_a[word2id.get(a, charac_num-1)] = 1
                y_out.append(temp_a)
            y = numpy_floatX(y_out)

            # print('++++---=======-----+++++++++')
            #bind all variables together

            cost_sentence = f_grad_shared(X, X_reverse, Y_prev, y)

            # update parameters 
            f_update(lr)   
            costAll.append(cost_sentence)

            print str(i)+' '
            i = i+1
            sys.stdout.flush()

        print('\n\n')
        print(str(epoch)+'th epoch finished\n')
        print('\n\n')
        ### adjust the lr ###
        if epoch == 0:
            difference = abs(best_cost - numpy.mean(costAll))
        else:
            difference = best_cost - numpy.mean(costAll)
        
        if difference > 0:
            best_p = unzip(tparams) # record weights
            best_cost = numpy.mean(costAll)
            count = 0
            # count = count + 1
        elif count>=options['up_times']: # difference <= 0 fluctuate happened, use last weight and new learning rate to recaculate
            zipp(best_p, tparams)
            lr = lr / 1.2
            continue
        else:
            count+=1

        sys.stdout.flush()

        mean_of_costAll = numpy.mean(costAll)
        print mean_of_costAll,' : ',epoch,' : ',lr
        
        if difference < 0.001 and difference > 0:
            break

        best_p = unzip(tparams)
        numpy.savez(options['save_to'], **best_p)

        if epoch%10==0:
            numpy.savez(str(epoch)+'th_epoch_model_song_cost='+str(mean_of_costAll)+'.npz', **best_p)



    end_time = time.clock()
    
    print 'The training process has finished.'
    
    print >> sys.stderr, 'Training took %.1fs' % (end_time - start_time)
    
    # save model
    best_p = unzip(tparams)
    numpy.savez(options['save_to'], **best_p)
    
    print 'The parameters has been saved to the model.'

    print('training process end')

    sys.stdout.flush()

def predict_process(options):

    reload_model = options['model_path']
    # reload_mapfile = 'experiment/map2id.txt',
    test_file = options['test_in_file']
    test_target = options['test_out_file']
    predict_file = options['predict_file']

    word2id = {}
    id2word = {}
    if os.path.exists(options['wordsFilePath']):
        wordMisc = pickle.load(open(options['wordsFilePath'], 'rb'))
        word2id = wordMisc['word2id']
        id2word = wordMisc['id2word']
        print('loaded word vectors')
    else:
        print('missing wordVec file')

    charac_num = len(word2id.keys())
    
    print 'Loading data'

    # lengt = int(len(open(test_file).readlines()))
    # r_num = random.randint(1, lengt-11)
    test = []
    target = []
    
    
    if options['use_first_predict']==0:
        aaa = open(test_file).readlines()
        for l in aaa:#[int(options['train_percentage']*len(aaa)):]:#[r_num:r_num+10]:
            if l != '\n':
                # test.append(l)
                lines = l.split('\t')
                if len(lines)>1:
                    lines = lines[:len(lines)-1]
                for ll in lines:
                    test.append(ll)

        aaa = open(test_target).readlines()
        for l in aaa:#[int(options['train_percentage']*len(aaa)):]:#[r_num:r_num+10]:
            if l != '\n':
                lines = l.split('\t')
                if len(lines)>1:
                    lines = lines[1:]
                for ll in lines:
                    target.append(ll)
                # target.append(l)
    else:
        aaa = open(test_file).readlines()
        for l in aaa:#[int(options['train_percentage']*len(aaa)):]:#[r_num:r_num+10]:
            if l != '\n':
                l_temp = l.split('\t')[0]
                test.append(l_temp)
                target.append(l.replace('\t', '    '))
    
    print 'Loading parameters'
    
    test = [options['input_sen']]
    target = [options['input_sen']]


    params = {}
    # CSM
    params["Emb"] = []
    # DECODER
    params['W_Z'] = []
    params['W_R'] = []
    params['W'] = []
    params['U_Z'] = []
    params['U_R'] = []
    params['U'] = []
    params['C_Z'] = []
    params['C_R'] = []
    params['de_bias_1'] = []
    params['de_bias_2'] = []
    params['de_bias_3'] = []
    params['de_out_bias'] = []
    params['C'] = []
    params['C_lstm'] = []
    params['W_S'] = []

    params['C_lstm_end_once'] = []

    # MAX_OUT (now is onlt softmax)
    params['W_O'] = []
    params['U_O'] = []
    params['V_O'] = []
    params['C_O'] = []

    # ENCODER_R
    params['Wr'] = []
    params['Wr_R'] = []
    params['Wr_Z'] = []
    params['Ur_Z'] = []
    params['Ur_R'] = []
    params['Ur'] = []


    # ENCODER_L
    params['Wl'] = []
    params['Wl_R'] = []
    params['Wl_Z'] = []
    params['Ul_Z'] = []
    params['Ul_R'] = []
    params['Ul'] = []

    params['lstm_W'] = []
    params['lstm_U'] = []
    params['input_bias'] = [] 

    params['lstm_de_W'] = []    
    params['lstm_de_U'] = []
    params['input_de_bias'] = []
    params['out_bias'] = []

    # MLP
    params['W_A'] = []
    params['U_A'] = []
    params['V_A'] = []
    

    load_params(reload_model, params)

    # for k in params.keys():
    #     print k
    #     print params[k].shape
        
    tparams = p_tparams(params)
    
    
    print 'Building model'
    
    X, X_reverse, Y_prev, f_predict = build_model_for_prediction(tparams, options)
    

    if os.path.exists(predict_file):
        os.remove(predict_file)
    pfile = open(predict_file, 'w')
    

    print 'Starting generate answers...'
    
    lentest = len(test)
    error=0
    for t in range(lentest):

        l_sentence = test[t]
        t_sentence = target[t]
        
        sen_w = ['START'] + l_sentence.split() + ['END']
        in_sentences_ids = [word2id.get(w, charac_num-1) for w in sen_w]
        

        X = numpy_floatX(in_sentences_ids) 
        X_reverse = numpy_floatX(in_sentences_ids[::-1])
    
        pre_words = []
        pre_words_e = []

        aij_max = []
        
        first_w = 'START'
        # pre_words.append(first_w)
        pre_words_e.append(word2id.get(first_w, charac_num-1))
        

        pfile.write('input: '+l_sentence) # , end=''
        pfile.write('\n')
        pfile.write('predict: ') # , end=''

        count=1
        iserror = 0
        for ij in options['predict_seq_len']:
            for j in range(0, ij):  # max sentence length
                
                Y_prev = numpy_floatX(pre_words_e) 
                
                predict_temp = f_predict(X, X_reverse, Y_prev)
                predict_word_id_list = predict_temp[0]
                # print predict_temp[1].argsort(axis=0)[-1]
                # print predict_temp[1])
                aij_max.append(predict_temp[1])
                # aij_max.append(predict_temp[1].argsort(axis=0)[-1])
                predict_word_id = predict_word_id_list.argsort(axis=0)[-1]#.argmax(axis=0)

                if (count in options['count']) and options['end_break']==0 and options['use_first_predict']==1:
                    if predict_word_id!=charac_num-1 and predict_word_id!=1:# and t>=options['train_percentage']*lentest:
                        iserror = 1

                    predict_word_id=charac_num-1
                    pre_words.append(id2word[int(predict_word_id)])
                    pre_words_e.append(predict_word_id)

                else:
                    if predict_word_id==1:
                        if options['end_break']==1:
                            pre_words.append(id2word[int(predict_word_id)])
                            pre_words_e.append(predict_word_id)
                            break
                        else:
                            predict_word_id = predict_word_id_list.argsort(axis=0)[-2]
                            pre_words.append(id2word[int(predict_word_id)])
                            pre_words_e.append(predict_word_id)
                    elif predict_word_id==charac_num-1:
                        if len(options['count'])>=1:
                            predict_word_id = predict_word_id_list.argsort(axis=0)[-2]
                            pre_words.append(id2word[int(predict_word_id)])
                            pre_words_e.append(predict_word_id)
                        else:
                            pre_words.append(id2word[int(predict_word_id)])
                            pre_words_e.append(predict_word_id)
                    else:
                        pre_words.append(id2word[int(predict_word_id)])
                        pre_words_e.append(predict_word_id)



                count+=1
            # for w1 in pre_words:
            #     pfile.write(w1+' ')
            # pfile.write('    ')

            for ii in range(0, len(pre_words)):
                # pfile.write(pre_words[ii]+'='+str(aij_max[ii])+' ')
                pfile.write(pre_words[ii]+' ')
            pfile.write('    ')

            sen_w = ['START'] + pre_words + ['END']
            in_sentences_ids = [word2id.get(w, charac_num-1) for w in sen_w]
            X = numpy_floatX(in_sentences_ids)
            X_reverse = numpy_floatX(in_sentences_ids)[::-1]

            pre_words = []
            pre_words_e = []

            pre_words_e.append(word2id.get('START', charac_num-1))

        pfile.write('\n')
        pfile.write('origin: '+t_sentence) # , end=''
        pfile.write('\n\n')
        if iserror:
            error+=1
    
    print 'The generation has finished, the file has been save to ' + predict_file
    pfile.write('error_rate: '+str(   float(error)/float(lentest)   )  )


if __name__ == '__main__':
    options = {}

    options['path'] = 'dataModel/'
    options['use_global'] = 1
    options['lrate'] = 0.3
    options['batch_size'] = 800
    options['cut_out_sort'] = 5000
    options['cut_predict'] = 5000
    options['train_percentage'] = 0.9
    options['word_count_threshold'] = 0
    options['up_times'] = 4
    options['use_first_predict'] = 1
    options['use_first_train'] = 1
    options['use_little_predict'] = 0
    options['predict_file'] = 'output.txt'
    options['use_correspond'] = 1
    options['top_k'] = 0

    options['emb_dim'] = 150
    options['max_epochs'] = 100
    options['hidden_size'] = 150
    options['mlp_hidden_size'] = 150
    options['maxout_size'] = 150


    if not os.path.exists('config.txt'):

        options['use_global'] = 1
        options['poem_type'] = 'ymr'
        options['emb_dim'] = 150
        options['max_epochs'] = 100
        options['hidden_size'] = 150
        options['mlp_hidden_size'] = 150
        options['maxout_size'] = 150
        options['lrate'] = 0.3
        options['wordsFilePath'] = 'word2vec_song.txt'
        options['inputFilePath'] = 'song.txt'
        options['targetFilePath'] = 'song.txt'
        options['model_path'] = 'model_song.npz'
        options['save_to'] = 'model_song_saved.npz'
        options['test_in_file'] = 'song.txt'
        options['test_out_file'] = 'song.txt'
        options['predict_file'] = 'out_song.txt'
        options['googleVec'] = 'googleVec.txt'
        # options['batch_size'] = 12
        
        options['train_percentage'] = 0.9
        options['word_count_threshold'] = 0
        options['up_times'] = 4

        options['end_break'] = 0
        options['predict_seq_len'] = [57]
        options['use_first_predict'] = 1
        options['use_first_train'] = 1
        # options['max_predict_len'] = 7
        # options['poem_length'] = 6
        

        options['count'] = [6, 14, 21, 25, 33, 39, 47, 54]
        options['predict_only'] = 1

    else:
        f = open('config.txt', 'r').readlines()
        for l in f:
            if l[0]!='#' and l!='\n':
                l = l.replace(' ', '')
                l = l.split('//')[0]
                
                if l.split('=')[0] in ['wordsFilePath', 'poem_type', 'inputFilePath', 'targetFilePath',\
                 'model_path', 'test_in_file', 'test_out_file', 'predict_file', 'googleVec', 'save_to', \
                 'test_head_file', 'lv_7', 'lv_5', 'input_sen', 'head']:
                    options[l.split('=')[0]]=l.split('=')[1].replace('\'', '')
                elif l.split('=')[0] in ['train_percentage', 'lrate']:
                    options[l.split('=')[0]]=float(l.split('=')[1].replace('\'', ''))
                else:
                    options[l.split('=')[0]]=int(l.split('=')[1].replace('\'', ''))


        options['wordsFilePath'] = options['path']+options['poem_type']+'/'+options['poem_type']+'_'+options['wordsFilePath']
        options['inputFilePath'] = options['path']+options['poem_type']+'/'+options['poem_type']+'_'+options['inputFilePath']
        options['targetFilePath'] = options['path']+options['poem_type']+'/'+options['poem_type']+'_'+options['targetFilePath']
        options['save_to'] = options['path']+options['poem_type']+'/'+options['poem_type']+'_'+options['save_to']


        if options['use_global']==0:
            
            options['model_path'] = options['path']+options['poem_type']+'/'+options['poem_type']+'_noGlobal_'+options['model_path']
        else:
            options['model_path'] = options['path']+options['poem_type']+'/'+options['poem_type']+'_'+options['model_path']

        options['test_in_file'] = options['path']+options['poem_type']+'/'+options['poem_type']+'_'+options['test_in_file']
        options['test_out_file'] = options['path']+options['poem_type']+'/'+options['poem_type']+'_'+options['test_out_file']
        # options['predict_file'] = options['path']+options['poem_type']+'/'+options['poem_type']+'_'+options['predict_file']
        options['predict_file'] = options['poem_type']+'_'+options['predict_file']
        options['googleVec'] = options['path']+options['poem_type']+'/'+options['poem_type']+'_'+options['googleVec']

        print options['wordsFilePath']



    if options['poem_type']=='ymr':
        options['count'] = [6, 14, 21, 25, 33, 39, 47, 54]
        options['predict_seq_len'] = [57]
    elif options['poem_type']=='dlh':
        options['count'] = [5, 11, 19, 27, 35, 40, 46, 54]
        options['predict_seq_len'] = [61]
    elif options['poem_type']=='poem5':
        options['count'] = [6, 12]
        options['predict_seq_len'] = [17]
    elif options['poem_type']=='poem7':
        options['count'] = [8, 16]
        options['predict_seq_len'] = [23]
    else:
        print 'this poem type has not been achieved!'

    if options['predict_only']==1:
        predict_process(options)
    else:
        training_process(options)
        predict_process(options)

