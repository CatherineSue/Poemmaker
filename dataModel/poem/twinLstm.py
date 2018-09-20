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
theano.config.warn_float64='warn'

# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32

# cut down parameters

def get_next_id(startWords, idSet, predictls_temp, candidate, _candidates, \
    sen_candidates, yunmuModel, count_accum, i_count, j, i, len_c, word2id, id2word,\
    lv_list_l, charac_num, predict_word_id_list, options, lenSt, l_sentence, full_words):

    if options['cut_to_one']==1:
        count_accum /= i_count
    
    lv_list = []
    if lenSt==7:
        lv_list = lv_list_l[0]
    else:
        lv_list = lv_list_l[1]

    if options['not_first']!=0:
        lv_list = lv_list[1:]

    # print lv_list


    yunD0 = {}


    k_count = 0
    k_count_sen = 0
    do_not_w = 0
    do_not_sen = 0

    id_s = set([])
    if options['cut_to_one']==0:
        for w in candidate['word'].split():
            # if w.decode('utf-8') not in startWords:
            id_s.add(word2id.get(w, charac_num-1))
    else:
        # print full_words
        for w in full_words.replace(' /', '').split():
            id_s.add(word2id.get(w, charac_num-1))


    finded = 0
    if len(startWords)<=i_count-1:
        finded = 1
    else:
        w_temp = startWords[i_count-1].encode('utf-8')
        
        if j==0:
            
            for w in word2id.keys():
                if w.find(w_temp)==0:
                    finded=1
            if finded==0:
                _candidates.append(  {'word': candidate['word']+' '+w_temp, 'prob' : candidate['prob']*1 } )
                # break

    avg = 0
    for x in predict_word_id_list:
        avg+=x
    avg/=len(predict_word_id_list)



    if finded==1 or j!=0:
        for id_t in predictls_temp[::-1]:

            word = id2word[id_t]

            yun0 = yunmuModel.getYunDiao(word)

            if len(startWords)==0 and (id_t in idSet):
                continue
            if len(startWords)==0 and (id_t in id_s):
                continue

            if (id_t in idSet) or (j!=0 and id_t in id_s) or (j==i-1 and len(startWords) and word.decode('utf-8')==startWords[i_count-1]):
                continue
            str_t = word.decode('utf-8')
            if j==0 and len(startWords)>=i_count and startWords[i_count-1].find(str_t)!=0:
                # print id2word[id_t].find(startWords[i])
                # print '------++'
                continue

            # !!!
            former_ws = []
            l = candidate['word'].replace('START', '')
            l = l.replace(' /', '')
            temp_str = ''
            count = 0
            t = l.split()

            for w in t:
                if w!=' ' and w!='':
                    count+=1
                    if count%i!=1 or len(startWords)==0:
                        former_ws.append(w)


            if j!=0 and word in former_ws:
                continue
            if j==0 and len(startWords)==0 and word in former_ws:
                continue

            if options['not_first'] and word in l_sentence.split():
                continue

            if len(startWords) and (word.decode('utf-8') in startWords) and j!=0:
                continue


            word_former = ''


            if j!=0 and options['hard_pz']:
                if len(lv_list)>i_count-1:
                    if len(lv_list[i_count-1])>j:
                        lv = lv_list[i_count-1][j]
                        if lv!='0':
                            if yun0['p']!=lv: # if word not in list, continue
                                continue


            if (len(word.decode('utf-8'))+len_c<count_accum) and not do_not_w:# and len(word.decode('utf-8'))<i-1:  
                # print '------====='
                _candidates.append(  {'word': candidate['word']+' '+word, 'prob' : candidate['prob'] * predict_word_id_list[id_t] } )
                # idSet.add(id_t)
                k_count+=1

            if (len(word.decode('utf-8'))+len_c==count_accum) and not do_not_sen:# and len(word.decode('utf-8'))<i-1:
                # yunD = yunmuModel.getYunDiao(word)
                i_count_f = i_count
                if options['not_first']!=0:
                    i_count_f+=1

                if i_count_f in options['yun_list'][1:] and j>1 and options['use_correspond'] != 1:
                    if options['not_first']!=0 and len(options['yun_list']) and options['yun_list'][0]==1:
                        yunD0 = yunmuModel.getYunDiao(l_sentence.split()[-1])
                    else:
                        word1=''
                        lentt=0
                        # wordss = candidate['word'].split()
                        wordss = []
                        if options['cut_to_one']:
                            wordss = full_words.replace(' /', '').split()
                            # print full_words
                        else:
                            wordss = candidate['word'].replace(' /', ' ').split()

                        for w in wordss:

                            if w!=' ' and w!='START' and w!='':# and w!='/':
                                lentt+=len(w.decode('utf-8'))

                            position = options['yun_list'].index(i_count_f)

                            if lentt==count_accum*i_count-i*(options['yun_list'][position]-options['yun_list'][position-1]):
                                # print w
                                word1 = w
                        # print word1
                        yunD0 = yunmuModel.getYunDiao(word1)
                        # print word1

                if i_count_f not in options['yun_list'][1:] or options['use_correspond'] == 1 or ( i_count_f in options['yun_list'][1:] and yunD0['y']==yun0['y']):  #  yunmuModel.yayun(yun0, base, sound)
 
                    sen_candidates.append(  {'word': candidate['word']+' '+word, 'prob' : candidate['prob'] * predict_word_id_list[id_t] } )
                    # idSet.add(id_t)
                    k_count_sen+=1

                
            if k_count>=options['top_k']:
                do_not_w=1
            if k_count_sen>=options['top_k']:
                do_not_sen=1
            

            if do_not_sen and do_not_w:
                break
            if k_count_sen+k_count>options['cut_out_sort']:
                break

    return sen_candidates, _candidates, yunD0


class yunLv(object):
    """docstring for yunMu"""
    # sheng = ['b','p','m','f','d','t','n','l','g','k','h','j','q','x','z','c','s','r','zh','ch','sh']
    def __init__(self, file_path, couplet_path, word_l_path):

        # word_l_list = []
        word_l_dict = {}
        for l in open(word_l_path).readlines():
            word_l = '1'
            if len(l.split()):
                word_l = l.split()[0]
            if not word_l.isalpha():
                s_w = word_l.decode('utf-8')[0].encode('utf-8')
                if s_w in word_l_dict.keys():
                    word_l_dict[s_w].append(word_l)
                else:
                    word_l_dict[s_w] = [word_l]
                # word_l_list.append(word_l)

        self.word_l_dict = word_l_dict

        vocSet = set([])
        yunDict = {}

        f = open(file_path).readlines()
        for l in f:
            ls = l.split()
            if len(ls)==3:
                word = ls[0]
                pingZ = ls[1]
                yun = ls[2]

                vocSet.add(word)
                yunDict[word] = {'p':pingZ, 'y':yun}
            else:
                # print l
                pass

        self.vocSet = vocSet
        self.yunDict = yunDict

        dui_zhang_dict = {}

        f1 = open(couplet_path).readlines()
        for l in f1:
            k = l.split('-----')[0].replace('\n', '')
            if k in dui_zhang_dict.keys():
                dui_zhang_dict[k].append(l.split('-----')[1].replace('\n', ''))
            else:
                dui_zhang_dict[k] = [l.split('-----')[1].replace('\n', '')]

        self.dui_zhang_dict = dui_zhang_dict

    def getYunDiao(self, x):
        y = 'a'
        p = '-1'
        if x in self.vocSet:
            p = self.yunDict[x]['p']
            y = self.yunDict[x]['y']

        return {'y':y, 'p':p}

    def find_start(self, word):
        if word in self.word_l_dict.keys():
            i = random.randint(0, len(self.word_l_dict[word])-1)
            return self.word_l_dict[word][i]
        return ''




def load_params(path, params):
    #Load arrays or pickled objects from .npy, .npz or pickled files.
    pp = numpy.load(path)
    for kk, vv in pp.iteritems():
        if kk not in params:
            raise Warning('%s is not in the archive' % kk)
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
    return numpy.asarray(data, dtype=theano.config.floatX)


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
            # embedding = numpy.array([float(w) for w in line.split()[1:]]).astype(theano.config.floatX)
            # params['Emb'][int(count)] = embedding
            count += 1

        word2id['UNK'] = count
        id2word[count] = 'UNK'
    
        print(str(len(open(inputFile).readlines()+open(targetFile).readlines())) + ' lines have been processed')
        print(str(len(word2id.keys())) + ' words have been initialized')

        wordMisc = {}
        wordMisc['word2id'] = word2id
        wordMisc['id2word'] = id2word

        charac_num = len(word2id.keys())

        P_Emb = (1/numpy.sqrt(charac_num) * ( 2 * numpy.random.rand(charac_num, options['emb_dim']) - 1)).tolist()

        if os.path.exists(options['googleVec']):
            for line in open(options['googleVec']).readlines()[1:]:
                word = line.split()[0]
                if word == '</s>' or word not in word2id.keys():
                    continue

                ids = word2id[word]
                P_Emb[ids] = [float(w) for w in line.split()[1:]]
                # embedding = numpy.array([float(w) for w in line.split()[1:]]).astype(theano.config.floatX)
                # P_Emb[ids] = embedding.tolist()
        else:
            print "do not have GoogleVec!"
            return

        wordMisc['P_Emb'] = P_Emb

        pickle.dump(wordMisc, open(wordsFilePath, 'wb'))
    else:
        wordMisc = pickle.load(open(wordsFilePath, 'rb'))
        word2id = wordMisc['word2id']
        id2word = wordMisc['id2word']
        P_Emb = wordMisc['P_Emb']
        print('loaded word vectors')


    charac_num = len(word2id.keys())

    print charac_num

    params['C_lstm'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(2 * options['hidden_size'], 4*options['hidden_size']) - 1)).astype(theano.config.floatX)
    params['C_lstm_end_once'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(2 * options['hidden_size'], options['hidden_size']) - 1)).astype(theano.config.floatX)

    # params['C'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(2 * options['hidden_size'], options['hidden_size']) - 1)).astype(theano.config.floatX)
    params['W_S'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']) - 1)).astype(theano.config.floatX)
    

    # MAX_OUT (now is onlt softmax)
    params['W_O'] = (1/numpy.sqrt(2 * options['maxout_size']) * ( 2 * numpy.random.rand(2 * options['maxout_size'], charac_num) - 1)).astype(theano.config.floatX)
    params['U_O'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], 2 * options['maxout_size']) - 1)).astype(theano.config.floatX)
    params['V_O'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], 2 * options['maxout_size']) - 1)).astype(theano.config.floatX)
    params['C_O'] = (1/numpy.sqrt(2 * options['hidden_size']) * ( 2 * numpy.random.rand(2 * options['hidden_size'], 2 * options['maxout_size']) - 1)).astype(theano.config.floatX)

    # # MLP
    params['W_A'] = (1/numpy.sqrt(options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size'], options['mlp_hidden_size']) - 1)).astype(theano.config.floatX)
    params['U_A'] = (1/numpy.sqrt(2 * options['hidden_size']) * ( 2 * numpy.random.rand(2 * options['hidden_size'], options['mlp_hidden_size']) - 1)).astype(theano.config.floatX)
    params['V_A'] = (1/numpy.sqrt(options['mlp_hidden_size']) * ( 2 * numpy.random.rand(options['mlp_hidden_size'], 1) - 1)).astype(theano.config.floatX)


    params['lstm_W'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], options['hidden_size']*4) - 1)).astype(theano.config.floatX)
    params['lstm_U'] = (1/numpy.sqrt(2 * options['emb_dim']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']*4) - 1)).astype(theano.config.floatX)
    params['input_bias'] = (1/numpy.sqrt(4*options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size']*4) - 1)).astype(theano.config.floatX)
        
    params['lstm_de_W'] = (1/numpy.sqrt(options['emb_dim']) * ( 2 * numpy.random.rand(options['emb_dim'], options['hidden_size']*4) - 1)).astype(theano.config.floatX)
    params['lstm_de_U'] = (1/numpy.sqrt(2 * options['emb_dim']) * ( 2 * numpy.random.rand(options['hidden_size'], options['hidden_size']*4) - 1)).astype(theano.config.floatX)
    params['input_de_bias'] = (1/numpy.sqrt(4*options['hidden_size']) * ( 2 * numpy.random.rand(options['hidden_size']*4) - 1)).astype(theano.config.floatX)
    params['out_bias'] = (1/numpy.sqrt(charac_num) * ( 2 * numpy.random.rand(charac_num) - 1)).astype(theano.config.floatX)




    if os.path.exists(options['saveto']):
        load_params(options['saveto'], params)
        print('weights have been loaded')
    else:
        print('weights of the network have been initialized')



    return params, word2id, id2word, P_Emb


def lstm_layer(X, tparams, options, if_reverse):

    # X_emb , updates = theano.scan(lambda x: tensor.dot(x, tparams['Emb']), sequences=[X])

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

    X_emb = tensor.dot(X, tparams['lstm_W'])

    
    hidden_and_c, updates = theano.scan(_step,
                                sequences=[X_emb],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), options['hidden_size']),
                                              tensor.alloc(numpy_floatX(0.), options['hidden_size'])])
    
    return tensor.switch( tensor.eq(if_reverse, 1) , hidden_and_c[0][::-1] , hidden_and_c[0])

    




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





def lstm_DECODER(h_right, h_left, Y_prev, tparams):

    # charac_num = tparams['W_O'].shape[1]
    # Y_prev_emb , updates = theano.scan(lambda y: tensor.dot(y, tparams['Emb']), sequences=[Y_prev])
    Y_prev_emb = tensor.dot(Y_prev, tparams['lstm_de_W'])

    s0 = tensor.tanh(tensor.dot(h_left[0], tparams['W_S']))

    ci_temp_end = tensor.concatenate([h_right[-1], h_left[-1]])
    ci_end = tensor.dot(ci_temp_end, tparams['C_lstm_end_once'])

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

    s_list, updates = theano.scan(_step_de,
                                sequences=[Y_prev_emb],
                                outputs_info=[s0, ci_end, None])
    
    s_list_temp = s_list[0] #tensor.concatenate([s0.reshape((1, s0.shape[0])), s_list[0]])   # !!!

    # y_temp, updates_y = theano.scan(lambda s_l, y_l: tensor.dot( tensor.dot(s_l, tparams['U_O']) + \
    #     tensor.dot(tensor.dot(y_l, tparams["Emb"]), tparams['V_O']) + tensor.dot(MLP(s_l, h_right, h_left, tparams)[0], tparams['C_O'] ) , tparams['W_O']) + tparams['out_bias'],\
    #                                sequences=[s_list_temp, Y_prev])
    
    y_temp, updates_y = theano.scan(lambda s_l, y_l: tensor.dot( tensor.dot(s_l, tparams['U_O']) + \
        tensor.dot(y_l, tparams['V_O']) + tensor.dot(MLP(s_l, h_right, h_left, tparams)[0], tparams['C_O'] ) , tparams['W_O']) + tparams['out_bias'],\
                                   sequences=[s_list_temp, Y_prev])

    y_g = tensor.nnet.softmax(y_temp)
    # y_g, updates_g = theano.scan(lambda y: tensor.nnet.softmax(y.reshape((1, charac_num))), sequences=[y_temp])
    return y_g, s_list[2]  # , s_list




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
    X = tensor.fmatrix('X')
    X_reverse = tensor.fmatrix('X_reverse') 
    Y_prev = tensor.fmatrix('Y_prev')

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
    X = tensor.fmatrix('X')
    X_reverse = tensor.fmatrix('X_reverse')
    Y_prev = tensor.fmatrix('Y_prev')

    h_right = lstm_layer(X, tparams, options, 0)  #  return[ (n*1) *l ]
    h_left = lstm_layer(X_reverse, tparams, options, 1)

    # h_right = ENCODER_R(X, tparams, options)  #  return[ (n*1) *l ]
    # h_left = ENCODER_L(X_reverse, tparams, options)

    y_g, aij_list = lstm_DECODER(h_right, h_left, Y_prev, tparams)

    
    f_predict = theano.function([X, X_reverse, Y_prev], [y_g[-1], aij_list[-1]], name='f_predict', mode = 'FAST_RUN', allow_input_downcast=True)
    # f_aij = theano.function([X, X_reverse], aij_list, name='f_aij', mode = 'FAST_RUN', allow_input_downcast=True)

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

    params, word2id, id2word, P_Emb = init_params(options, options['inputFilePath'], options['targetFilePath'])
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
        
        for idx in get_idx(options['batch_size'], True):

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

            ve = []

            little_vecs = []

            for vei in range(0, options['emb_dim']):
                ve.append(0.0)

            for q in ques:

                x_in.append(word2id.get(q, charac_num-1))


            for a in answIn:
                # temp_a = numpy.zeros(charac_num, dtype="int32")
                # temp_a[word2id[a]] = 1
                y_prev.append(word2id.get(a, charac_num-1))


            # Y_prev = numpy_floatX(y_prev)

            for a in answOut:
                temp_a = numpy.zeros(charac_num, dtype="int32")
                temp_a[word2id.get(a, charac_num-1)] = 1
                y_out.append(temp_a)
            y = numpy_floatX(y_out)


            y_pre_temp = []
            x_in_temp = []
            x_in_temp_r = []



            for xi in x_in:
                x_in_temp.append(P_Emb[xi])
            for xi in x_in[::-1]:
                x_in_temp_r.append(P_Emb[xi])
            for xi in y_prev:
                y_pre_temp.append(P_Emb[xi])


            X = numpy_floatX(x_in_temp)
            X_reverse = numpy_floatX(x_in_temp_r)
            Y_prev = numpy_floatX(y_pre_temp)

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
            print "cost so little"
            break

        best_p = unzip(tparams)
        numpy.savez(options['saveto'], **best_p)

        if epoch%10==0:
            numpy.savez(str(epoch)+'th_epoch_model_cover_poem_5k_cost='+str(mean_of_costAll)+'.npz', **best_p)



    end_time = time.clock()
    
    print 'The training process has finished.'
    
    print >> sys.stderr, 'Training took %.1fs' % (end_time - start_time)
    
    # save model
    best_p = unzip(tparams)
    numpy.savez(options['saveto'], **best_p)
    
    print 'The parameters has been saved to the model.'

    print('training process end')

    sys.stdout.flush()

def predict_process(options):

    reload_model = options['saveto']
    # reload_mapfile = 'experiment/map2id.txt',
    test_file = options['test_in_file']
    test_target = options['test_out_file']
    predict_file = options['predict_file']
    theme_file = options['test_head_file']

    word2id = {}
    id2word = {}
    if os.path.exists(options['wordsFilePath']):
        wordMisc = pickle.load(open(options['wordsFilePath'], 'rb'))
        word2id = wordMisc['word2id']
        id2word = wordMisc['id2word']
        P_Emb = wordMisc['P_Emb']
        print('loaded word vectors')
    else:
        print('missing wordVec file')

    charac_num = len(word2id.keys())
    
    print 'Loading data'

    # lengt = int(len(open(test_file).readlines()))
    # r_num = random.randint(1, lengt-11)
    test = []
    target = []
    theme_list = []
    
    
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
    elif options['use_first_predict']==1:
        aaa = open(test_file).readlines()
        for l in aaa:#[int(options['train_percentage']*len(aaa)):]:#[r_num:r_num+10]:
            if l != '\n':
                l_temp = l.split('\t')[0]
                test.append(l_temp)
                target.append(l.replace('\t', '    '))

    else:
        print 'mothod not complished'
        return

    for l in open(theme_file).readlines():
        theme_list.append(l.split('\t')[0])
    
    print 'Loading parameters'
    
    if options['use_little_predict']:
        test = test[int(options['train_percentage']*len(test)):]
        target = target[int(options['train_percentage']*len(target)):]


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

        theme = ''

        if len(theme_list)>t:
            theme = theme_list[t]
        elif len(theme_list)>0:
            theme = theme_list[-1]
        else:
            pass

        l_sentence = test[t]
        t_sentence = target[t]
        
        sen_w = ['START'] + l_sentence.split() + ['END']
        in_sentences_ids = [word2id.get(w, charac_num-1) for w in sen_w]
        


        pre_words = []
        pre_words_e = []

        aij_max = []


        x_in = []
        ve = []

        for vei in range(0, options['emb_dim']):
            ve.append(0.0)

        for q in in_sentences_ids:
            # temp_q = numpy.zeros(charac_num, dtype="int32")
            # temp_q[word2id[q]] = 1
            x_in.append(q)


        
        first_w = 'START'
        # pre_words.append(first_w)
        pre_words_e.append(word2id.get(first_w, charac_num-1))
        

        pfile.write('input: '+l_sentence) # , end=''
        pfile.write('\n')
        pfile.write('head: '+theme)
        pfile.write('\n')
        pfile.write('predict: \n') # , end=''

        count=1
        iserror = 0

        

        if options['top_k']==0:

            if len(l_sentence.split())==5:
                options['predict_seq_len'] = [17]
                options['count'] = [6, 12]

            else:
                options['predict_seq_len'] = [23]
                options['count'] = [8, 16]


            for ij in options['predict_seq_len']:
                for j in range(0, ij):  # max sentence length

                    x_in_temp = []
                    x_in_temp_r = []
                    y_pre_temp = []

                    for xi in x_in:
                        x_in_temp.append(P_Emb[xi])
                    for xi in x_in[::-1]:
                        x_in_temp_r.append(P_Emb[xi])
                    for xi in pre_words_e:
                        y_pre_temp.append(P_Emb[xi])

                    X = numpy_floatX(x_in_temp)
                    X_reverse = numpy_floatX(x_in_temp_r)
                    Y_prev = numpy_floatX(y_pre_temp)


                    predict_temp = f_predict(X, X_reverse, Y_prev)
                    predict_word_id_list = predict_temp[0]
                    # print predict_temp[1].argsort(axis=0)[-1]
                    # print predict_temp[1])
                    aij_max.append(predict_temp[1])
                    # aij_max.append(predict_temp[1].argsort(axis=0)[-1])
                    predict_word_id = predict_word_id_list.argsort(axis=0)[-1]#.argmax(axis=0)


                    for predict_word_id in predict_word_id_list.argsort(axis=0)[::-1]:

                        if predict_word_id in x_in or predict_word_id in pre_words_e:
                            continue

                        else:
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
                            break

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
        else:
            sen_candidates = []
            # _sen = set(['START'])
            # for i in range(0, options['top_k']):
            sen_candidates.append({'word':'START', 'prob':1.0})

            # assert len(options['predict_seq_len']) ==1
            # # pfile.write('\n')
            candidates = []

            # yu = 'a'
            # pz = '-1'
            yunD0 = {}

            # yunmuModel = yunMu(options['yunmu_file'])
            yunmuModel = yunLv(options['yunmu_file'], options['dui_lian'], options['word_l_file'])

            startWords = []#theme.split()
            for s in theme.split():
                s = s.decode('utf-8')
                # for si in s:
                startWords.append(s)
            # print startWords

            print str(t)+'/'+str(lentest)

            if t>options['cut_predict']:
                break

            count_accum=0
            i_count=0

            idSet = set([0, 1, len(word2id.keys())-1])

            ran = []
            if len(startWords):
                ran = range(len(startWords))
            else:
                if options['not_first']==0:
                    ran = options['predict_seq_len']
                else:
                    ran = options['predict_seq_len'][1:]
            # for i in options['predict_seq_len']:  # max sentence length

            if options['75_gen']!=0:

                lll = len(l_sentence.split())

                if len(options['predict_seq_len']):
                    lll = options['predict_seq_len'][0]

                ran = []

                if len(startWords):
                    for i in range(len(startWords)):
                        ran.append(lll)
                else:
                    for i in range(options['75_gen']):
                        ran.append(lll)

            full_candidates = ''
            for ii in ran:
                i = 0
                if len(startWords):
                    if len(options['predict_seq_len']):
                        i = options['predict_seq_len'][0]
                    else:
                        i = len(l_sentence.split())
                else:
                    i=ii
            
                candidates = sen_candidates
                sen_candidates = []
                count_accum+=i
                i_count+=1

                for j in range(0,i):

                    _candidates = []

                    for c in candidates:

                        print candidates

                        len_c = 0
                        for w in c['word'].split():
                            if w!='START' and w!='/':
                                len_c+=len(w.decode('utf-8'))
                        # print len_c
                        # print c['word'].decode('utf-8').encode('utf-8')

                        pre_words_e = []
                        for w in c['word'].split():
                            pre_words_e.append(word2id.get(w, charac_num-1))


                        x_in_temp = []
                        x_in_temp_r = []
                        y_pre_temp = []

                        for xi in x_in:
                            x_in_temp.append(P_Emb[xi])
                        for xi in x_in[::-1]:
                            x_in_temp_r.append(P_Emb[xi])
                        for xi in pre_words_e:
                            y_pre_temp.append(P_Emb[xi])

                        X = numpy_floatX(x_in_temp)
                        X_reverse = numpy_floatX(x_in_temp_r)
                        Y_prev = numpy_floatX(y_pre_temp)


                        predict_temp = f_predict(X, X_reverse, Y_prev)
                        #!! attention
                        predict_word_id_list = predict_temp[0]
                        # aij_max.append(predict_temp[1])
                        aij_max.append(predict_temp[1].argsort(axis=0)[-1])
                        predictls_temp = predict_word_id_list.argsort(axis=0)

                        full_words = full_candidates+c['word'].replace('START', '')
                        # print full_words

                        sen_candidates, _candidates, yunD1 = get_next_id(startWords, idSet, predictls_temp, c, \
                            _candidates, sen_candidates, yunmuModel, count_accum, i_count, j, i, len_c, word2id, id2word,\
                            options['lv_list'], charac_num, predict_word_id_list, options, i, l_sentence, full_words )

                        if yunD1!={}:
                            yunD0=yunD1

                    candidates  =  sorted(_candidates, key=lambda k: k['prob'] )[-1*options['top_k']:] 

                sen_candidates = sorted(sen_candidates, key=lambda k: k['prob'] )[-1*options['top_k']:]
                
                if options['add_cut']:
                    for s in sen_candidates:
                        s['word'] = s['word']+' /'

                if options['cut_to_one'] and options['top_k']==1 and i_count!=len(ran) and len(sen_candidates):
                    pfile.write(sen_candidates[0]['word'].replace('START', ''))
                    # pfile.write('\t')
                    full_candidates += sen_candidates[0]['word'].replace('START', '')
                    sen_candidates = [{'word':'START', 'prob':1.0}]


                # print '======='
            for c in sen_candidates:
                pfile.write(c['word'].replace('START', ''))
                # for ii in range(0, len(c)):
                #     # pfile.write(c[ii]+'='+str(aij_max[ii])+' ')
                #     pfile.write(c[ii]+' ')
                pfile.write('\n')
            # pfile.write(sound+' '+str(base)+'\n')
            if yunD0!={}:
                pfile.write(yunD0['p']+'-----'+yunD0['y'])
            
        pfile.write('\n')
        # pfile.write('origin: '+t_sentence) # , end=''
        pfile.write('\n\n')
        if iserror:
            error+=1
    
    print 'The generation has finished, the file has been save to ' + predict_file
    pfile.write('error_rate: '+str(   float(error)/float(lentest)   )  )


if __name__ == '__main__':
    options = {}

    options['emb_dim'] = 150
    options['max_epochs'] = 1000
    options['hidden_size'] = 600
    options['mlp_hidden_size'] = 400
    options['maxout_size'] = 300
    options['lrate'] = 0.3
    options['wordsFilePath'] = 'word2vec_cover_poem_5k.txt'
    options['inputFilePath'] = 'cover_poem_5k.txt'
    options['targetFilePath'] = 'cover_poem_5k.txt'
    options['saveto'] = 'model_cover_poem_5k.npz'
    options['test_in_file'] = 'test_cover_poem_5k.txt'
    options['test_out_file'] = 'test_cover_poem_5k.txt'
    options['test_head_file'] = 'head_cover_poem_5k.txt'
    options['predict_file'] = 'out_cover_poem_5k_top5_7_theme.txt'
    options['batch_size'] = 800
    
    options['train_percentage'] = 0.9
    options['word_count_threshold'] = 0
    options['up_times'] = 4

    options['end_break'] = 0
    options['predict_seq_len'] = []
    options['use_first_predict'] = 1
    options['use_first_train'] = 1
    # options['max_predict_len'] = 7
    # options['poem_length'] = 6
    options['googleVec'] = 'si_vectors.txt'
    options['use_little_predict'] = 0
    options['cut_out_sort'] = 5000
    options['cut_predict'] = 5000

    options['count'] = []

    # options['max_input_len'] = 80
    options['top_k'] = 1
    options['yunmu_file'] = 'yun_utf8.txt'


    options['dui_lian'] = 'liewngduiyun_dui.txt'
    options['word_l_file'] = 'yunmu.txt'

    options['yun_list'] = [1,2,4,6] # only for poems, the i th sentence need to yayun
    options['lv_list'] = []


    lv_l0 = ['0','p','0','z','0','p','0'],['0','z','0','p','0','z','p'],['0','z','0','p','0','z','0'],['0','p','0','z','0','p','p'], ['0','p','0','z','0','p','0'],['0','z','0','p','0','z','p']
    lv_l1 = ['0','z','0','p','0'],['0','p','0','z','p'],['0','p','0','z','0'],[ '0','z','0','p','p'], ['0','z','0','p','0'],['0','p','0','z','p']

    options['lv_list'].append(lv_l0)
    options['lv_list'].append(lv_l1)

    options['use_single_word'] = 0
    options['use_correspond'] = 0
    options['use_correspond_finetune'] = 0
    options['use_pz_finetune'] = 0
    options['hard_pz'] = 1
    options['use_connect_word'] = 0

    options['75_gen'] = 3

    options['add_cut'] = 1

    options['not_first'] = 1 # change 75_gen to 3

    options['cut_to_one'] = 1
    
    # training_process(options)
    predict_process(options)

