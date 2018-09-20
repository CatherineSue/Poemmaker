#!/bin/sh
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 /nfs/disk/perm/linux-x86_64/bin/python twinLstm.py