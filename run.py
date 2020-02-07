#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    run.py train --vocab=<file> --src-file=<file> --tgt-file=<file>
    run.py decode --model-file=<file> --sent-file=<file> --vocab=<file> [options]
    run.py batch_decode --model-file=<file> --test-file=<file> --vocab=<file>

Options:
    -h --help                  Show this screen.
    --model-file=<file>        Pre-trained model file in hdf5 format
    --test-file=<file>         File of test sentences
    --vocab=<file>              Vocab file
    --EMBED_SIZE=<int>         Size of word embedding
    --HIDDEN_SIZE=<int>        Size of hidden units in encoder and decoder
    --DROPOUT_RATE=<float>     Dropout rate
    --BATCH_SIZE=<int>         Batch size
    --NUM_TRAIN_STEPS=<int>    Number of epochs
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from docopt import docopt
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from vocab import *
from nmt import Encoder, Decoder, Attention, NMT, define_checkpoints, train_step, train, decode
import math
import time

if __name__ == '__main__':
    
    args = docopt(__doc__)
    
    print("defining parameters...")
    if args['--EMBED_SIZE']:
        EMBED_SIZE = args['--EMBED_SIZE']
    else:
        EMBED_SIZE = 256
    
    if args['--HIDDEN_SIZE']:
        HIDDEN_SIZE = args['--HIDDEN_SIZE']
    else:
        HIDDEN_SIZE = 512
    
    if args['--DROPOUT_RATE']:
        DROPOUT_RATE = args['--DROPOUT_RATE']
    else:
        DROPOUT_RATE = 0.2
    
    if args['--BATCH_SIZE']:
        BATCH_SIZE = args['--BATCH_SIZE']
    else:
        BATCH_SIZE = 256
    
    if args['--NUM_TRAIN_STEPS']:
        NUM_TRAIN_STEPS = args['--NUM_TRAIN_STEPS']
    else:
        NUM_TRAIN_STEPS = 10
    
    if args['train']:
        
        print('reading vocabulary file: %s' % args['--vocab'])
        VOCAB = Vocab.load(args['--vocab'])
        
        print('reading source sentence file: %s' % args['--src-file'])
        src_sents = read_file(args['--src-file'], source='src')
        
        print('reading target sentence file: %s' % args['--tgt-file'])
        tgt_sents = read_file(args['--tgt-file'], source='tgt')
        
        
        print("padding sequences...")
        src_pad = VOCAB.src.to_input_tensor(src_sents)
        tgt_pad = VOCAB.tgt.to_input_tensor(tgt_sents)
        
        BUFFER_SIZE = len(src_pad)
        steps_per_epoch = len(src_pad)//BATCH_SIZE
        vocab_inp_size = len(VOCAB.src) +1
        vocab_tar_size = len(VOCAB.tgt) +1
            
        print("preparing data pipline...") 
        dataset = tf.data.Dataset.from_tensor_slices((src_pad, tgt_pad)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        
        print("deleting files not required...") 
        del src_sents,tgt_sents,src_pad, tgt_pad
        
        print("beginning training...")
        
        train(dataset, 
              EMBED_SIZE, 
              HIDDEN_SIZE, 
              DROPOUT_RATE, 
              BATCH_SIZE, 
              NUM_TRAIN_STEPS, 
              BUFFER_SIZE,
              steps_per_epoch,
              vocab_inp_size,
              vocab_tar_size,
              VOCAB)
        print("training complete!")
    
    if args['decode']:
        
        print('reading vocabulary file: %s' % args['--vocab'])
        VOCAB = Vocab.load(args['--vocab'])
        
        vocab_inp_size = len(VOCAB.src) +1
        vocab_tar_size = len(VOCAB.tgt) +1        
        
        print('restoring pre-trained model')
        model = NMT(vocab_inp_size, vocab_tar_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE)
        sample_hidden = model.encoder.initialize_hidden_state()
        sample_output, sample_hidden = model.encoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden)
        sample_decoder_output, _, _ = model.decoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, sample_output)
        model.load_weights(args['--model-file'])
        print("beginning decoding...")
        decode(model, args['--sent-file'], VOCAB, 0)