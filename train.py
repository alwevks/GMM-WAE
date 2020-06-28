import os

from model_config import model_argparse
config = model_argparse()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['device']


import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

import numpy as np
import utils

from supervised_stochastic_wae import StochasticWAEModel

data, labels, idx_genre = utils.new_get_mnli_data('../data/genre-all.txt', '../data/mnli-all.txt')

print('[INFO] Number of sentences = {}'.format(len(data)))

sentences = [s.strip() for s in data]

print('[INFO] Tokenizing input and output sequences')
filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
x, word_index = utils.tokenize_sequence(sentences,
                                             filters,
                                             config['num_tokens'],
                                             config['vocab_size'])
print('[INFO] Split data into train-validation-test sets')
l = int(0.8 * len(x) // 512) * 512

x_train, labels_train = x[:l], labels[:l]
x_val, labels_val = x[l:], labels[l:]



w2v = config['w2v_file']
embeddings_matrix = utils.create_embedding_matrix(word_index,
                                                  config['embedding_size'],
                                                  w2v)

# Re-calculate the vocab size based on the word_idx dictionary
config['vocab_size'] = len(word_index)

#----------------------------------------------------------------#

model = StochasticWAEModel(config,
                    embeddings_matrix,
                    word_index,
                    idx_genre)

model.train(x_train, x_val, labels_train, labels_val)

gl.log_writer.close()

#----------------------------------------------------------------#
