# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

if __name__ == '__main__':

    with open('./data/rnn_data_without_padding.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    input_data = data_dict['input']
    target_data = data_dict['target']
    input_lengths = data_dict['input_lengths']
    target_lengths = data_dict['target_lengths']
    input_word2idx = data_dict['input_word2idx']
    target_word2idx = data_dict['target_word2idx']

    print(">>> #. inputs  : ", len(input_data))
    print(">>> #. targets : ", len(target_data))
    print(">>> Input vocab size  : ", len(input_word2idx))
    print(">>> Target vocab size : ", len(target_word2idx))

    # TODO: ...
