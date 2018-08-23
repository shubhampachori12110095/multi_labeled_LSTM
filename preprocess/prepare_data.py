# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import collections

import numpy as np
import pandas as pd
import sklearn as sk

from preprocess.utils import Dictionary
from preprocess.utils import custom_truncate


if __name__ == '__main__':

    max_vocab_size = 50000
    max_sequence_len = 110

    # 0-1. Load data
    with open('./data/tokenized_data.pkl', 'rb') as f:
        data = pickle.load(f)
        assert isinstance(data, pd.DataFrame)

    # 0-2. Get only the first 100 summaries for each of
    # 30 topics, and save them in a dictionary (key: 'topic name', value: '[summaries, titles]'
    num_per_topic = 100
    topics = collections.defaultdict(list)
    grouped_by_topics = data.groupby(by='cluster')
    for name, group in grouped_by_topics:
        topics[name].append(group['token_summary'][:num_per_topic].tolist())
        topics[name].append(group['token_title'][:num_per_topic].tolist())

    # 0-3. Unfold the hierarchical structure the dictionary to lists
    summaries_as_list = []
    titles_as_list = []
    for summaries, titles in topics.values():
        for summary, title in zip(summaries, titles):
            summary = [(sent + ['.']) for sent in summary]
            summaries_as_list.append(summary)
            titles_as_list.append(title)

    # 1-1. Build input dictionary
    inputDictionary = Dictionary()
    for summary in summaries_as_list:
        for sent in summary:
            inputDictionary.add_words(sent)

    # 1-2. Make input data: list of lists, where each internal
    # list corresponds to a summary of variable length
    input_data = []
    longest = 0
    for summary in summaries_as_list:
        summary_ = []
        for sent in summary:
            for word in sent:
                word_idx = inputDictionary.word2idx[word]
                summary_.append(word_idx)
        input_data.append(summary_)
        longest = max([longest, len(summary_)])

    # 1-3. Truncate each input sequence to be under maximum length
    truncate_method = 'back'
    input_data = custom_truncate(docs=input_data,
                                 maxlen=max_sequence_len,
                                 method=truncate_method)

    # 2-1. Build target dictionary
    targetDictionary = Dictionary()
    for title in titles_as_list:
        targetDictionary.add_words(title)

    # 2-2. Make target data
    # FIXME: Remove 'PAD' from dictionary
    target_data = []
    for title in titles_as_list:
        target = [0] * len(targetDictionary)
        for word in title:
            word_idx = targetDictionary.word2idx[word]
            target[word_idx] = 1
        target_data.append(target)

    # 3-1. Get sequence lengths (for use in pack_padded_sequences & pad_packed_sequences)
    input_lengths = [len(inp) for inp in input_data]
    target_lengths = [len(tar) for tar in target_data]

    # 3-2. Shuffle data
    shuffle = True
    if shuffle:
        input_data, target_data, input_lengths = sk.utils.shuffle(input_data, target_data, input_lengths,
                                                                  random_state=2015010720)

    # 3-3. Convert results to dictionary, split train/test, and write to file
    result = {'input': input_data,
              'target': target_data,
              'input_lengths': input_lengths,
              'target_lengths': target_lengths,
              'input_word2idx': inputDictionary.word2idx,
              'input_idx2word': inputDictionary.idx2word,
              'target_word2idx': targetDictionary.word2idx,
              'target_idx2word': targetDictionary.idx2word}

    save_to_pkl = True
    if save_to_pkl:
        with open('./data/rnn_data_without_padding.pkl', 'wb') as f:
            pickle.dump(result, f)
    else:
        raise NotImplementedError
