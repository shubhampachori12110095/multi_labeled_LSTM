# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import spacy
from gensim.utils import simple_preprocess
import en_core_web_sm


"""
    @ Contents:
        1. class: Dictionary
        2. func: remove_garbage_characters
        3. func: documents2words
        4. func: remove_stopwords
        5. func: lemmatization
        6. func: custom_truncate
        7. func: custom_pad
"""

def remove_garbage_characters(docs):
    """Function to remove garbage characters."""
    assert isinstance(docs, list)
    # Remove Emails
    docs = [re.sub('\S*@\S*\s?', '', doc) for doc in docs]
    # Remove new line characters
    docs = [re.sub('\s+', ' ', doc) for doc in docs]
    # Remove distracting single quotes
    docs = [re.sub("\'", " ", doc) for doc in docs]
    # Remove special characters
    docs = [doc.split('. ') for doc in docs]
    docs = [[re.sub('\W+', ' ', s) for s in doc] for doc in docs]
    return docs


def documents2words(docs):
    # TODO: Reimplement in sentence level
    """
    Tokenize documents to words.\n
    Arguments:\n
        docs: a list of documents, where each document is
              also a list of sentences (str).\n
    Returns:\n
        Similar to input 'docs', but differs in that each sentence has been
        tokenized in to lists of words.
    """
    output = []
    for doc in docs:
        out = []
        for sent in doc:
            o = simple_preprocess(str(sent), deacc=True)
            out.append(o)
        output.append(out)
    return output


def remove_stopwords(docs, stopwords_):
    # TODO: Reimplement in sentence level
    """Remove stopwords."""
    output = []
    for doc in docs:
        out = []
        for sent in doc:
            o = [word for word in sent if word not in stopwords_]
            out.append(o)
        output.append(out)
    return output


def lemmatization(docs, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    # TODO: Reimplement in sentence level
    """Lemmatize documents."""
    try:
        nlp = spacy.load('en', disable=['parser', 'ner'])
    except:
        nlp = en_core_web_sm.load()
        
    output = []
    for doc in docs:
        out = []
        for sent in doc:
            o = [word.lemma_ for word in nlp(" ".join(sent)) if word.pos_ in allowed_postags]
            out.append(o)
        output.append(out)
    return output


class Dictionary(object):
    """Builds dictionary given a set of documents."""
    def __init__(self):
        self.word2idx = {}
        self.word2idx['PAD'] = 0
        self.word2idx['UNKNOWN'] = 1
        self.idx2word = []
        self.idx2word.extend(['PAD', 'UNKNOWN'])
        self.word_count = {'PAD': 1,
                           'UNKNOWN': 1}

    def __len__(self):
        return len(self.idx2word)

    def add_word(self, word):
        if not word in self.word2idx.keys():
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word_count[word] = 1
        else:
            self.word_count[word] += 1

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def limit_dictionary_size(self, max_size):
        # Sort in descending order, and truncate less frequent words
        word_count = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        word_count = word_count[:max_size - 2]  # For 'PAD' & 'UNKNOWN'
        word_count.append(('PAD', 1))
        word_count.append(('UNKNOWN', 1))
        self.word_count = dict(word_count)

        word2idx = {}
        word2idx['PAD'] = 0
        word2idx['UNKNOWN'] = 1
        for i, word in enumerate(self.idx2word):
            if word in ['PAD', 'UNKNOWN']:
                continue
            else:
                if word in self.word_count.keys():
                    word2idx[word] = len(word2idx)
                else:
                    pass
        self.word2idx = word2idx
        self.idx2word = list(self.word2idx.keys())


def custom_truncate(docs, maxlen, method='back'):
    """Truncate list of sequences, either from the front or the back."""
    output = []
    if method == 'back':
        for doc in docs:
            if len(doc) > maxlen:
                out = doc[:maxlen]
            else:
                out = doc
            output.append(out)
    elif method == 'front':
        for doc in docs:
            if len(doc) > maxlen:
                out = doc[-maxlen:]
            else:
                out = doc
            output.append(out)
    else:
        raise ValueError("Expected one of 'front' or 'back', received {}".format(method))

    return output


def custom_pad(docs, maxlen, pad_token=0, method='back'):
    """Pad list of sequences, either at the front or the back."""
    output = []
    if method == 'back':
        for doc in docs:
            if len(doc) < maxlen:
                pad_length = maxlen - len(doc)
                out = doc + [pad_token] * pad_length
            else:
                out = doc
            output.append(out)
    elif method == 'front':
        for doc in docs:
            if len(doc) < maxlen:
                pad_length = maxlen - len(doc)
                out = [pad_token] * pad_length + doc
            else:
                out = doc
            output.append(out)
    else:
        raise ValueError("Expected one of 'front' or 'back', received {}".format(method))

    return output
