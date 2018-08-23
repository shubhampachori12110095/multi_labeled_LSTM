import numpy as np
import csv
import itertools
from collections import Counter
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
stopwords_ = stopwords.words('english')

import operator
from functools import reduce

from preprocess.utils import remove_garbage_characters
from preprocess.utils import documents2words
from preprocess.utils import remove_stopwords
from preprocess.utils import lemmatization

def read_tsv(path):
    data = []
    with open(path, encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for n, line in enumerate(tsvreader):
            index = line[0]
            categories = line[1].split(' ')
            text = line[2] + '. ' + line[3]  # join title & abstract
            keywords = line[5]

            if '1702' in categories:
                categories = ', '.join(categories)
                data.append([index, categories, text, keywords])

    print('{} AI papers in the data'.format(len(data)))
    data = np.array(data)

    return data


def make_multi_label(data, min_cnt=100):
    keywords = data[:, 3]  # keywords

    keywords_split = [x.split(' | ') for x in keywords]
    # each paper has 12.26 keywords on average

    # keyword counting
    keywords_count = list(itertools.chain(*keywords_split))
    keyword_count = Counter(keywords_count)

    keyword_text = []
    keyword_cnt = []
    for n, (k, i) in enumerate(keyword_count.items()):
        if i > min_cnt:
            keyword_text.append(k)
            keyword_cnt.append(i)

    num_target = len(keyword_text)
    print('{} target keywords in data'.format(num_target))

    return make_multi_label_target(keywords_split, keyword_text)


def make_multi_label_target(keywords_split, keyword_text):
    num_target = len(keyword_text)
    keyword_text = np.array(keyword_text)
    target_matrix = np.zeros(shape=(len(keywords_split), num_target))

    for i, y in enumerate(keywords_split):
        label_pos = reduce(operator.add, [np.where(keyword_text == x)[0].tolist() for x in y])
        # assgin label
        target_matrix[i, label_pos] = 1

    return target_matrix, keyword_text


def preprocess_text(text):
    print('start preprocessing...')
    text = [x.split('©')[0] for x in text]
    text = remove_garbage_characters(text)
    text = documents2words(text)
    text = remove_stopwords(text, stopwords_)
    text = lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # remove empty list
    for idx, x in enumerate(text):
        x = np.array(x)
        x = x[np.array([len(i) for i in x]) > 0]
        x = x.tolist()
        text[idx] = x

    return text


if __name__ == '__main__':
    path = 'datasets/여운동_201808.tsv'
    data = read_tsv(path)
    label, keywords = make_multi_label(data, min_cnt=100)
    text = preprocess_text(data[:, 2])

    np.save('label.npy', label)
    np.save('text.npy', text)
    np.save('keyword.npy', keywords)


