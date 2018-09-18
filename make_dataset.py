import numpy as np
import pandas as pd
import io
import csv
from sklearn.model_selection import train_test_split
from torchtext import data
import pickle
import re
import spacy
import en_core_web_sm
try:
    nlp = spacy.load('en', disable=['parser', 'ner'])
except:
    nlp = en_core_web_sm.load()

import torch


def split_train_test(text, label, keywords, test_size, seed):
    X_train, X_test, y_train, y_test = train_test_split(
            text, label, test_size=test_size, random_state=seed)

    # is exist keyword
    lbl_tr = np.sum(y_train, axis=0) > 0
    lbl_ts = np.sum(y_test, axis=0) > 0
    lbl_target = lbl_tr * lbl_ts

    print('{} target keywords'.format(sum(lbl_target)))

    y_train = y_train[:, lbl_target]
    y_test = y_test[:, lbl_target]
    keywords = keywords[lbl_target]

    return X_train, X_test, y_train, y_test, keywords

# text = X_test
# label = y_test
# keywords
def make_torchcsv_form(text, label, keywords, csvpath):
    with io.open(csvpath, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        keywords = keywords.astype('str').tolist()
        writer.writerow(['TEXT'] + keywords)

        for n, (t, l) in enumerate(zip(text, label)):
            t = [' '.join(x) for x in t]
            t = '. '.join(t) + '.'
            l = l.tolist()
            line = [t] + l
            writer.writerow(line)


def tokenizer(comment):
    MAX_CHARS = 20000
    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    return [x.text for x in nlp.tokenizer(comment) if x.text != " "]


def get_dataset(keywords, file_train, file_test, fix_length=500, lower=False, vectors=None):
    comment = data.Field(
        sequential=True,
        fix_length=fix_length,
        tokenize=tokenizer,
        pad_first=True,
        tensor_type=torch.cuda.LongTensor,
        lower=lower
    )

    fields_csv = []
    fields_csv.append(('TEXT', comment))
    for l in keywords:        
        label_field = (l, data.Field(use_vocab=False,
                                     sequential=False,
                                     tensor_type=torch.cuda.FloatTensor))
        # torch.cuda.ByteTensor

        fields_csv.append(label_field)

    train, test = data.TabularDataset.splits(
        path='datasets/', format='csv', skip_header=True,
        train=file_train, validation=file_test,
        fields=fields_csv)

    comment.build_vocab(
        train,
        max_size=50000,
        min_freq=20,
        vectors=vectors)

    with open('datasets/vocab.pickle', 'wb') as handle:
        pickle.dump(comment, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train, test, comment


def get_iterator(dataset, batch_size, train=True, shuffle=True, repeat=False):
    dataset_iter = data.Iterator(
        dataset, batch_size=batch_size, device=0,
        train=train, shuffle=shuffle, repeat=repeat,
        sort=False
    )
    return dataset_iter



if __name__ == '__main__':
    label = np.load('datasets/label.npy')
    text = np.load('datasets/text.npy')
    keywords = np.load('datasets/keyword.npy')

    X_train, X_test, y_train, y_test, keywords = split_train_test(text, label, keywords,
                                                                  test_size=1000,
                                                                  seed=12345)
    make_torchcsv_form(X_train, y_train, keywords,
                       csvpath='datasets/scopus_ai_train.csv')

    make_torchcsv_form(X_test, y_test, keywords,
                       csvpath='datasets/scopus_ai_test.csv')

    np.save('datasets/scopus_ai_keywords.npy', keywords)


