# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import pickle
import pandas as pd

from nltk.corpus import stopwords

from preprocess.utils import remove_garbage_characters
from preprocess.utils import documents2words
from preprocess.utils import remove_stopwords
from preprocess.utils import lemmatization

"""
    Preprocess corpus.
"""

if __name__ == '__main__':

    root_dir = 'D:/projects/PROJECT_kisti/data_by_keyword/'  # directory to where the data lies

    papers = list()
    for cl in os.listdir(root_dir):
        for p in os.listdir(os.path.join(root_dir, cl)):
            file = os.path.join(root_dir, cl, p)
            with open(file, 'rb') as f:
                paper = json.load(f)

            idx = paper['id']
            title = paper['title'].replace('\n ', '')
            summary = paper['summary'].replace('\n', ' ')

            papers.append([idx, cl, title, summary])

    # Make a pandas dataframe
    df = pd.DataFrame(papers, columns=['idx', 'cluster', 'title', 'summary'])

    # Get summaries & titles
    summaries = df['summary'].tolist()
    titles = df['title'].tolist()

    # Remove garbage characters
    summaries = remove_garbage_characters(summaries)
    titles = remove_garbage_characters(titles)

    # Documents to words
    summaries = documents2words(summaries)
    titles = documents2words(titles)

    # Remove stopwords
    stopwords_ = stopwords.words('english')
    stopwords_.extend(['ctl*', 'ctl*k', 'ctl*k.', 'ctl', 'empha'])
    summaries = remove_stopwords(summaries, stopwords_)
    titles = remove_stopwords(titles, stopwords_)

    # Perform lemmatization (takes some time)
    summaries = lemmatization(summaries, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    titles = lemmatization(titles, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Remove double-nested brackets (only for titles)
    titles = [title[0] for title in titles]

    # Save result
    save = True
    if save:
        df['token_summary'] = summaries
        df['token_title'] = titles
        df.to_csv('./data/tokenized_data.csv')
        with open('./data/tokenized_data.pkl', 'wb') as writefile:
            pickle.dump(df, writefile)
