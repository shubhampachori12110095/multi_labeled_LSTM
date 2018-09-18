import numpy as np
import pandas as pd
import torch
import pickle
import itertools
torch.nn.Module.dump_patches = True
model = torch.load('training_history/mlc_20180903.pt')

##
data = pd.read_csv('datasets/scopus_ai_test_gttp.csv')
data = data.values

with open('datasets/vocab.pickle', 'rb') as handle:
    comment = pickle.load(handle)

'''
{'<unk>': 0, '<pad>': 1, '.': 2, ...
'''

def process_batch(batch):
    x = batch[:, -1]
    x = [xx.split(' ') for xx in x]
    for i in range(len(x)):
        x[i] = [xi.replace('.', '') for xi in x[i]]
        x[i].append('.')
    x = list(itertools.chain.from_iterable(x))
    return x


def string2id(x, comm=comment.vocab):
    pad_len = 1000 - len(x)
    x = ['<pad>'] * pad_len + x

    out = []
    for word in x:
        if comm.freqs[word] > 20:
            out.append(comm.stoi[word])
        elif word == '<pad>':
            out.append(comm.stoi[word])
        else:
            out.append(comm.stoi['<unk>'])
    return out

N_docs = 20
Nsim = 1000

target = []
index = []
predicted = []

for _ in range(Nsim):
    ind = np.random.choice(np.arange(len(data)), size=N_docs, replace=False)
    batch = data[ind]

    x = process_batch(batch)
    x = string2id(x, comm=comment.vocab)
    x = np.array(x, dtype='int64')
    x = np.expand_dims(x, axis=-1)

    x = torch.from_numpy(x)
    x = x.to('cuda')
    pred = model(x)
    prob = torch.sigmoid(pred[0])
    y = np.sum(batch[:, 2:-1], axis=0)
    y[y > 0] = 1
    y = y.astype('int')

    target.append(y)
    index.append(ind)
    predicted.append(prob.cpu().data.numpy())

from sklearn.metrics import recall_score, precision_score

Y = np.array(target)
P = np.array(predicted)

recalls, precisions = [], []
for target, pred in zip(Y, P):
    pred = pred > 0.2
    pred = pred.astype(int)
    pred = pred.reshape((1248,))

    recall = recall_score(target, pred)
    precision = precision_score(target, pred)
    recalls.append(recall)
    precisions.append(precision)
av_recall = np.mean(recalls)
av_precision = np.mean(precisions)

print('recall: {:.5f}  precision: {:.5f}'.format(av_recall, av_precision))