import numpy as np
from make_dataset import get_dataset, get_iterator
from model.rnn_atten_1 import Encoder, Attention, Classifier
import os, sys
import time
import numpy as np
import torch
import torch.nn as nn

from datasets import dataset_map
from torchtext.vocab import GloVe


def update_stats(accuracy, confusion_matrix, logits, y):
  _, max_ind = torch.max(logits, 1)
  equal = torch.eq(max_ind, y)
  correct = int(torch.sum(equal))

  for j, i in zip(max_ind, y):
    confusion_matrix[int(i),int(j)]+=1

  return accuracy + correct, confusion_matrix


def learn(model, data, optimizer, criterion, args):
  model.train()
  accuracy, confusion_matrix = 0, np.zeros((args.nlabels, args.nlabels), dtype=int)
  t = time.time()
  total_loss = 0
  for batch_num, batch in enumerate(data):
    model.zero_grad()
    x, lens = batch.text
    y = batch.label

    logits, _ = model(x)
    loss = criterion(logits.view(-1, args.nlabels), y)
    total_loss += float(loss)
    accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    print("[Batch]: {}/{} in {:.5f} seconds".format(
          batch_num, len(data), time.time() - t), end='\r', flush=True)
    t = time.time()

  print()
  print("[Loss]: {:.5f}".format(total_loss / len(data)))
  print("[Accuracy]: {}/{} : {:.3f}%".format(
        accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))
  print(confusion_matrix)
  return total_loss / len(data)


if __name__ == '__main__':
    keywords = np.load('datasets/scopus_ai_keywords.npy')
    train, test = get_dataset(keywords, fix_length=500)
    train_iter = get_iterator(train, batch_size=32, train=True, shuffle=True, repeat=False)
    test_iter = get_iterator(test, batch_size=32, train=True, shuffle=False, repeat=False)


