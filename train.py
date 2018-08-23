import time
import numpy as np
import torch
from model_config import model_argument
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def learn(model, data, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch_num, batch in enumerate(data):
        model.zero_grad()

        y = []
        for k in batch.fields:
            if k == 'TEXT':
                x = batch.__dict__.get(k)
            else:
                y.append(batch.__dict__.get(k))

        y = torch.stack(y, dim=1)

        logits, _ = model(x)
        logits = logits.view(-1, model_argument['nlabels'])
        loss = criterion(logits, y)
        total_loss += float(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model_argument['clipgrad'])
        optimizer.step()

    return total_loss / len(data)


def evaluate(model, data, criterion, type='Valid'):
    model.eval()
    total_loss = 0

    Y = []
    P = []
    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            y = []
            for k in batch.fields:
                if k == 'TEXT':
                    x = batch.__dict__.get(k)
                else:
                    y.append(batch.__dict__.get(k))

            y = torch.stack(y, dim=1)

            logits, _ = model(x)
            logits = logits.view(-1, model_argument['nlabels'])
            loss = criterion(logits, y)
            total_loss += float(loss)

            # TODO: recall / precision
            prob = torch.nn.functional.sigmoid(logits)
            P.append(prob.cpu().data.numpy())
            Y.append(y.cpu().data.numpy())

    P = np.concatenate(P, axis=0)
    Y = np.concatenate(Y, axis=0)

    recalls, precisions = [], []
    for target, pred in zip(Y, P):
        pred = pred > 0.5
        pred = pred.astype(int)

        recall = recall_score(target, pred)
        precision = precision_score(target, pred)
        recalls.append(recall)
        precisions.append(precision)
    av_recall = np.mean(recalls)
    av_precision = np.mean(precisions)

    print()
    print("[{} loss]: {:.5f}".format(type, total_loss / len(data)))
    print("[{} recall]: {:.5f}".format(type, av_recall))
    print("[{} precision]: {:.5f}".format(type, av_precision))

    return total_loss / len(data), av_recall, av_precision


def load_pretrained_vectors(dim):
    pretrained_GloVe_sizes = [50, 100, 200, 300]

    if dim in pretrained_GloVe_sizes:
        # Check torchtext.datasets.vocab line #383
        # for other pretrained vectors. 6B used here
        # for simplicity
        name = 'glove.{}.{}d'.format('6B', str(dim))
        return name
    return None

