import torch
import torch.nn as nn
import numpy as np
import os
from make_dataset import get_dataset, get_iterator
from model.rnn_atten_1 import Encoder, Attention, Classifier
from train import learn, evaluate, load_pretrained_vectors

from model_config import model_argument

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    keywords = np.load('datasets/scopus_ai_keywords.npy')
    nlabels = len(keywords)
    model_argument['nlabels'] = nlabels

    file_train = 'scopus_ai_train.csv'
    file_test = 'scopus_ai_test.csv'
    train, test, comment = get_dataset(keywords, file_train, file_test, fix_length=500)

    train_iter = get_iterator(train, batch_size=32, train=True, shuffle=True, repeat=False)
    test_iter = get_iterator(test, batch_size=32, train=False, shuffle=False, repeat=False)

    cuda = torch.cuda.is_available() and model_argument['cuda']
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")

    print("[Corpus]: train: {}, test: {}, vocab: {}, labels: {}".format(
        len(train_iter.dataset), len(test_iter.dataset), len(comment.vocab), nlabels))

    vectors = load_pretrained_vectors(None)
    ntokens = len(comment.vocab)
    embedding = nn.Embedding(ntokens, model_argument['emsize'], padding_idx=1, max_norm=1)

    if model_argument['vectors']:
        embedding.weight.data.copy_(comment.vocab.vectors)

    encoder = Encoder(model_argument['emsize'],
                      model_argument['hidden'],
                      nlayers=model_argument['nlayers'],
                      dropout=model_argument['drop'],
                      bidirectional=model_argument['bi'],
                      rnn_type=model_argument['model'])

    attention_dim = model_argument['hidden'] if not model_argument['bi'] else 2 * model_argument['hidden']
    attention = Attention(attention_dim, attention_dim, attention_dim)
    model = Classifier(embedding, encoder, attention, attention_dim, nlabels)
    model.to(device)

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), model_argument['lr'], amsgrad=True)

    R = []
    P = []
    try:
        best_valid_loss = None

        for epoch in range(1, model_argument['epochs'] + 1):
            loss_train = learn(model, train_iter, optimizer, criterion)
            print("[{} loss]: {:.5f}".format('Train', loss_train))
            loss, r, p = evaluate(model, test_iter, criterion)
            R.append(r)
            P.append(p)

            if not best_valid_loss or loss < best_valid_loss:
                best_valid_loss = loss

        np.save('recall.npy', R)
        np.save('precison.npy', P)

    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")

    torch.save(model, 'training_history/mlc_20180903.pt')
    # model = torch.load('training_history/mlc_20180903.pt')

