import torch
import torch.nn as nn
import numpy as np
from make_dataset import get_dataset, get_iterator
from model.rnn_atten_1 import Encoder, Attention, Classifier
from train import learn, evaluate, load_pretrained_vectors

from model_config import model_argument

if __name__ == '__main__':
    keywords = np.load('datasets/scopus_ai_keywords.npy')
    nlabels = len(keywords)
    train, test, comment = get_dataset(keywords, fix_length=500)

    train_iter = get_iterator(train, batch_size=32, train=True, shuffle=True, repeat=False)
    test_iter = get_iterator(test, batch_size=32, train=True, shuffle=False, repeat=False)

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

    try:
        best_valid_loss = None

        for epoch in range(1, model_argument['epochs'] + 1):
            learn(model, train_iter, optimizer, criterion)
            loss = evaluate(model, test_iter, optimizer, criterion)

            if not best_valid_loss or loss < best_valid_loss:
                best_valid_loss = loss

    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")