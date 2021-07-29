import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from sklearn import metrics
import argparse
import timeit
import os

import poptorch

from model import MolecularGraphNeuralNetwork

def data_load(args):
    filename = os.path.join('dataset', '%s.pth' % args.dataset)
    data = torch.load(filename)
    dataset_train = data['dataset_train']
    dataset_test = data['dataset_test']
    N_fingerprints = data['N_fingerprints']

    '''Transform numpy data to torch tensor'''
    #TODO: use FP16 or FP32
    for index, (fingerprints, adjacency, molecular_size, property) in enumerate(dataset_train):
        fingerprints = torch.LongTensor(fingerprints)
        adjacency = torch.FloatTensor(adjacency)
        property = torch.LongTensor([int(property)])
        dataset_train[index] = (fingerprints, adjacency, molecular_size, property)

    for index, (fingerprints, adjacency, molecular_size, property) in enumerate(dataset_test):
        fingerprints = torch.LongTensor(fingerprints)
        adjacency = torch.FloatTensor(adjacency)
        property = torch.LongTensor([int(property)])
        dataset_test[index] = (fingerprints, adjacency, molecular_size, property)

    np.random.shuffle(dataset_train)

    return dataset_train, dataset_test, N_fingerprints

def train(dataset, net, optimizer, batch_size, epoch, opts):
    train_loss = 0
    net.train()

    data_train = poptorch.DataLoader(opts, data_train, batch_size, mode=poptorch.DataLoaderMode.Async)
    poptorch_model = poptorch.trainingModel(net, opts, optimizer)

    for batch_index, index in enumerate(range(0, len(dataset), batch_size), 1):
        data_batch = list(zip(*dataset[index:index+batch_size]))
        correct = torch.cat(data_batch[-1])

        predicted, loss = poptorch_model(net.forward(data_batch, correct))
        poptorch_model.setOptimizer(optimizer)
        train_loss += loss.item()
        
    print('epoch %4d batch %4d/%4d train_loss %5.3f' % \
            (epoch, batch_index, np.ceil(len(dataset) / batch_size), train_loss / batch_index), end='')

def test(dataset, net, batch_size, opts):
    net.eval()
    poptorch_model = poptorch.inferenceModel(net, options=opts)
    test_loss = 0
    y_score, y_true = [], []
    for batch_index, index in enumerate(range(0, len(dataset), batch_size), 1):
        data_batch = list(zip(*dataset[index:index+batch_size]))
        correct = torch.cat(data_batch[-1])
        with torch.no_grad():
            predicted, loss = poptorch_model(net.forward(data_batch, correct))
        test_loss += loss.item()
        score = F.softmax(predicted, 1).cpu()
        y_score.append(score)
        y_true.append(correct.cpu())

    y_score = np.concatenate(y_score)
    y_pred = [np.argmax(x) for x in y_score]
    y_true = np.concatenate(y_true)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[1,0]).flatten()

    if np.sum(y_pred) != 0:
        acc = metrics.accuracy_score(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_score[:,1])
        prec = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)

        print(' %s test_loss %5.3f test_auc %5.3F test_prec %5.3f test_recall %5.3f' % (confusion_matrix, test_loss / index, auc, prec, recall), end='')
    else:
        print(' %s test_loss %5.3f' % (confusion_matrix, test_loss / index), end='')

    return test_loss / batch_index

def main(args):
    #TODO: set prefer PopTorch options
    opts = poptorch.Options()
    opts.deviceIterations(100)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_train, dataset_test, N_fingerprints = data_load(args)

    print('# of training data samples:', len(dataset_train))
    print('# of test data samples:', len(dataset_test))

    n_output = 2
    net = MolecularGraphNeuralNetwork(N_fingerprints, 
            dim=args.dim, 
            layer_hidden=args.layer_hidden, 
            layer_output=args.layer_output, 
            n_output=n_output)
    print('# of model parameters:', sum([np.prod(p.size()) for p in net.parameters()]))

    if args.modelfile:
        net.load_state_dict(torch.load(args.modelfile))
        poptorch_model = poptorch.inferenceModel(net, options=opts)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    test_losses = []

    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()

        if epoch % args.decay_interval == 0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

        train(dataset_train, net, optimizer, args.batch_size, epoch, opts)
        test_loss = test(dataset_test, net, args.batch_size, opts)

        print(' %5.2f sec' % (timeit.default_timer() - epoch_start))

        test_losses.append(test_loss)

        if test_loss <= min(test_losses) and args.model_save:
            os.makedirs('model', exist_ok=True)
            torch.save(net.state_dict(), 'model/%5.3f.pth' % test_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # classification target is a binary value (e.g., drug or not).
    parser.add_argument('--dataset', default='hiv', choices=['hiv', 'postera'])
    parser.add_argument('--modelfile', default=None)
    parser.add_argument('--model_save', default=False)
    parser.add_argument('--dim', default=50, type=int)
    parser.add_argument('--layer_hidden', default=6, type=int)
    parser.add_argument('--layer_output', default=6, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_decay', default=0.99, type=float)
    parser.add_argument('--decay_interval', default=10, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--seed', default=123, type=int)
    args = parser.parse_args()
    print(vars(args))

    main(args)
