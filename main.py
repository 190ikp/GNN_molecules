import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from sklearn import metrics
import argparse
import timeit
import os

from model import MolecularGraphNeuralNetwork

def data_load(args, device):
    filename = os.path.join('dataset', '%s.pth' % args.dataset)
    data = torch.load(filename)
    dataset_train = data['dataset_train']
    dataset_test = data['dataset_test']
    N_fingerprints = data['N_fingerprints']

    '''Transform numpy data to torch tensor'''
    for index, (fingerprints, adjacency, molecular_size, property) in enumerate(dataset_train):
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        if args.task == 'classification':
            property = torch.LongTensor([int(property)]).to(device)
        if args.task == 'regression':
            property = torch.FloatTensor([[float(property)]]).to(device)
        dataset_train[index] = (fingerprints, adjacency, molecular_size, property)

    for index, (fingerprints, adjacency, molecular_size, property) in enumerate(dataset_test):
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        if args.task == 'classification':
            property = torch.LongTensor([int(property)]).to(device)
        if args.task == 'regression':
            property = torch.FloatTensor([[float(property)]]).to(device)
        dataset_test[index] = (fingerprints, adjacency, molecular_size, property)

    np.random.shuffle(dataset_train)

    return dataset_train, dataset_test, N_fingerprints

def train(dataset, net, optimizer, loss_function, batch_train, epoch):
    train_loss = 0
    net.train()

    for batch_index, index in enumerate(range(0, len(dataset), batch_train), 1):
        data_batch = list(zip(*dataset[index:index+batch_train]))
        correct = torch.cat(data_batch[-1])

        optimizer.zero_grad()
        predicted = net.forward(data_batch)
        loss = loss_function(predicted, correct)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    print('epoch %4d batch %4d/%4d train_loss %5.3f' % \
            (epoch, batch_index, np.ceil(len(dataset) / batch_train), train_loss / batch_index), end='')

def test(dataset, net, loss_function, batch_test):
    net.eval()
    test_loss = 0
    y_score, y_true = [], []
    for batch_index, index in enumerate(range(0, len(dataset), batch_test), 1):
        data_batch = list(zip(*dataset[index:index+batch_test]))
        correct = torch.cat(data_batch[-1])
        with torch.no_grad():
            predicted = net.forward(data_batch)
        loss = loss_function(predicted, correct)
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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    dataset_train, dataset_test, N_fingerprints = data_load(args, device)

    print('# of training data samples:', len(dataset_train))
    print('# of test data samples:', len(dataset_test))

    n_output = 1 if args.task == 'regression' else 2
    net = MolecularGraphNeuralNetwork(N_fingerprints, 
            dim=args.dim, 
            layer_hidden=args.layer_hidden, 
            layer_output=args.layer_output, 
            n_output=n_output).to(device)
    print('# of model parameters:', sum([np.prod(p.size()) for p in net.parameters()]))

    if args.modelfile:
        net.load_state_dict(torch.load(args.modelfile))

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    loss_function = F.cross_entropy if args.task == 'classification' else F.mse_loss

    test_losses = []

    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()

        if epoch % args.decay_interval == 0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

        train(dataset_train, net, optimizer, loss_function, args.batch_train, epoch)
        test_loss = test(dataset_test, net, loss_function, args.batch_test)

        print(' %5.2f sec' % (timeit.default_timer() - epoch_start))

        test_losses.append(test_loss)

        if test_loss <= min(test_losses) and args.model_save:
            os.makedirs('model', exist_ok=True)
            torch.save(net.state_dict(), 'model/%5.3f.pth' % test_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # classification target is a binary value (e.g., drug or not).
    # regression target is a real value (e.g., energy eV).
    parser.add_argument('--task', default='classification', choices=['classification', 'regression'])
    parser.add_argument('--dataset', default='hiv', choices=['hiv', 'photovoltaic', 'postera'])
    parser.add_argument('--modelfile', default=None)
    parser.add_argument('--model_save', default=False)
    parser.add_argument('--dim', default=50, type=int)
    parser.add_argument('--layer_hidden', default=6, type=int)
    parser.add_argument('--layer_output', default=6, type=int)
    parser.add_argument('--batch_train', default=32, type=int)
    parser.add_argument('--batch_test', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_decay', default=0.99, type=float)
    parser.add_argument('--decay_interval', default=10, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--seed', default=123, type=int)
    args = parser.parse_args()
    print(vars(args))

    main(args)
