'''Train CIFAR10 with PyTorch'''
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

# batch_size = 128
# learning_rate = 0.1
# epochs = 50
log_interval = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_dataset_CIFAR10(batch_size):
    print('Start loading dataset CIFAR10...')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    trainset_CIFAR10 = torchvision.datasets.CIFAR10(root='./data/CIFAR10/', train=True, download=True, transform=transform)
    trainloader_CIFAR10 = torch.utils.data.DataLoader(trainset_CIFAR10, batch_size=batch_size, shuffle=True, num_workers=1)
    # trainloader_CIFAR10 = torch.utils.data.DataLoader(trainset_CIFAR10, batch_size=batch_size, shuffle=False, num_workers=1)
    testset_CIFAR10 = torchvision.datasets.CIFAR10(root='./data/CIFAR10/', train=False, download=True, transform=transform)
    testloader_CIFAR10 = torch.utils.data.DataLoader(testset_CIFAR10, batch_size=batch_size, shuffle=False, num_workers=1)
    print('Done loading dataset CIFAR10!')
    return trainloader_CIFAR10, testloader_CIFAR10

def load_dataset_MNIST(batch_size):
    print('Start loading dataset MNIST...')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ])
    trainset_MNIST = torchvision.datasets.MNIST(root='./data/MNIST/', train=True, download=True, transform=transform)
    trainloader_MNIST = torch.utils.data.DataLoader(trainset_MNIST, batch_size=batch_size, shuffle=True)
    # trainloader_MNIST = torch.utils.data.DataLoader(trainset_MNIST, batch_size=batch_size, shuffle=False, num_workers=1)
    testset_MNIST = torchvision.datasets.MNIST(root='./data/MNIST/', train=False, download=True, transform=transform)
    testloader_MNIST = torch.utils.data.DataLoader(testset_MNIST, batch_size=batch_size, shuffle=False)
    print('Done loading dataset MNIST!')
    return trainloader_MNIST, testloader_MNIST


def train(net, batch_size, learning_rate, dataset, n, epo):
    if dataset == 'CIFAR10':
        train_loader, test_loader = load_dataset_CIFAR10(batch_size)
    elif dataset == 'CIFAR100':
        train_loader, test_loader = load_dataset_CIFAR100(batch_size)
    elif dataset == 'MNIST':
        train_loader, test_loader = load_dataset_MNIST(batch_size)

    net = net.to(device)
    if 'cuda' in device:
        print('Use CUDA!')
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # create a loss function
    criterion = nn.CrossEntropyLoss()
    # end of one shot pruning
    timesPerEpoch = len(train_loader.dataset) / batch_size

    path_weights = './net'
    if not os.path.exists(path_weights):
        os.makedirs(path_weights)

    path_log = './log'
    if not os.path.exists(path_log):
        os.makedirs(path_log)

    file_name = path_log + '/' + str(n)
    with open(file_name, "a") as f:
        data = ('bs=' + str(batch_size) + '\t' + 'lr=' + str(learning_rate) + '\t' + 'network=' + str(n))
        f.write(data)
        f.write('\n')

    epoch = 0
    test_arr_5 = []
    prev_loss = 100
    # for epoch in range(epochs):
    while True:
        net.train()
        train_correct = 0
        train_loss = 0
        train = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = net_out.max(1)
            train += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        train_loss  = train_loss * batch_size / len(train_loader.dataset)
        train_accuracy = 1. * train_correct / len(train_loader.dataset)

        net.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data, target = Variable(data), Variable(target)
                net_out = net(data)
                # sum up batch loss
                test_loss += criterion(net_out, target).item()
                _, pred = net_out.data.max(1)  # get the index of the max log-probability
                test_correct += pred.eq(target.data).sum().item()
        test_loss = test * batch_size / len(test_loader.dataset)
        test_accuracy = 1. * test_correct / len(test_loader.dataset)

        train_correct = 0
        train_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data, target = Variable(data), Variable(target)
                net_out = net(data)
                train_loss += criterion(net_out, target).item()
                _, predicted = net_out.max(1)
                train_correct += predicted.eq(target).sum().item()
        train_loss  = train_loss * batch_size / len(train_loader.dataset)
        train_accuracy = 1. * train_correct / len(train_loader.dataset)

        print(
            'Train Epoch: {}  \tTest Average loss: {:.6f}\tTest Accuracy: {}/{} ({:.4f}%)\tTrain Average loss: {:.6f}\tTrain Accuracy: {}/{} ({:.4f}%)'
                .format(epoch + 1, test_loss, test_correct,
                        len(test_loader.dataset), 100. * test_accuracy, train_loss, train_correct, len(train_loader.dataset),
                        100. * train_accuracy))

        with open(file_name, "a") as f:
            data = (
                'Train Epoch: {}  \tTest Average loss: {:.6f}\tTest Accuracy: {}/{} ({:.4f}%)\tTrain Average loss: {:.6f}\tTrain Accuracy: {}/{} ({:.4f}%)'
                    .format(epoch + 1, test_loss, test_correct,
                            len(test_loader.dataset), 100. * test_accuracy, train_loss, train_correct, len(train_loader.dataset),
                            100. * train_accuracy))
            f.write(data)
            f.write('\n')

        # if (epoch+1) % log_interval == 0:
        #     PATH = path_weights + '/epoch=' + str(epoch+1)
        #     torch.save(net.state_dict(), PATH)

        if epoch < 5:
            test_arr_5.append(test_accuracy)
        else:
            ave = 1. * sum(test_arr_5) / 5
            if ave <= 0.2:
                PATH = path_weights + '/' + str(n)
                torch.save(net.state_dict(), PATH)
                break
            else:
                test_arr_5.pop(0)
                test_arr_5.append(test_accuracy)

        # if (train_loss < 0.0001 and prev_loss < 0.0001) or epoch >= 50:
        # if epoch >= epo:
        if (train_loss < 0.001 and prev_loss < 0.001) or epoch >= 250:
            PATH = path_weights + '/' + str(n)
            torch.save(net.state_dict(), PATH)
            break
        epoch += 1
        prev_loss = train_loss