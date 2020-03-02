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
    testset_CIFAR10 = torchvision.datasets.CIFAR10(root='./data/CIFAR10/', train=False, download=True, transform=transform)
    testloader_CIFAR10 = torch.utils.data.DataLoader(testset_CIFAR10, batch_size=batch_size, shuffle=False, num_workers=1)
    print('Done loading dataset CIFAR10!')
    return trainloader_CIFAR10, testloader_CIFAR10


def train(net, epochs, batch_size, learning_rate, dataset, times, i):
    if dataset == 'CIFAR10':
        train_loader, test_loader = load_dataset_CIFAR10(batch_size)
    elif dataset == 'CIFAR100':
        train_loader, test_loader = load_dataset_CIFAR100(batch_size)

    if device == 'cuda':
        print('Use CUDA!')
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # create a loss function
    criterion = nn.CrossEntropyLoss()
    # end of one shot pruning
    timesPerEpoch = len(train_loader.dataset) / batch_size

    path = './times=' + str(times)
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + '/' +i
    if not os.path.exists(path):
        os.makedirs(path)

    path_weights = path + '/weights'
    if not os.path.exists(path_weights):
        os.makedirs(path_weights)
    path_weights = path_weights + '/bs=' + str(batch_size) + '_lr=' + str(learning_rate)
    if not os.path.exists(path_weights):
        os.makedirs(path_weights)
    path_log = path + '/log'
    if not os.path.exists(path_log):
        os.makedirs(path_log)


    PATH = path_weights + '/epoch=0'
    torch.save(net.state_dict(), PATH)
    epoch = 0
    test_arr_5 = []
    # for epoch in range(epochs):
    while True:
        train_correct = 0
        train_loss = 0
        train = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if device == 'cuda':
                # data, target = data.cuda(), target.cuda()
                data, target = data.to(device), target.to(device)
                data, target = Variable(data).cuda(), Variable(target).cuda()
            else:
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
        train_loss /= train
        train_accuracy = 1.*train_correct / train

        test_loss = 0
        test_correct = 0
        # test = 0
        for data, target in test_loader:
            with torch.no_grad():
                if device == 'cuda':
                    # data, target = data.cuda(), target.cuda()
                    data, target = data.to(device), target.to(device)
                    data, target = Variable(data).cuda(), Variable(target).cuda()
                else:
                    data, target = Variable(data), Variable(target)
            net_out = net(data)
            # sum up batch loss
            # test_loss += criterion(net_out, target).data[0]
            test_loss += criterion(net_out, target).item()
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            test_correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 1. * test_correct.item() / len(test_loader.dataset)

        print(
            'Train Epoch: {}  \tTest Average loss: {:.4f}\tTest Accuracy: {}/{} ({:.4f}%)\tTrain Average loss: {:.4f}\tTrain Accuracy: {}/{} ({:.4f}%)'
                .format(epoch+1, 100.*test_loss, test_correct,
                        len(test_loader.dataset), 100. * test_accuracy, 100.*train_loss, train_correct, train, 100.*train_accuracy))

        file_name = path_log + '/bs=' + str(batch_size) + '_lr=' + str(learning_rate)
        with open(file_name, "a") as f:
            data = (
            'Train Epoch: {}  \tTest Average loss: {:.4f}\tTest Accuracy: {}/{} ({:.4f}%)\tTrain Average loss: {:.4f}\tTrain Accuracy: {}/{} ({:.4f}%)'
                .format(epoch+1, 100.*test_loss, test_correct,
                        len(test_loader.dataset), 100. * test_accuracy, 100.*train_loss, train_correct, train, 100.*train_accuracy))
            f.write(data)
            f.write('\n')

        if (epoch+1) % log_interval == 0:
            PATH = path_weights + '/epoch=' + str(epoch+1)
            torch.save(net.state_dict(), PATH)

        # if epoch < 5:
        #     test_arr_5.append(test_accuracy)
        # else:
        #     ave = 1.*sum(test_arr_5) / 5
        #     if(abs(test_accuracy-ave) < 0.0005):
        #         if (epoch+1) % log_interval != 0:
        #             PATH = './weights/'+net.module.name() + '_epoch=' + str(epoch+1) + '_lr=' + str(learning_rate) + '_bs=' + str(batch_size)
        #             torch.save(net.state_dict(), PATH)
        #         break
        #     else:
        #         test_arr_5.pop(0)
        #         test_arr_5.append(test_accuracy)

        if train_accuracy > 0.9 or epoch > 1:
            if (epoch+1) % log_interval != 0:
                PATH = path_weights + '/epoch=' + str(epoch+1)
                torch.save(net.state_dict(), PATH)
            break
        epoch += 1


