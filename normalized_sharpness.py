import torch
import random
import numpy as np
from models.fcn_6 import *
import torch.nn as nn
from torch.autograd import Variable
from train_networks import train, load_dataset_CIFAR10
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


use_cuda = torch.cuda.is_available()
learning_rate = 0.1


def gradient(net, data, target, class_num):
	if use_cuda:
		data, target = data.cuda(), target.cuda()
		data, target = Variable(data).cuda(), Variable(target).cuda()
	else:
		data, target = Variable(data), Variable(target)

	optimizer = optim.SGD(net.parameters(), lr=learning_rate)
	optimizer.zero_grad()
	criterion = nn.CrossEntropyLoss()

	net_out = net(data)
	mu = torch.sum(net_out) / class_num
	new_out = net_out / torch.sqrt(torch.sum(torch.pow(net_out-mu,2)) / class_num)
	loss = criterion(new_out, target)
	loss.backward()
	grad = []
	for param in net.parameters():
		grad.append(param.grad)
	return grad


def Hessian(net, dataset):
	if (dataset == 'CIFAR10'):
		class_num = 10
		train_loader, test_loader = load_dataset_CIFAR10(1)
	elif (dataset == 'CIFAR100'):
		class_num = 100
		train_loader, test_loader = load_dataset_CIFAR10(1)

	hessian = []
	for p in net.parameters():
		hessian.append(torch.zeros(p.shape))

	num = 0
	for data, target in test_loader:
		# print(num)
		num += 1
		if num > 1:
			break

		# Change parameters and get two new networks.
		epsilon = []
		r = []
		net_plus =  FCN_test()
		net_minus = FCN_test()
		for (p, p_plus, p_minus) in zip(net.parameters(), net_plus.parameters(), net_minus.parameters()):
			ep = torch.randn(p.shape)
			epsilon.append(ep)
			norm = torch.norm(p.data)
			r.append(norm)
			change = ep * norm
			p_plus.data = p.data + change
			p_minus.data = p.data - change

		# Calculate gradient.
		grad_plus = gradient(net_plus, data, target, class_num)
		grad_minus = gradient(net_minus, data, target, class_num)

		# Sum hessian of each data.
		for i in range(0, len(hessian)):
			hessian[i] += epsilon[i] * (grad_plus[i]-grad_minus[i]) / (2 * r[i])

	# Average hessian.
	for i in range (0, len(hessian)):
		# hessian[i] = hessian[i] / len(test_loader.dataset)
		hessian[i] = hessian[i] / num
		hessian[i] = hessian[i].numpy()
	return np.array(hessian)


def sharpness_layer(weight, hessian):
	S = 0
	alpha = np.random.rand(len(weight))
	beta = np.random.rand(len(weight[0]))
	while True:
		s = 0
		sigma_row = np.exp(alpha)
		sigma_col = np.exp(beta)
		for row in range (0, len(weight)):
			for col in range (0, len(weight[row])):
				s += hessian[row][col]*pow(sigma_row[row]*sigma_col[col], 2)
				s += pow(weight[row][col], 2)/pow(sigma_row[row]*sigma_col[col], 2)
		if abs(S-s) < 0.0001:
			break
		else:
			if S > s:
				S = s
			SGDUpdate(L, alpha, beta)
	return S



def normalize_sharpness(net):
	hessian = Hessian(net, 'CIFAR10')
	print(hessian)
	# print(hessian)
	# print('test')
	# np.save('hessian1', hessian)
	# hessian = np.load('hessian.npy', allow_pickle=True)
	# S = 0
	# weights = []
	# for param in net.parameters():
	# 	weights.append(param.data.numpy())
	# weights = np.array(weights)
	#
	# for i in range (0, len(weights)):
	# 	S += sharpness_layer(weights[i], hessian[i])
	# return S




if __name__ == "__main__":
	net = FCN_test()
	# torch.save(net, net.name())
	# net = torch.load('fcn-test')
	normalize_sharpness(net)
