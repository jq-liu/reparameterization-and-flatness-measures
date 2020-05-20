import os
import torch
import random
import numpy as np
import torchvision
from collections import OrderedDict
from models.fcn_6 import *
from models.lenet import *
from models.fcn_3 import *
from models.fcn_4 import *
import torch.nn as nn
from torch.autograd import Variable
from train_networks import load_dataset_CIFAR10
from train_networks import load_dataset_MNIST
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils import data


root = './FCN_4/'

net_path = root + 'net/'
flatness_path = root + 'flatness'
if not os.path.exists(flatness_path):
	os.makedirs(flatness_path)

hessian_path = root + 'hessian'
if not os.path.exists(hessian_path):
	os.makedirs(hessian_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if 'cuda' in device:
	print('Use CUDA!')
else:
	print('No CUDA!')


def neuron_wise_measure(net, net_name, dataset):
	if dataset == 'CIFAR10':
		train_loader, test_loader = load_dataset_CIFAR10(50000)
	elif dataset == 'CIFAR100':
		train_loader, test_loader = load_dataset_CIFAR100(50000)
	elif dataset == 'MNIST':
		train_loader, test_loader = load_dataset_MNIST(60000)

	for batch_idx, (data, target) in enumerate(train_loader):
		print('Start calculation for flatness!')
	net = net.to(device)
	data, target = data.to(device), target.to(device)
	data, target = Variable(data), Variable(target)
	criterion = nn.CrossEntropyLoss()

	net_out = net(data)
	loss = criterion(net_out, target)
	net_measure = []
	for layer_i, layer_p in enumerate(net.parameters()):

		# skip the bias
		if layer_i%2 == 1:
			continue

		shape = layer_p.shape
		layer_g = grad(loss, layer_p, create_graph=True, retain_graph=True)[0]

		layer_measure = []
		for column in range(shape[1]):
			if column % 50 == 0:
				print("Calculate the flatness of the " + str(layer_i) + "-th layer and the " + str(column) + "-th column.")
			column_p = layer_p[:, column]
			hessian = []
			for neuron_g in layer_g:
				grad2rd = grad(neuron_g[column], layer_p, retain_graph=True)
				hessian.append(grad2rd[0][:, column].data.cpu().numpy().flatten())

			weight_ij = np.reshape(column_p.detach().cpu().numpy(),(1,shape[0]))
			m = weight_ij.dot(np.array(hessian)).dot(np.transpose(weight_ij))[0][0]
			layer_measure.append(m)
		net_measure.append(np.array(layer_measure))
	np.save(flatness_path + '/' + net_name, net_measure)

def fcn_measure():

	for i in range(1, 41):
		net = FCN_4()
		print(str(i) + '-th network')
		if 'cuda' in device:
			print('Use CUDA!')
			net = torch.nn.DataParallel(net)
			net.load_state_dict(torch.load(net_path+str(i)))
			cudnn.benchmark = True
		else:
			state_dict = torch.load(net_path+str(i), map_location=torch.device('cpu'))
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				name = k[7:]  # remove module.
				new_state_dict[k] = v
			net.load_state_dict(new_state_dict)
		neuron_wise_measure(net, str(i), 'MNIST')



if __name__ == "__main__":
	fcn_measure()






