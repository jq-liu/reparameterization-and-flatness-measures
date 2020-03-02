import torch
import random
import numpy as np
from models.fcn_6 import *
import torch.nn as nn
from torch.autograd import Variable
from train_networks import load_dataset_CIFAR10
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# np.set_printoptions(threshold=np.inf)

learning_rate = 0.1

def hessian(net):
	train_loader, test_loader = load_dataset_CIFAR10(1)
	num = 1
	for data, target in test_loader:
		if num > 1:
			break

		data, target = Variable(data), Variable(target)
		criterion = nn.CrossEntropyLoss()

		net_out = net(data)
		loss = criterion(net_out, target)
		# calculate the first order gradients
		grad1st = torch.autograd.grad(loss, net.parameters(), create_graph=True)

		# calculate the second order gradients
		net_grad2rd = []
		# get the first order gradients of each layer
		for layer_i, (layer_g, layer_p) in enumerate(zip(grad1st, net.parameters())):
			layer_grad2rd = []
			# get the first order gradients of each neuron
			for neuron_i, neuron_g in enumerate(layer_g):
				neuron_grad2rd = []
				print("Calculate the Hessian of the " + str(layer_i) + "-th layer and the " + str(
					neuron_i) + "-th neuron.")
				# calculate the second order gradients for bias
				if neuron_g.ndim == 0:
					grad2rd = torch.autograd.grad(neuron_g, layer_p, create_graph=True)
					neuron_grad2rd.append(torch.index_select(grad2rd[0], 0, torch.tensor(neuron_i)).cpu().data.numpy())
				else:
					# calculate individual second order gradients for each individual weight of one neuron
					for g in neuron_g:
						grad2rd = torch.autograd.grad(g, layer_p, create_graph=True)
						# only the diagonal of Hessian is useful
						neuron_grad2rd.append(torch.index_select(grad2rd[0], 0, torch.tensor(neuron_i)).cpu().data.numpy()[0])
				layer_grad2rd.append(np.array(neuron_grad2rd))
			net_grad2rd.append(np.array(layer_grad2rd))
		net_grad2rd = np.array(net_grad2rd)
		np.save('hessian', net_grad2rd)
		num += 1


if __name__ == "__main__":
	net = FCN_test()
	hessian(net)
	# b = np.load('hessian.npy', allow_pickle=True)
	# print(b)


