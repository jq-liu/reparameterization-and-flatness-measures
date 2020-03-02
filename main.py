import sys
import torch
from models.fcn_6 import *
from normalized_sharpness import normalize_sharpness
from train_networks import train, load_dataset_CIFAR10


if __name__ == "__main__":
	batch_size = (512, 256, 128, 64)
	learning_rate = (0.1, 0.05, 0.01, 0.005)
	init = ['normal', 'kaiming_uniform', 'uniform', 'xavier_normal',]
	for times in range(1, 11):
		for (bs, lr) in zip(batch_size, learning_rate):
			for i in init:
				print('bs=' + str(bs) + '\t' + 'lr=' + str(lr) + '\t' + 'times = ' + str(times) + '\t' + 'init=' + i)
				net = FCN_6()
				if i == 'xavier_normal':
					net.apply(weight_init_xavier_normal)
				elif i == 'normal':
					net.apply(weight_init_normal)
				elif i == 'kaiming_uniform':
					net.apply(weight_init_kaiming_uniform)
				elif i == 'uniform':
					net.apply(weight_init_uniform)
				train(net, 0, bs, lr, 'CIFAR10', times, i)



