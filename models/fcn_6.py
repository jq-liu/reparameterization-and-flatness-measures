'''FCN_6 in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F


class FCN_6(nn.Module):
	def __init__(self):
		super(FCN_6, self).__init__()
		self.fc1 = nn.Linear(3*32*32, 32*32)
		self.fc2 = nn.Linear(32*32, 32*16)
		self.fc3 = nn.Linear(32*16, 32*8)
		self.fc4 = nn.Linear(32*8, 32*4)
		self.fc5 = nn.Linear(32*4, 64)
		self.fc6 = nn.Linear(64, 10)
	def	forward(self, x):
		x = x.view(x.size(0), -1)
		out = F.relu(self.fc1(x))
		out = F.relu(self.fc2(out))
		out = F.relu(self.fc3(out))
		out = F.relu(self.fc4(out))
		out = F.relu(self.fc5(out))
		out = self.fc6(out)
		return out
	def name(self):
		return "fcn-6"


def weight_init_xavier_normal(m):
	if isinstance(m, nn.Linear):
		nn.init.xavier_normal_(m.weight)

def weight_init_normal(m):
	if isinstance(m, nn.Linear):
		nn.init.normal_(m.weight, mean=0, std=0.1)

def weight_init_kaiming_uniform(m):
	if isinstance(m, nn.Linear):
		nn.init.kaiming_uniform_(m.weight)

def weight_init_uniform(m):
	if isinstance(m, nn.Linear):
		nn.init.uniform_(m.weight, a=-0.1, b=0.1)

class FCN_test(nn.Module):
	def __init__(self):
		super(FCN_test, self).__init__()
		self.fc1 = nn.Linear(3*32*32, 20)
		self.fc2 = nn.Linear(20, 10)
	def	forward(self, x):
		x = x.view(x.size(0), -1)
		out = F.relu(self.fc1(x))
		out = self.fc2(out)
		return out
	def name(self):
		return "fcn-test"