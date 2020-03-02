'''(2.5, 0.8, 2.5, 0.5, 0.8, 0.5)'''
from models.fcn_6 import *

ratio = [2.5, 0.8, 2.5, 0.5, 0.8, 0.5]


def reparameterize(net, sym, ratio):
	for (param_net, param_sym, r)  in zip(net.parameters(), sym.parameters(), ratio):
		param_sym.data = param_net.data*r
	return sym



if __name__ == "__main__":
	net = FCN_test()
	# for param in net.parameters():
	# 	print(param.data)
	sym = FCN_test()
	sym = reparameterize(net, sym, ratio)
	print('Reparameterize End!')
