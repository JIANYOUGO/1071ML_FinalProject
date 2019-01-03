# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 12:20:11 2019

@author: JoshuaChen
"""
import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet


#
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=1000, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

#


#
for i in range(iters_num):
	#隨機抽樣 1~6000，抽100個
	batch_mask = np.random.choice(train_size,batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	grad = network.gradient(x_batch,t_batch)
    #grad = numerical_gradient(x_batch,t_batch)

	for key in ('W1','b1','W2','b2'):
		network.params[key] -= learning_rate * grad[key]

	loss = network.loss(x_batch,t_batch)
	train_loss_list.append(loss)

	if i%iter_per_epoch == 0:
		train_acc = network.accuracy(x_train,t_train)
		test_acc = network.accuracy(x_test,t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print("train acc : " + str(train_acc) + "  ||  test acc : " + str(test_acc))

#


#print chart
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.show()

q = np.arange(len(train_loss_list))
plt.plot(q,train_loss_list)
plt.xlabel("Loss function")
plt.ylabel("iteration")
plt.show()

#print chart end

