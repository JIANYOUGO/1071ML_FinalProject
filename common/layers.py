# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 10:29:18 2019

@author: JoshuaChen
"""
import numpy as np
from common.functions import *


#class
class Relu:
	def __init__(self):
		self.mask = None

	def forward(self,x):
		self.mask = (x <= 0)
		out = x.copy()
		out[self.mask] = 0

		return out

	def backward(self,dout):
		dout[self.mask] = 0
		dx = dout

		return dx

class Sigmoid:
	def __init__(self):
		self.out = None

	def forward(self,x):
		out = sigmoid(x)
		self.out = out

		return out

	def backward(self,dout):
		dx = dout * self.out * (1.0 - self.out)

		return dx

class Affine:
	def __init__(self,W,b):
		self.W = W
		self.b = b
		self.x = None
		self.original_x_shape = None
		self.dW = None
		self.db = None

	def forward(self,x):
		#將可能為(3,4,2)三維的矩陣轉化為 .shape[0]固定，其餘合併。 -1 代表剩下的由python自己分配
		#所以 x.shape = (3,8)。 若為(3,4,2,2) 則轉為 (3,16)
		self.original_x_shape = x.shape
		x = x.reshape(x.shape[0],-1)
		self.x = x

		out = np.dot(self.x,self.W) + self.b

		return out

	def backward(self,dout):
		dx = np.dot(dout,self.W.T)
		self.dW = np.dot(self.x.T,dout)
		self.db = np.sum(dout,axis=0)

		dx = dx.reshape(*self.original_x_shape)
		return dx

class SoftmaxWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None
		#one-hot vector
		self.t = None

	def forward(self,x,t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y,self.t)

		return self.loss

	def backward(self,dout=1):
		batch_size = self.t.shape[0]
		if self.t.size == self.y.size:
			dx = (self.y - self.t) / batch_size
		else:
			dx = self.y.copy()
			#減1是因為softmax會把值壓縮在0~1之間，完成正規化。
			dx[np.arange(batch_size),self.t] -=1
			dx = dx / batch_size
			
		return dx
					
#
