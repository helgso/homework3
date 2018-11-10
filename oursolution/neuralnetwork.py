#!/usr/bin/env python

import os
# Set current working directory as this file's path
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import json
import sys

import mnist_reader
import numpy as np

from sklearn.utils import shuffle

def init_weights(shape):
    return np.random.uniform(-1/np.sqrt(shape[0]), 1/np.sqrt(shape[0]), shape)

def softmax(x, axis=0):
    a = np.exp(x - np.amax(x, axis=axis, keepdims=True))
    return a / np.sum(a, axis=axis, keepdims=True)

def softmax_prime(a, y, axis=0):
    return  

def relu(z):
    return np.maximum(z, 0)

def vectorof(x):
    if len(x.shape) == 1:
        return np.expand_dims(x, axis=1)
    if len(x.shape) == 2 and x.shape[0] == 1:
        return x.T
    return x

def relu_prime(z):
    return np.greater(z, 0).astype(int)

def log_loss(o_y):
    return -np.log(o_y)

def gradient_approx(x, func, eps=None):
    if eps is None or eps < 10 ** -6 or eps > 10 ** -4:
        eps = np.random.uniform(10 ** -6, 10 ** -4)
    return (func(x+eps) - func(x-eps)) / (2 * eps)

def about_equal(a, b):
    if a == 0. and b == 0.5 or b == 0. and a == 0.5:
        return True
    return 0.99 <= a/b <= 1.01


def main():
    """
    # Circles.txt training data
    #Questions 1,2
    circles_data = np.loadtxt('data/circles/circles.txt')
    circles_train = circles_data[:,:-1]
    circles_target = vectorof(circles_data[:,-1].astype(int))
    
    circles = Network([2,2,2])
    circles.finite_difference_gradient_check(vectorof(circles_train[0,:]),
            vectorof(circles_target[0,:]))
<<<<<<< HEAD

    #Question 6
    circles.finite_difference_gradient_check_mat(nn.vectorof(circles_train),nn.vectorof(circles_target));

=======
    """
>>>>>>> c8710f0f852242ab78c46b19dc2a1916258604a1
    # MNIST training data
    x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    
    model = Network([784,300,10])
    model.train(x_train, y_train, epochs=10)

class Network(object):
    def __init__(
        self,
        num_neurons
    ):
        """
            Initialization method

        :param num_neurons: A list of the amount of neurons in each layer [input, hidden_1, ..., hidden_n, output]
        """
        self.layer_sizes = num_neurons
        self.b = [np.zeros((layer_size, 1)) for layer_size in self.layer_sizes[1:]]
        self.w = [
            init_weights(size) for size in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]

        print("Weights: %s" % self.w)

    def finite_difference_gradient_check(
        self,
        test_data,
        test_target,
        display = True
    ):
        """
        Used to estimate the gradient numerically. Used to check our gradient computation
        :param test_data: An array of examples+labels we want to verify our back propagation algorithm with
        """
        x = test_data
        y = test_target
        output, loss = self.fprop(x, y)

        nabla_L = -1/output
        approx_L = gradient_approx(output, log_loss)

        #nabla_s = output * (1 - output)
        #approx_s = gradient_approx(relu(self.zs[-1]), softmax)
    
        nabla_a = relu_prime(self.zs[-1])
        approx_a = gradient_approx(self.zs[-1], relu)
        
        for i in range(len(output)):
            assert about_equal(nabla_L[i], approx_L[i])
            assert about_equal(nabla_a[i], approx_a[i])
        
        if display:
            print('gradient: ')
            print(nabla_L)
            print(nabla_a)
            print('finite difference: ')
            print(approx_L)
            print(approx_a)

<<<<<<< HEAD
    def finite_difference_gradient_check_mat(
        self,
        test_data,
        test_target,
        display = True
    ):
        """
        Used to estimate the gradient numerically. Used to check our gradient computation

        :param test_data: An array of examples+labels we want to verify our back propagation algorithm with
        """
        x = test_data
        y = test_target
        output, loss = self.fprop_mat(x, y)

        nabla_L = -1/output
        approx_L = gradient_approx(output, log_loss)

        #nabla_s = output * (1 - output)
        #approx_s = gradient_approx(relu(self.zs[-1]), softmax)
    
        nabla_a = relu_prime(self.zs[-1])
        approx_a = gradient_approx(self.zs[-1], relu)

        for i in range(len(output)):
            for j in range(len(output[0])):
                print("i %s"%i)
                print("j %s"%j)
                print(nabla_L[i][j])
                print(approx_L[i][j])
                assert about_equal(nabla_L[i][j], approx_L[i][j])
                assert about_equal(nabla_a[i][j], approx_a[i][j])
        
        if display:
            print('gradient: ')
            print(nabla_L)
            print(nabla_a)
            print('finite difference: ')
            print(approx_L)
            print(approx_a)

    def train(
        self,
        x_train,
        y_train,
        epochs,
        mini_batch_size=1,
        step_size=0.0001
    ):
        """
        Train the Network

        :param training_data: An array of all training examples+labels we will use
        :param epochs: The amount of epochs we will use to train
        :param mini_batch_size: The size of our mini-batches
        :param step_size: The learning rate
        """
=======
    def train(self, x_train, y_train, epochs, mini_batch_size=1, step_size=0.0001):
>>>>>>> c8710f0f852242ab78c46b19dc2a1916258604a1
        print("Training ...")

        next_percentage = 0.1
        percentage_increment = 0.1
        for epoch in range(epochs):
            shuffle(x_train, y_train)
            mini_batches = [
                    zip(x_train[k:k+mini_batch_size], y_train[k:k+mini_batch_size])
                for k in range(0, len(x_train), mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_network_parameters(mini_batch, step_size)

            # Printing progress every percentage_increment percent
            percentage_done = (1.0*epoch)/epochs
            if percentage_done > next_percentage:
                print("{}% done".format(percentage_done))
                next_percentage += percentage_increment

    def update_network_parameters(
        self,
        mini_batch,
        step_size
    ):
        # TODO: Define all the gradients and update our weights and biases
        # Copy-paste from network2.py (lmbda regularization term removed.
        # We need to add our elastic-net regularization)
        nabla_b = [np.zeros(b.shape) for b in self.b]
        nabla_w = [np.zeros(w.shape) for w in self.w]

        for x, y in mini_batch:
            print(x.shape)
            print(y.shape)
            o, L = self.fprop(x,y)
            delta_nabla_b, delta_nabla_w = self.bprop(o, y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.w = [w - (step_size / len(x)) * nw for w, nw in zip(self.w, nabla_w)]
        self.b = [b - (step_size / len(x)) * nb for b, nb in zip(self.b, nabla_b)]


    def fprop(self, x, y):
        x = vectorof(x)
        y = vectorof(y)
        self.zs = [x]
        z = x
        for w, b in zip(self.w, self.b):
            z = relu(w.T @ z + b)
            self.zs.append(z)
       
        o = softmax(z)
        L = log_loss(o[y])
        return o, L

    def fprop_mat(self, x, y):
        self.zs = [x]
        z = x.T
        for w, b in zip(self.w, self.b):
            z = relu(w.T@z + b)
            self.zs.append(z)
       
        o = softmax(z)
        L = log_loss(o[y])
        return o, L
            
    def bprop(self, output, y):
        """
        Returns a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x
        """
        nabla_b = [np.zeros(b.shape) for b in self.b]
        nabla_w = [np.zeros(w.shape) for w in self.w]

        grad_output = output - self.onehot(y)
        e_prev = (grad_output * relu_prime(self.zs[-1]))
        nabla_w[-1] = relu(self.zs[-2]) @ e_prev.T

        for k in reversed(range(len(self.w[:-1]))):
            e_now = self.w[k+1] @ e_prev * relu_prime(self.zs[k+1])
            nabla_w[k] = self.zs[k] @ e_now.T
            nabla_b[k] = e_now
            e_prev = e_now

        return nabla_b, 

    def onehot(self, y):
        oh = np.zeros((self.layer_sizes[-1],1)).astype(int)
        oh[y-1] = 1
        return oh


if __name__ == '__main__':
    main()
