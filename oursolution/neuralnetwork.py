#!/usr/bin/env python

import os
# Set current working directory as this file's path
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import json
import random
import sys

import mnist_reader
import numpy as np


def init_weights(shape):
    return np.random.uniform(-1/np.sqrt(shape[0]), 1/np.sqrt(shape[0]), shape)

def softmax(x, axis=0):
    a = np.exp(x - np.amax(x, axis=axis, keepdims=True))
    return a / np.sum(a, axis=axis, keepdims=True)

def softmax_prime(a, y, axis=0):
    """Derivative of the softmax function."""
    return  

def relu(z):
    """
    The RELU function

    :param z: the pre-activation of the node with a relu activation
    """
    return np.maximum(z, 0)

def relu_prime(z):
    return np.greater(z, 0)

def log_loss(o_y):
    return -np.log(o_y)

def gradient_approx(x, func, eps=None):
    if eps is None or eps < 10 ** -6 or eps > 10 ** -4:
        eps = np.random.uniform(10 ** -6, 10 ** -4)
    return (func(x+eps) - func(x-eps)) / (2 * eps)

def main():
    # Circles.txt training data
    circles_data = np.loadtxt('data/circles/circles.txt')
    circles_train = circles_data[:,:-1]
    circles_target = circles_data[:,-1].reshape(1,-1).astype(int).T

    # MNIST training data
    x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    
    circles = Network([2,10,2])
    circles.finite_difference_gradient_check(circles_train[0,:].reshape(1,-1).T,
            circles_target[0,:].reshape(1,-1))

    # our_algorithm()
    #model = Network([784,300,10])
    #model.train(x_train

class Network(object):
    def __init__(
        self,
        num_neurons
    ):
        """
            Initialization method

        :param num_neurons: A list of the amount of neurons in each layer [input, hidden_1, ..., hidden_n, output]
        """
        self.num_layers = len(num_neurons)
        self.layer_sizes = num_neurons
        self.b = [np.zeros((layer_size, 1)) for layer_size in self.layer_sizes[1:]]
        self.w = [
            init_weights(size) for size in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]

    def finite_difference_gradient_check(
        self,
        test_data,
        test_target
    ):
        """
        Used to estimate the gradient numerically. Used to check our gradient computation

        :param test_data: An array of examples+labels we want to verify our back propagation algorithm with
        """
        x = test_data
        y = test_target
        output, loss = self.fprop(x, y)
        gradient = -1/output[y]
        finite_difference = gradient_approx(output[y], log_loss)
        gradient_2 = relu_prime(self.zs[-1])
        finite_diff_2 = gradient_approx(self.zs[-1], relu)
        assert 0.99 <= finite_difference/gradient <= 1.01
        for i in range(len(gradient_2)):
            assert 0.99 <= finite_diff_2[i]/gradient_2[i] <= 1.01

    def train(
        self,
        training_data,
        epochs,
        mini_batch_size,
        step_size
    ):
        """
        Train the Network

        :param training_data: An array of all training examples+labels we will use
        :param epochs: The amount of epochs we will use to train
        :param mini_batch_size: The size of our mini-batches
        :param step_size: The learning rate
        """
        print("Training ...")

        next_percentage = 0.1
        percentage_increment = 0.1
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, len(training_data), mini_batch_size)
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

        for x, y in mini_batch:
            o, L = self.fprop(x,y)
            delta_nabla_b, delta_nabla_w = self.bprop(o, y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.w = [w - (step_size / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.b = [b - (step_size / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def fprop(self, x, y):
        self.zs = [x]
        z = x
        for w, b in zip(self.w, self.b):
            z = relu(w.T @ z + b)
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
            nabla_w[k] = e_now @ self.zs[k].T
            nabla_b[k] = e_now
            e_prev = e_now

        return nabla_b, nabla_w

    def onehot(self, y):
        oh = np.zeros((self.layer_sizes[-1],1)).astype(int)
        oh[y-1] = 1
        return oh


if __name__ == '__main__':
    main()
