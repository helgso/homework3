import os
# Set current working directory as this file's path
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import json
import random
import sys

import mnist_reader
import numpy as np


def softmax(x, axis=0):
    a = np.exp(x-np.expand_dims(np.max(x), axis=axis), axis)
    return a / np.expand_dims(np.sum(a, axis=axis), axis)

def main():
    # Circles.txt training data
    circles_data = np.loadtxt('data/circles/circles.txt')

    # MNIST training data
    x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # our_algorithm()


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
        self.biases = [np.zeros(layer_size) for layer_size in self.layer_sizes[1:]]
        self.weights = [
            [np.random.uniform(-1/np.sqrt(y), 1/np.sqrt(y), x) for i in range(0, y)]
            for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]

    def finite_difference_gradient_check(
        self,
        test_data
    ):
        """
        Used to estimate the gradient numerically. Used to check our gradient computation

        :param test_data: An array of examples+labels we want to verify our back propagation algorithm with
        """
        # 1: Calculate the loss function for the current parameter values (single example or mini batch)
        #loss_current_params = loss_function(test_data, current_params)

        # 2: Calculate the loss function for the current parameter values but add a small number ksi [10^-6; 10^-4] to each parameter first
        #ksi = np.random.uniform(10 ** -6, 10 ** -4)
        #loss_current_params_plus_ksi = loss_function(test_data, current_params + ksi)

        # 3: Divide the change of the loss function by ksi
        #finite_difference = np.absolute(loss_current_params - loss_current_params_plus_ksi)/ksi

        # 4: Validate that the ratio of our gradient computed by the back propagation and the finite difference is between 0.99 and 1.01
        #gradient_params = back_propagation(test_data)
        #loss_new_params = loss_function(test_data, gradient_params)
        #assert 0.99 <= finite_difference/loss_new_params <= 1.01

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
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.bprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (step_size / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (step_size / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def bprop(
        self,
        x,
        y
    ):
        """
        Returns a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x
        """
        # TODO: Replace the hidden neurons' activation with relu and the output neurons' activation with softmax
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = CostFunction.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return nabla_b, nabla_w


def relu(z):
    """
    The RELU function

    :param z: the pre-activation of the node with a relu activation
    """
    if z >= 0:
        return z

    return 0


class CostFunction(object):
    @staticmethod
    def fn(a, y):
        """
        Returns the cost associated with an output ` and desired output y
        """
        # TODO: Implement
        pass

    @staticmethod
    def delta(z, a, y):
        """
        Returns the error delta from the output layer.
        """
        # TODO: Implement
        pass


if __name__ == '__main__':
    main()
