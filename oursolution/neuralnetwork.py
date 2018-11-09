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

def softmax_prime(output_activations, y):
    """
    Derivative of the softmax function

    :param delta: The change to the output activations we want
    :param y: The output label
    :return: dL/do^a
    """
    return output_activations - onehot(y, len(output_activations))


def onehot(y, m):
    oh = np.zeros(m)
    oh[y] = 1
    return oh


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


def gradient_approx(x, eps=None):
    if eps is None or eps < 10 ** -6 or eps > 10 ** -4:
        eps = np.random.uniform(10 ** -6, 10 ** -4)
    return (log_loss(x+eps) - log_loss(x-eps)) / (2 * eps)


def main():
    # Circles.txt training data
    circles_raw = np.loadtxt('data/circles/circles.txt')
    circles_training_data = [[x, int(y)] for x, y in zip(circles_raw[:, :-1], circles_raw[:, -1])]

    helgi_circles_test = [circles_training_data[0]]

    # MNIST training data
    # x_mnist_raw_train, y_mnist_raw_train = mnist_reader.load_mnist('data/fashion', kind='train')
    # x_mnist_raw_test, y_mnist_raw_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    # mnist_training_set = [[x, y] for x, y in (x_mnist_raw_train, y_mnist_raw_train)]
    # mnist_test_set = [[x, y] for x, y in (x_mnist_raw_test, y_mnist_raw_test)]

    circles_network = Network([2, 2, 2])
    circles_network.train(
        training_data=circles_training_data,
        epochs=1,
        mini_batch_size=10,
        step_size=1
    )


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
        self.b = [np.zeros(layer_size) for layer_size in self.layer_sizes[1:]]
        self.w = [init_weights(size) for size in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

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
        gradient_b = [np.zeros(b.shape) for b in self.b]
        gradient_w = [np.zeros(w.shape) for w in self.w]

        for x, y in mini_batch:
            delta_gradient_b, delta_gradient_w = self.bprop(x, y)
            gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
            gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]

        self.w = [w - (step_size / len(mini_batch)) * nw for w, nw in zip(self.w, gradient_w)]
        self.b = [b - (step_size / len(mini_batch)) * nb for b, nb in zip(self.b, gradient_b)]

    def fprop(self, x, y):
        self.zs = []
        z = x
        for w, b, a in zip(self.w, self.b):
            z = relu(np.dot(w.T, z) + b)
            self.zs.append(z)
       
        o = softmax(a)
        L = log_loss(o[y])
        return o, L

    def bprop(
        self,
        output,
        y
    ):
        """
        Returns gradient_b, gradient_w representing the gradients for the cost function
        """
        gradient_b = [np.zeros(b.shape) for b in self.b]
        gradient_w = [np.zeros(w.shape) for w in self.w]

        # forward pass
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        iterations = self.num_layers-1
        for i in range(iterations):
            z = np.dot(self.w[i], activation) + self.b[i]
            zs.append(z)

            if i < iterations-1:
                # Every intermediate layer uses a relu activation
                activation = relu(z)
            else:
                # The output layer uses a softmax activation
                activation = softmax(z)

            activations.append(activation)

        # backward pass
        dL_do_a = softmax_prime(activations[-1], y)

        last_hidden_size = self.layer_sizes[-2]
        output_size = self.layer_sizes[-1]

        gradient_b[-1] = dL_do_a
        gradient_w[-1] = np.array([
            [dL_do_a[p] * activations[-2][k] for p in range(0, output_size)] for k in range(0, last_hidden_size)
        ])

        for l in range(2, self.num_layers):
            dodL_dhdo = np.dot(self.w[-l+1].transpose(), dL_do_a)

            gradient_b[-l] = dodL_dhdo
            gradient_w[-l] = np.array([
                [activations[-l-1][j] * sum([self.w[-l+1][k] * dL_do_a[p] for p in range(0, output_size)]) for k in range(0, last_hidden_size)] for j in range(self.layer_sizes[-l-1])
            ])

        return gradient_b, gradient_w


if __name__ == '__main__':
    main()
