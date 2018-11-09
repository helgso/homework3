#!/usr/bin/env python

import neuralnetwork as nn

import mnist_reader
import numpy as np

circles_data = np.loadtxt('data/circles/circles.txt')
circles_x = circles_data[:,:-1]
circles_y = nn.vectorof(circles_data[:,-1].astype(int))

def main():
    question1()
    question2()
    question3()
    question4()
    question5()
    question6()
    question7()
    question8()
    question9()
    question10()

def question1():
    pass

def question2():

   model = Network([2,2])
   model.finite_difference_gradient_check(
           nn.vectorof(circles_x[0,:]),
           nn.vectorof(circles_y[0,:]),
           display=True)

def question3():
    pass
    # Done

def question4():
    pass

def question5():
    pass

def question6():
    pass 

def question7():
    pass

def question8():
    pass

def question9():
    pass

def question10():
    pass
