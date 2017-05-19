import random

import numpy as np

from util.Util import *
from ReLULayer import ReLULayer
from MaxPoolingLayer import MaxPoolingLayer
from SoftmaxLayer import SoftmaxLayer
from FCLayer import FCLayer


class ConvLayer:
    def __init__(self, input_size, output_size, kernel_size, lr=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.input = []
        self.output = []
        self.kernel = np.random.rand(kernel_size)
        self.delta_kernel = []
        self.b = random.random()
        self.lr = lr
        self.error = []
        # self.momentum = momentum

    def forward(self, in_data):
        self.input = in_data
        self.output = np.convolve(in_data, self.kernel, mode="same").tolist()
        return self.output

    def backward(self, residual):
        tmp_r = list(residual)
        tmp_r.reverse()
        tmp_in = list(self.input)
        tmp_in.reverse()
        self.error = np.convolve(residual, tmp_r, mode="same")
        self.delta_kernel = np.convolve(residual[1:len(residual) - 1], tmp_in, mode='valid')  # flip input
        for i in range(len(self.kernel)):
            self.kernel[i] -= self.lr * self.delta_kernel[i]
        return self.error


if __name__ == '__main__':

    cl = ConvLayer(10, 10, 3)
    rl = ReLULayer()
    ml = MaxPoolingLayer(2)
    fc1 = FCLayer(5, 5)
    rl1 = ReLULayer()
    fc2 = FCLayer(5, 5)
    sl = SoftmaxLayer()

    layers = []
    layers.append(cl)
    layers.append(rl)
    layers.append(ml)
    layers.append(fc1)
    layers.append(rl1)
    layers.append(fc2)
    layers.append(sl)
    data = [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
    target = [[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1],
              ]
    r = []

    for c in range(120000):
        input = data[c % len(data)]
        for l in range(len(layers)):
            input = layers[l].forward(input)
        r = target[c % len(target)]
        for l in range(len(layers) - 1, -1, -1):
            r = layers[l].backward(r)
            # output = cl.forward(data)
            # output = rl.forward(output)
            # output = ml.forward(output)  # BUG! no Softmax
            # sl.forward(output)
            # print cl.kernel
            # print cl.output
            # r = [0, 0, 0, 0, 0]
            # for i in range(len(r)):
            #    r[i] = ml.output[i] - target[i]
            #
            # r=sl.backward(target)
            # r = ml.backward(r)
            # r = rl.backward(r)
            # cl.backward(r)
            # print residual
            # print cl.backward(residual)

    print(cl.kernel)
    print(cl.delta_kernel)
    print(cl.output)
    print(layers[-1].output)
    print(r)
    print(cl.error)

    input = data[0]
    for l in range(len(layers)):
        input = layers[l].forward(input)
    print(layers[-1].output)

    input = data[1]
    for l in range(len(layers)):
        input = layers[l].forward(input)
    print(layers[-1].output)

    input = data[2]
    for l in range(len(layers)):
        input = layers[l].forward(input)
    print(layers[-1].output)

    input = data[3]
    for l in range(len(layers)):
        input = layers[l].forward(input)
    print(layers[-1].output)

    input = data[4]
    for l in range(len(layers)):
        input = layers[l].forward(input)
    print(layers[-1].output)

    input = [1, 0.5, 0.2, 0.1, 0, 0, 0, 0, 0, 0]
    for l in range(len(layers)):
        input = layers[l].forward(input)
    print(layers[-1].output)
