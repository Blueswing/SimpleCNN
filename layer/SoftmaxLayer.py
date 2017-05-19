import numpy as np


class SoftmaxLayer:
    def __init__(self):
        pass

    def forward(self, in_data):
        self.input = in_data
        exp_out = np.exp(in_data)
        sum_out = np.sum(exp_out)
        self.output = [x / sum_out for x in exp_out]
        return self.output

    def backward(self, target):
        return np.subtract(self.output, target)
