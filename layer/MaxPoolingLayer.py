import numpy as np


class MaxPoolingLayer:
    def __init__(self, downsampling_rate):
        self.downsampling_rate = downsampling_rate
        self.input = []
        self.output = []
        self.max_index = []

    def forward(self, input):
        self.input = input
        input = np.array(input)
        output = np.empty(len(input) / self.downsampling_rate)
        self.max_index = np.zeros(len(input) / self.downsampling_rate, dtype="int64")
        for i in range(len(self.max_index)):
            self.max_index[i] = np.argmax(input[
                                          i * self.downsampling_rate:i * self.downsampling_rate + self.downsampling_rate]) + i * self.downsampling_rate
            output[i] = input[self.max_index[i]]
        self.output = output.tolist()
        return self.output

    def backward(self, residual):
        error = np.zeros(len(self.input))
        for i in range(len(self.max_index)):
            error[self.max_index[i]] = residual[i]
        self.error = error.tolist()
        return self.error


if __name__ == '__main__':
    pl = MaxPoolingLayer(2)
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    r = [1, 2, 3, 4, 5]
    print(pl.forward(data))
    print (pl.max_index)
    print(pl.backward(r))
    data = [1, 2, 3, 4, 5, -6, 7, 8, 9, 0]
    print(pl.forward(data))
    print (pl.max_index)
    print(pl.backward(r))
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 110]
    print(pl.forward(data))
    print (pl.max_index)
    print(pl.backward(r))
    data = [1, 2, -3, -4, 5, 6, 7, 8, 9, 0]
    print(pl.forward(data))
    print (pl.max_index)

    print(pl.backward(r))
