import numpy as np
from SoftmaxLayer import SoftmaxLayer
from ReLULayer import ReLULayer


class FCLayer:
    def __init__(self, input_size, output_size, lr=0.01, momentum=0.9):
        self.input_size = input_size
        self.output_size = output_size
        # self.w = np.random.randn(in_num, out_num)
        self.w = np.random.rand(output_size, input_size)
        self.b = np.random.rand(output_size)
        self.lr = lr
        self.momentum = momentum

    def forward(self, in_data):
        self.output = np.dot(self.w, in_data) + self.b
        self.input = in_data
        return self.output

    def backward(self, residual):
        # batch_size = loss.shape[0]
        error = np.zeros(self.input_size)
        for i in range(len(error)):
            error[i] = np.dot(self.w[:, i], residual)
        self.error = error.tolist()
        self.delta_b = residual
        rr = np.transpose([residual])
        self.delta_w = np.dot(rr, [self.input])
        self.delta_w = np.multiply(self.delta_w, -self.lr)
        self.w = np.add(self.w, self.delta_w)
        self.delta_b = np.multiply(self.delta_b, -self.lr)
        self.b = np.add(self.b, self.delta_b)

        return self.error


if __name__ == '__main__':
    nn = []
    data = [[1, 1], [1, 2], [2, 1], [2, 2]]
    target = [[0, 1], [0, 1], [0, 1], [1, 0]]
    nn.append(FCLayer(2, 2))
    nn.append(ReLULayer())
    nn.append(FCLayer(2, 2))
    nn.append(ReLULayer())
    nn.append(FCLayer(2, 2))
    nn.append(SoftmaxLayer())

    # input = data
    # for l in range(len(nn)):
    #     input = nn[l].forward(input)
    #     print(input)



    for i in range(50000):
        input = data[i % len(data)]
        for l in range(len(nn)):
            input = nn[l].forward(input)

        r = target[i % len(data)]
        for l in range(len(nn) - 1, -1, -1):
            r = nn[l].backward(r)
    print(nn[-1].output)

    input = [1, 1]
    for l in range(len(nn)):
        input = nn[l].forward(input)
    print(nn[-1].output)

    input = [1, 2]
    for l in range(len(nn)):
        input = nn[l].forward(input)
    print(nn[-1].output)

    input = [2, 1]
    for l in range(len(nn)):
        input = nn[l].forward(input)
    print(nn[-1].output)

    input = [2, 2]
    for l in range(len(nn)):
        input = nn[l].forward(input)
    print(nn[-1].output)

    input = [1.7, 1.7]
    for l in range(len(nn)):
        input = nn[l].forward(input)
    print(nn[-1].output)
