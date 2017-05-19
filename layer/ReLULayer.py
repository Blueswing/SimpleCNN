class ReLULayer:
    def __init__(self):
        self.name = "ReLU"

    def forward(self, input):
        self.input = input
        self.output = list(input)
        self.output = [x if x > 0 else 0.01 * x for x in self.output]
        return self.output

    def backward(self, residual):
        self.error = list(residual)
        for i in range(len(self.error)):
            if self.output[i] < 0:
                self.error[i] *= 0.01
        return self.error


if __name__ == '__main__':
    rl = ReLULayer()
    in_put = [-1, -100, 1, 2, 3, 4, 5, 6, 7]
    print rl.forward(in_put)
    print rl.backward([1, 2, 3, 4, 5, 6, 7, 8, 9])
