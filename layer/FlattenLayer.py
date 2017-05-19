class FlattenLayer:
    def __init__(self):
        pass

    def forward(self, in_data):
        self.in_batch, self.in_channel, self.r, self.c = in_data.shape
        return in_data.reshape(self.in_batch, self.in_channel * self.r * self.c)

    def backward(self, residual):
        return residual.reshape(self.in_batch, self.in_channel, self.r, self.c)
