import numpy as np


class MSELoss:
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return np.mean((predictions - targets) ** 2)

    def backward(self):
        return 2 * (self.predictions - self.targets) / self.targets.size
