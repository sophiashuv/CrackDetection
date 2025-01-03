import numpy as np

class BCELoss:
    def forward(self, predictions, targets):
        self.predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        self.targets = targets
        return -np.mean(targets * np.log(self.predictions) + (1 - targets) * np.log(1 - self.predictions))

    def backward(self):
        return -(self.targets / self.predictions - (1 - self.targets) / (1 - self.predictions)) / self.targets.size
