import numpy as np
import pandas as pd

class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.coeff = None
        self.features = None
        self.target = None
        self.mse_history = []

    def set_coefficients(self, *args):
        self.coeff = np.array(args).reshape(-1, 1)

    def cost(self):
        predicted = self.features.dot(self.coeff)
        s = ((predicted - self.target) ** 2).sum()
        return (0.5 / len(self.features)) * s

    def predict(self, features: pd.DataFrame):
        features = features.copy()
        features.insert(0, 'c0', np.ones((len(features), 1)))
        features = features.to_numpy()
        return features.dot(self.coeff).reshape(-1, 1).flatten()

    def gradient_descent_step(self):
        predicted = self.features.dot(self.coeff)
        s = self.features.T.dot(predicted - self.target)
        gradient = (1.0 / len(self.features)) * s
        self.coeff = self.coeff - self.learning_rate * gradient
        return self.coeff, self.cost()

    def perform_gradient_descent(self):
        self.mse_history = []
        for _ in range(self.n_iter):
            _, curr_cost = self.gradient_descent_step()
            self.mse_history.append(curr_cost)
        return self.coeff, self.mse_history

    def fit(self, features: pd.DataFrame, target: pd.Series):
        self.features = features.copy()
        coeff_shape = len(features.columns) + 1
        self.coeff = np.zeros(shape=(coeff_shape, 1))
        self.features.insert(0, 'c0', np.ones((len(features), 1)))
        self.features = self.features.to_numpy()
        self.target = target.to_numpy().reshape(-1, 1)

        return self.perform_gradient_descent()
