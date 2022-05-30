import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        exps = np.exp(x - x.max(axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    exps = np.exp(x - x.max())
    return exps / np.sum(exps)


class TwoLayersClassifier:
    def __init__(self, n_h=6, *, rate=0.01, epochs=20):
        self.n_h = n_h
        self.rate = rate
        self.epochs = epochs

    def fit(self, x, labels):
        N, M = np.shape(x)
        n_out = np.max(labels)
        # Make one-hot label.
        y = np.zeros((N, n_out))
        for i in range(N):
            y[i, labels[i] - 1] = 1
        # Initial value of weight
        self.w1 = np.random.rand(self.n_h, M + 1) * np.sqrt(1/self.n_h)
        self.w2 = np.random.rand(n_out, self.n_h + 1) * np.sqrt(1/n_out)
        # SGD
        yp = np.zeros(N)
        mse = np.zeros(self.epochs)
        for i in range(self.epochs):
            for j in range(N):
                rand = np.random.randint(0, N)
                self.forward(x[rand])
                grad = self.backprop(y[rand])
                self.w1 -= grad[0] * self.rate
                self.w2 -= grad[1] * self.rate
                yp[rand] = self.a3.argmax() + 1
            mse[i] = mean_squared_error(labels, yp)
        plt.plot(mse)

    def predict(self, xt):
        yp = np.zeros(xt.shape[0], np.int64)
        for i in range(xt.shape[0]):
            self.forward(xt[i])
            yp[i] = self.a3.argmax() + 1
        return yp

    def forward(self, x):
        self.a1 = np.insert(x, 0, 1)
        self.z2 = self.w1 @ self.a1
        self.a2 = np.insert(sigmoid(self.z2), 0, 1)
        self.z3 = self.w2 @ self.a2
        self.a3 = softmax(self.z3)
    
    def backprop(self, y):
        delta3 = self.a3 - y
        delta2 = self.w2[:, 1:].T @ delta3 * sigmoid(self.z2, derivative=True)
        w2_grad = np.outer(delta3, self.a2)
        w1_grad = np.outer(delta2, self.a1)
        return w1_grad, w2_grad


class TwoLayersClassifier_batch:
    def __init__(self, n_h=6, *, rate=0.1, max_iters=10000):
        self.n_h = n_h
        self.rate = rate
        self.max_iters = max_iters

    def fit(self, x, labels):
        N, M = np.shape(x)
        n_out = np.max(labels)
        # Make one-hot label.
        y = np.zeros((N, n_out))
        for i in range(N):
            y[i, labels[i] - 1] = 1
        # Initial value of weight
        self.w1 = np.random.rand(self.n_h, M + 1) * np.sqrt(1/self.n_h)
        self.w2 = np.random.rand(n_out, self.n_h + 1) * np.sqrt(1/n_out)
        # Batch
        training_loss = []
        for i in range(self.max_iters):
            self.forward(x)
            grad = self.backprop(y)
            self.w1 -= grad[0] * self.rate
            self.w2 -= grad[1] * self.rate
            training_loss.append(self.cross_entropy(y))
        plt.plot(np.array(training_loss))

    def predict(self, xt):
        self.forward(xt)
        return self.a3.argmax(axis=1) + 1

    def forward(self, x):
        self.a1 = np.insert(x, 0, 1, axis=1)
        self.z2 = self.a1 @ self.w1.T
        self.a2 = np.insert(sigmoid(self.z2), 0, 1, axis=1)
        self.z3 = self.a2 @ self.w2.T
        self.a3 = softmax(self.z3)
    
    def cross_entropy(self, y):
        return -np.sum(y * np.log(self.a3)) / y.shape[0]

    def backprop(self, y):
        delta3 = self.a3 - y
        delta2 = delta3 @ self.w2[:, 1:] * sigmoid(self.z2, derivative=True)
        w2_grad = delta3.T @ self.a2
        w1_grad = delta2.T @ self.a1
        w2_grad /= y.shape[0]
        w1_grad /= y.shape[0]
        return w1_grad, w2_grad

