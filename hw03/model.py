import numpy as np
import matplotlib.pyplot as plt


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



class NNClassifier:
    def __init__(self, n_o, n_f, *, n_l=3, n_h=6, 
                rate=0.01, epochs=50, batch_size=3000):
        self.n_l = n_l
        self.rate = rate
        self.epochs = epochs
        self.batch_size = batch_size
        # Initial value of weight
        theta = np.random.rand(n_h, n_h + 1) * np.sqrt(1/n_h)
        self.w = [theta for _ in range(n_l)]
        self.w[0] = np.random.rand(n_h, n_f + 1) * np.sqrt(1/n_h)
        self.w[n_l-1] = np.random.rand(n_o, n_h + 1) * np.sqrt(1/n_o)
        self.z = [0 for _ in range(n_l+1)]
        self.a = [0 for _ in range(n_l+1)]
    
    def cross_entropy(self, yt, yp):
        return -np.sum(yt * np.log(yp)) / yt.shape[0]

    def make_one_hot(self, labels):
        y = np.zeros((len(labels), np.max(labels)))
        for i in range(len(labels)):
            y[i, labels[i] - 1] = 1
        return y

    def fit(self, x, labels, x_val=None, label_val=None):
        y = self.make_one_hot(labels)
        yp = np.ones_like(y)
        if x_val is not None:
            y_val = self.make_one_hot(label_val)
            yp_val = np.ones_like(y_val)
        training_loss = np.zeros(self.epochs)
        valid_loss = np.zeros(self.epochs)
        for k in range(self.epochs):
            for j in range(self.batch_size):
                rand = np.random.randint(0, x.shape[0])
                self.forward(x[rand])
                grad = self.backprop(y[rand])
                for i in range(0, self.n_l):
                    self.w[i] -= grad[i] * self.rate
                yp[rand] = self.z[self.n_l]
            training_loss[k] = self.cross_entropy(y, yp)
            if x_val is not None:
                for j in range(len(x_val)):
                    self.forward(x_val[j])
                    yp_val[j] = self.z[self.n_l]
                valid_loss[k] = self.cross_entropy(y_val, yp_val)
        if x_val is not None:
            return training_loss, valid_loss
        return training_loss

    def predict(self, xt):
        yp = np.zeros(xt.shape[0])
        for i in range(xt.shape[0]):
            self.forward(xt[i])
            yp[i] = self.a[self.n_l].argmax() + 1
        return yp

    def forward(self, x):
        self.a[0] = np.insert(x, 0, 1)
        for i in range(1, self.n_l + 1):
            self.z[i] = self.w[i-1] @ self.a[i-1]
            self.a[i] = np.insert(sigmoid(self.z[i]), 0, 1)
        self.a[self.n_l] = softmax(self.z[self.n_l])

    def backprop(self, y):
        dw = [0 for _ in range(self.n_l)]
        delta = self.a[self.n_l] - y
        for i in range(self.n_l-1, 0, -1):
            dw[i] = np.outer(delta, self.a[i])
            delta = self.w[i][:, 1:].T @ delta * sigmoid(self.z[i], True)
        dw[0] = np.outer(delta, self.a[0])
        return dw


class NNClassifier_batch:
    def __init__(self, n_o, n_f, *, n_l=3, n_h=6, rate=0.1, max_iters=10000):
        self.n_l = n_l
        self.rate = rate
        self.max_iters = max_iters
        # Initial value of weight
        theta = np.random.rand(n_h, n_h + 1) * np.sqrt(1/n_h)
        self.w = [theta for _ in range(n_l)]
        self.w[0] = np.random.rand(n_h, n_f + 1) * np.sqrt(1/n_h)
        self.w[n_l-1] = np.random.rand(n_o, n_h + 1) * np.sqrt(1/n_o)
        self.z = [0 for _ in range(n_l+1)]
        self.a = [0 for _ in range(n_l+1)]
    
    def cross_entropy(self, y):
        return -np.sum(y * np.log(self.a[self.n_l])) / y.shape[0]

    def fit(self, x, labels):
        # Make one-hot label.
        y = np.zeros((len(labels), np.max(labels)))
        for i in range(len(labels)):
            y[i, labels[i] - 1] = 1

        training_loss = []
        self.a[0] = np.insert(x, 0, 1, axis=1)
        for i in range(self.max_iters):
            self.forward()
            grad = self.backprop(y)
            for i in range(0, self.n_l):
                self.w[i] -= grad[i] * self.rate
            training_loss.append(self.cross_entropy(y))
        plt.plot(np.array(training_loss))

    def predict(self, xt):
        self.a[0] = np.insert(xt, 0, 1, axis=1)
        self.forward()
        return self.a[self.n_l].argmax(axis=1) + 1

    def forward(self):
        for i in range(1, self.n_l + 1):
            self.z[i] = self.a[i-1] @ self.w[i-1].T
            self.a[i] = np.insert(sigmoid(self.z[i]), 0, 1, axis=1)
        self.a[self.n_l] = softmax(self.z[self.n_l])

    def backprop(self, y):
        dw = [0 for _ in range(self.n_l)]
        delta = self.a[self.n_l] - y
        for i in range(self.n_l-1, 0, -1):
            dw[i] = delta.T @ self.a[i] / y.shape[0]
            delta = delta @ self.w[i][:, 1:] * sigmoid(self.z[i], True)
        dw[0] = delta.T @ self.a[0] / y.shape[0]
        return dw
