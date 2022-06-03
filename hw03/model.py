import numpy as np


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


def cross_entropy(yt, yp):
    return -np.sum(yt * np.log(yp)) / yt.shape[0]


def make_one_hot(labels):
    y = np.zeros((len(labels), np.max(labels)))
    for i in range(len(labels)):
        y[i, labels[i] - 1] = 1
    return y


class NNClassifier:
    def __init__(self, n_o, n_f, *, n_l=3, n_h=6,
                 rate=0.01, epochs=30, batch_size=1000, method='SGD'):
        self.n_l = n_l
        self.rate = rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.method = method
        # Initial value of weight
        theta = np.random.rand(n_h, n_h + 1) * np.sqrt(1/n_h)
        self.w = [theta for _ in range(n_l)]
        self.w[0] = np.random.rand(n_h, n_f + 1) * np.sqrt(1/n_h)
        self.w[n_l-1] = np.random.rand(n_o, n_h + 1) * np.sqrt(1/n_o)
        self.best = self.w
        self.act = [0 for _ in range(n_l+1)]
        self.unit = [0 for _ in range(n_l+1)]
        print(f'Hidden Layers: {n_l}\nHidden Units: {n_h}')
        print(f'Learning Rate: {rate}\nBatch Size: {batch_size}')

    def fit(self, x, labels, x_val=None, label_val=None):
        y = make_one_hot(labels)
        y_val = make_one_hot(label_val)
        training_loss = np.zeros(self.epochs)
        valid_loss = np.zeros(self.epochs)
        min_valid_loss = np.finfo(np.float64).max
        for k in range(self.epochs):
            if self.method == 'SGD':
                training_loss[k] = self.__SGD(x, y)
                if x_val is None:
                    continue
                valid_loss[k] = self.__SGD(x_val, y_val, valid=True)
            elif self.method == 'BGD':
                training_loss[k] = self.__BGD(x, y)
                if x_val is None:
                    continue
                valid_loss[k] = self.__BGD(x_val, y_val, valid=True)
            if min_valid_loss > valid_loss[k]:
                print('Update best model to epoch =', k, end='\r')
                min_valid_loss = valid_loss[k]
                self.best = self.w
        if x_val is None:
            self.best = self.w
            return training_loss
        print()
        return training_loss, valid_loss

    def predict(self, xt):
        self.w = self.best
        if self.method == 'SGD':
            yp = np.zeros(xt.shape[0])
            for i in range(xt.shape[0]):
                self.__forward(xt[i])
                yp[i] = self.unit[self.n_l].argmax() + 1
        elif self.method == 'BGD':
            self.__forward(xt)
            yp = self.unit[self.n_l].argmax() + 1
        return yp
    
    def __SGD(self, x, y, valid=False):
        yp = np.ones_like(y)
        if valid:
            for j in range(len(x)):
                self.__forward(x[j])
                yp[j] = self.unit[self.n_l]
            return cross_entropy(y, yp)
        for j in range(self.batch_size):
            rand = np.random.randint(0, x.shape[0])
            self.__forward(x[rand])
            grad = self.__backprop(y[rand])
            for i in range(0, self.n_l):
                self.w[i] -= grad[i] * self.rate
            yp[rand] = self.unit[self.n_l]
        return cross_entropy(y, yp)

    def __BGD(self, x, y, valid=False):
        self.__forward(x)
        if not valid:
            grad = self.__backprop(y)
            for i in range(0, self.n_l):
                self.w[i] -= grad[i] * self.rate
        return cross_entropy(y, self.unit[self.n_l])

    def __forward(self, x):
        self.unit[0] = np.insert(x, 0, 1)
        if np.ndim(x) == 1:
            for i in range(1, self.n_l + 1):
                self.act[i] = self.w[i-1] @ self.unit[i-1]
                self.unit[i] = np.insert(sigmoid(self.act[i]), 0, 1)
        else:
            for i in range(1, self.n_l + 1):
                self.act[i] = self.unit[i-1] @ self.w[i-1].T
                self.unit[i] = np.insert(sigmoid(self.act[i]), 0, 1, axis=1)
        self.unit[self.n_l] = softmax(self.act[self.n_l])

    def __backprop(self, y):
        dw = [0 for _ in range(self.n_l)]
        delta = self.unit[self.n_l] - y
        if np.ndim(y) == 1:
            for i in range(self.n_l-1, -1, -1):
                dw[i] = np.outer(delta, self.unit[i])
                delta = self.w[i][:, 1:].T @ delta * sigmoid(self.act[i], True)
        else:
            for i in range(self.n_l-1, -1, -1):
                dw[i] = delta.T @ self.unit[i] / y.shape[0]
                delta = delta @ self.w[i][:, 1:] * sigmoid(self.act[i], True)
        return dw


class NNClassifierBatch:
    def __init__(self, n_o, n_f, *, n_l=3, n_h=6, rate=0.1, epochs=10000):
        self.n_l = n_l
        self.rate = rate
        self.epochs = epochs
        # Initial value of weight
        theta = np.random.rand(n_h, n_h + 1) * np.sqrt(1/n_h)
        self.w = [theta for _ in range(n_l)]
        self.w[0] = np.random.rand(n_h, n_f + 1) * np.sqrt(1/n_h)
        self.w[n_l-1] = np.random.rand(n_o, n_h + 1) * np.sqrt(1/n_o)
        self.act = [0 for _ in range(n_l+1)]
        self.unit = [0 for _ in range(n_l+1)]

    def fit(self, x, labels):
        y = make_one_hot(labels)
        training_loss = np.zeros(self.epochs)
        self.unit[0] = np.insert(x, 0, 1, axis=1)
        for j in range(self.epochs):
            self.forward()
            grad = self.backprop(y)
            for i in range(0, self.n_l):
                self.w[i] -= grad[i] * self.rate
            training_loss[j] = cross_entropy(y, self.unit[self.n_l])
        return training_loss

    def predict(self, xt):
        self.unit[0] = np.insert(xt, 0, 1, axis=1)
        self.forward()
        return self.unit[self.n_l].argmax(axis=1) + 1

    def forward(self):
        for i in range(1, self.n_l + 1):
            self.act[i] = self.unit[i-1] @ self.w[i-1].T
            self.unit[i] = np.insert(sigmoid(self.act[i]), 0, 1, axis=1)
        self.unit[self.n_l] = softmax(self.act[self.n_l])

    def backprop(self, y):
        dw = [0 for _ in range(self.n_l)]
        delta = self.unit[self.n_l] - y
        for i in range(self.n_l-1, -1, -1):
            dw[i] = delta.T @ self.unit[i] / y.shape[0]
            delta = delta @ self.w[i][:, 1:] * sigmoid(self.act[i], True)
        return dw
