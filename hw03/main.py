import os
import argparse
from sys import argv
from turtle import forward

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score



def load_data(path: str) -> tuple[np.ndarray]:
    # All images MUST have same dimension.
    # Train and Test MUST have same set of labels.
    i = 0
    labels = []
    for label in os.listdir(path):
        if os.path.isdir(os.path.join(path,label)):
            i += 1
            for filename in os.listdir(os.path.join(path, label)):
                labels.append(i)
                img = Image.open(os.path.join(path, label, filename))
                img = np.array(img)[:, :, 0].flatten()
                try:
                    images = np.vstack([images, img])
                except UnboundLocalError:
                    images = img
    return images, np.array(labels)


def pre_processing(x: np.ndarray, xt: np.ndarray) -> tuple[np.ndarray]:
    pca = PCA(2)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + np.finfo(np.float64).eps
    pca = PCA(2)
    x_pca = pca.fit_transform( (x - mean) / std )
    xt_pca = pca.transform((xt - mean) / std)
    return x_pca, xt_pca


# TODO
# 1. Softmax and it's impact.
# 2. Weight of bias
# 3. SGD
# 4. Initial value of weight
class TwoLayersClassifier:
    def __init__(self, n_h=6, lambda_=0, rate=1, atol=1e-5, max_iters=10000):
        self.n_h = n_h
        self.lambda_ = lambda_
        self.rate = rate
        self.atol = atol
        self.max_iters = max_iters
        self.n_f  = 0
        self.w1 = None
        self.w2 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.a3 = None

    # Sigmoid as Activation Function
    def sigma(self, x):
        return 1 / (1 + np.exp(-x))
    def d_sigma(self, x):
        return self.sigma(x) * (1 - self.sigma(x))

    def fit(self, x, labels):
        N, self.n_f = np.shape(x)
        y = np.zeros((N, np.max(labels)))
        for i in range(N):
            y[i, labels[i] - 1] = 1
        # Initial value of weight
        self.w1 = np.random.rand(self.n_h, self.n_f + 1) / self.n_f
        self.w2 = np.random.rand(np.max(labels), self.n_h + 1) / self.n_h
        for i in range(self.max_iters):
            self.forward(x)
            # cost = self.compute_cost(y)
            grad = self.backprop(y)
            self.w1 -= grad[0] * self.rate
            self.w2 -= grad[1] * self.rate
            if grad[0].max() < self.atol and grad[1].max() < self.atol:
                break

    def predict(self, xt):
        self.forward(xt)
        return self.a3.argmax(axis=1) + 1

    def forward(self, x):
        self.a1 = np.insert(x, 0, 1, axis=1)
        self.z2 = self.a1.dot(self.w1.T)
        self.a2 = np.insert(self.sigma(self.z2), 0, 1, axis=1)
        self.z3 = self.a2.dot(self.w2.T)
        self.a3 = self.sigma(self.z3)
    
    def compute_cost(self, y):
        N = y.shape[0]
        ones = np.ones_like(y)
        A = y * np.log(self.a3) + (ones - y) * np.log(ones - self.a3)
        J = -1 / N * A.trace()
        J += self.lambda_ / (2 * N) \
            * (np.sum(self.w1[:, 1:] ** 2) + np.sum(self.w2[:, 1:] ** 2))
        return J

    def backprop(self, y):
        delta3 = self.a3 - y
        delta2 = delta3.dot(self.w2[:, 1:]) * self.d_sigma(self.z2)
        w2_grad = self.a2.T.dot(delta3).T
        w1_grad = self.a1.T.dot(delta2).T
        w2_grad[:, 1:] += self.lambda_ * self.w2[:, 1:]
        w1_grad[:, 1:] += self.lambda_ * self.w1[:, 1:]
        w2_grad /= self.n_f
        w1_grad /= self.n_f
        return w1_grad, w2_grad


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--hidden_node', type=int, default=6)
    parser.add_argument('--train_path', type=str, default='Data_train')
    parser.add_argument('--test_path', type=str, default='Data_test')
    args = parser.parse_args()
    
    x,  y  = load_data(args.train_path)
    xt, yt = load_data(args.test_path)
    x,  xt = pre_processing(x, xt)

    clf = TwoLayersClassifier(args.hidden_node)
    clf.fit(x, y)
    yp = clf.predict(xt)
    print(accuracy_score(yt, yp))
    ConfusionMatrixDisplay.from_predictions(yt, yp)
    plt.show()
