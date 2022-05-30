import os
import argparse

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
# 1. Confirm softmax and it's impact.
# 2. SGD
class TwoLayersClassifier:
    def __init__(self, n_h=6, *, rate=1, atol=1e-5, max_iters=10000):
        self.n_h = n_h
        self.rate = rate
        self.atol = atol
        self.max_iters = max_iters

    # Sigmoid as Activation Function
    def sigma(self, x):
        return 1 / (1 + np.exp(-x))
    def d_sigma(self, x):
        return self.sigma(x) * (1 - self.sigma(x))
    
    def softmax(self, x):
        exps = np.exp(x - x.max(axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

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
        for i in range(self.max_iters):
            self.forward(x)
            # print(self.cross_entropy_loss(y))
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
        self.z2 = self.a1 @ self.w1.T
        self.a2 = np.insert(self.sigma(self.z2), 0, 1, axis=1)
        self.z3 = self.a2 @ self.w2.T
        self.a3 = self.softmax(self.z3)
    
    def cross_entropy_loss(self, y):
        return -np.sum(y * np.log(self.a3)) / y.shape[0]

    def backprop(self, y):
        delta3 = self.a3 - y
        delta2 = delta3 @ self.w2[:, 1:] * self.d_sigma(self.z2)
        w2_grad = delta3.T @ self.a2
        w1_grad = delta2.T @ self.a1
        w2_grad /= y.shape[0]
        w1_grad /= y.shape[0]
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
