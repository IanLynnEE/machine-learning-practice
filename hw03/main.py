import os
import argparse

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

from model import NNClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hidden_units', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--train_path', type=str, default='Data_train')
    parser.add_argument('--test_path', type=str, default='Data_test')
    args = parser.parse_args()

    labels = ['Carambula', 'Lychee', 'Pear']
    x, y = load_data(args.train_path)
    xt, yt = load_data(args.test_path)
    x, xt = reduce_dimension(x, xt)
    x, xv, y, yv = train_test_split(x, y, test_size=0.2)

    clf = NNClassifier(3, 2, n_l=args.layers, n_h=args.hidden_units,
                       epochs=args.epochs, batch_size=args.batch_size)
    training_loss, valid_loss = clf.fit(x, y, xv, yv)

    plt.plot(training_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.title('Cross Entropy')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('training_loss.png', dpi=300)

    yp = clf.predict(xt)
    print('ACC =', accuracy_score(yt, yp))
    ConfusionMatrixDisplay.from_predictions(yt, yp, display_labels=labels)
    plt.savefig('confusion_matrix.png', dpi=300)

    plot_decision_boundary(clf, xt, yt)
    return


def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    # All images MUST have same dimension.
    # Train and Test MUST have same set of labels.
    i = 0
    labels = []
    for label in os.listdir(path):
        if os.path.isdir(os.path.join(path, label)):
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


def train_test_split(x: np.ndarray, y: np.ndarray, test_size: float):
    x_train = np.empty((0, *x.shape[1:]))
    x_test = np.empty((0, *x.shape[1:]))
    y_train = np.empty((0, ), y.dtype)
    y_test = np.empty((0, ), y.dtype)
    for i in range(1, y.max() + 1):
        select_idx = np.where(y[:] == i)[0]
        xx, yy = x[select_idx, ...], y[select_idx, ...]
        end_idx = -int(-xx.shape[0] // (1 / test_size + np.finfo(float).eps))
        indices = np.random.permutation(xx.shape[0])
        test_idx, train_idx = indices[:end_idx], indices[end_idx:]
        x_train = np.vstack((x_train, xx[train_idx, ...]))
        y_train = np.hstack((y_train, yy[train_idx, ...]))
        x_test = np.vstack((x_test, xx[test_idx, ...]))
        y_test = np.hstack((y_test, yy[test_idx, ...]))
    return x_train, x_test, y_train, y_test


def reduce_dimension(x: np.ndarray, xt: np.ndarray, xv=np.empty()):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + np.finfo(np.float64).eps
    pca = PCA(2)
    if xv is None:
        return pca.fit_transform((x-mean)/std), pca.transform((xt-mean)/std)
    return (
        pca.fit_transform((x-mean) / std),
        pca.transform((xt-mean) / std),
        pca.transform((xv-mean) / std)
    )


def plot_decision_boundary(clf, x, y):
    grid0 = np.arange(x[:, 0].min()-1, x[:, 0].max()+1, 0.1)
    grid1 = np.arange(x[:, 1].min()-1, x[:, 1].max()+1, 0.1)
    xx, yy = np.meshgrid(grid0, grid1)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1, r2))
    yp = clf.predict(grid)
    zz = yp.reshape(xx.shape)
    plt.clf()
    plt.contourf(xx, yy, zz, cmap='rocket_r', alpha=0.3)

    df = pd.DataFrame(x, columns=['PCA 0', 'PCA 1'])
    df['label'] = y
    sns.scatterplot(data=df, x='PCA 0', y='PCA 1', hue='label')
    plt.title('Test Dataset')
    plt.savefig('decision_boundary.png', dpi=300)
    plt.clf()
    return


if __name__ == '__main__':
    main()
