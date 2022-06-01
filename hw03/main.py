import os
import argparse

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

from model import NNClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--hidden_node', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--train_path', type=str, default='Data_train')
    parser.add_argument('--test_path', type=str, default='Data_test')
    args = parser.parse_args()

    labels = ['Carambula', 'Lychee', 'Pear']
    x, y = load_data(args.train_path)
    xt, yt = load_data(args.test_path)
    x, xv, y, yv = train_test_split(x, y, test_size=0.2)
    x,  xv, xt = pre_processing(x, xv, xt)

    clf = NNClassifier(3, 2, n_l=args.layer, n_h=args.hidden_node,
                       epochs=args.epochs, batch_size=args.batch_size)
    training_loss, valid_loss = clf.fit(x, y, xv, yv)

    plt.plot(training_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.title('Cross Entropy')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    yp = clf.predict(xt)
    print('ACC =', accuracy_score(yt, yp))
    ConfusionMatrixDisplay.from_predictions(yt, yp, display_labels=labels)
    plt.show()

    plot_decision_boundary(clf)


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
    end_idx_train = -int(-x.shape[0] // (1 / test_size + np.finfo(float).eps))
    indices = np.random.permutation(x.shape[0])
    train_idx, test_idx = indices[:end_idx_train], indices[end_idx_train:]
    return (
        x[train_idx, ...], x[test_idx, ...],
        y[train_idx, ...], y[test_idx, ...]
    )


def pre_processing(x: np.ndarray, xv: np.ndarray, xt: np.ndarray):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + np.finfo(np.float64).eps
    pca = PCA(2)
    pca.fit((x - mean) / std)
    return (
        pca.transform((x - mean) / std),
        pca.transform((xv - mean) / std),
        pca.transform((xt - mean) / std)
    )


def plot_decision_boundary(clf):
    return


if __name__ == '__main__':
    main()
