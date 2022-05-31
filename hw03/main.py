import os
import argparse

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

from model import TwoLayersClassifier

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

    if args.layer == 2:
        clf = TwoLayersClassifier(args.hidden_node)
    clf.fit(x, y)
    yp = clf.predict(xt)
    print(accuracy_score(yt, yp))
    ConfusionMatrixDisplay.from_predictions(yt, yp)
    plt.show()
