import os
import argparse

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score



def load_data(path: str) -> tuple[np.ndarray, list]:
    # Each image MUST have same dimension, and the 2nd channel is all 255.
    images = None
    labels = []
    for label in os.listdir(path):
        if os.path.isdir(os.path.join(path,label)):
            for filename in os.listdir(os.path.join(path, label)):
                labels.append(label)
                img = Image.open(os.path.join(path, label, filename))
                img = np.array(img)[:, :, 0].flatten()
                if images is None:
                    images = img
                else:
                    images = np.vstack([images, img])
    return images, labels


def pre_processing(x: np.ndarray, xt, y_list, yt_list) -> tuple[np.ndarray]:
    scaler = StandardScaler()
    pca = PCA(2)
    encoder = LabelEncoder()
    x_pca = pca.fit_transform(scaler.fit_transform(x))
    xt_pca = pca.transform(scaler.transform(xt))
    y = encoder.fit_transform(y_list)
    yt = encoder.transform(yt_list)
    return x_pca, xt_pca, y, yt


def initialize():
    return


def forward():
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--node', type=int, default=5)
    parser.add_argument('--train_path', type=str, default='Data_train')
    parser.add_argument('--test_path', type=str, default='Data_test')
    args = parser.parse_args()
    x, y_list = load_data(args.train_path)
    xt, yt_list = load_data(args.test_path)
    x, xt, y, yt = pre_processing(x, xt, y_list, yt_list)

    from sklearn.svm import SVC
    clf = SVC(kernel='linear')
    clf.fit(x, y)
    ConfusionMatrixDisplay.from_estimator(clf, xt, yt)
    plt.show()