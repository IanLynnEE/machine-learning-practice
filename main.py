# Determine 3 types of wines from 13 features.
# Warning: Labels need to be integer.

import csv
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate
from sklearn.decomposition import PCA

def _split_data():
    ''' Randomly pick 18 instances from each type as testing data. '''
    with open('Wine.csv', 'r', newline='') as f: 
        data = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))
    a = [x for x in data if x[0] == 1]
    b = [x for x in data if x[0] == 2]
    c = [x for x in data if x[0] == 3]
    random.shuffle(a); random.shuffle(b); random.shuffle(c);
    with open('test.csv', 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(a[0:18])
        writer.writerows(b[0:18])
        writer.writerows(c[0:18])
    with open('train.csv', 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(a[18:])
        writer.writerows(b[18:])
        writer.writerows(c[18:])
    return

def _read_data(filename, shuffle=False):
    '''
    Read data and return features and labels.
    Input: filename.
    Output: 
        x as an array with shape: (num_data, num_features_per_data)
        y as a list with length of num_data.
    '''
    with open(filename, 'r', newline='') as f: 
        data = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))
    if shuffle == True:                     # For test data.
        random.shuffle(data)
    x = [row[1:] for row in data]
    y = [int(row[0]) for row in data]       # Labels are integer. 
    return np.array(x), y

class MyClassifier:
    def __init__(self, data, labels):
        self.data = data            # (num_data, num_features_per_data)
        self.labels = labels
        self.unique_labels = dict(zip(*np.unique(y, return_counts=True)))
        self.num_data = np.shape(x)[0]
        self.num_features = np.shape(x)[1]
        self.likelihoods = {}       # {label: [stats.norm for each feature]}
        self.priors = {}            # {label:  Prior for each label (float)}

    def train(self):
        '''
        Calculate likelihood and prior distribution for each label.
        Assuming all features are normal distribution.
        '''
        L = 0
        for label, counter in self.unique_labels.items():
            # shape of features: (num_data_per_label, num_data_per_data)
            features = self.data[L:L+counter].T
            L += counter
            self.likelihoods[label] = []
            for i in range(self.num_features):
                mean = np.mean(features[i])
                std = np.std(features[i])
                self.likelihoods[label].append(stats.norm(mean, std))
            self.priors[label] = counter / self.num_data

    def _inference(self, tests, delta):
        '''
        tests: Test dataset.
        dalta: Integrate region.
        posts: Posterior for each test data. Each element is a dict:
            {label: posterior}
        '''
        posts = [{} for _ in range(np.shape(tests)[0])]
        for counter, data in enumerate(tests):
            for label in self.unique_labels:
                tmp = self.priors[label]
                for i in range(self.num_features):
                    tmp *= integrate.quad(self.likelihoods[label][i].pdf, 
                            data[i], data[i]+delta)[0]
                posts[counter][label] = tmp
        return posts

    def predict(self, tests):
        result = [] 
        for post in self._inference(tests, 1e-6):
            result.append(max(post, key=post.get))
        return result

if __name__ == '__main__':
    # Part 1
    _split_data()

    # Part 2
    x, y = _read_data('train.csv')
    xt, yt = _read_data('test.csv', shuffle=True)
    detector = MyClassifier(x, y)
    detector.train()
    result = detector.predict(xt)

    correct = 0
    for i in range(len(yt)):
        if result[i] == yt[i]:
            correct += 1
    print('Acc:', correct/len(yt))

    # Part 3
    


