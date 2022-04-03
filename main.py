# Determine 3 types of wines from 13 features.

import os

import random
import csv

import numpy as np
from scipy import stats
from scipy import integrate
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Randomly pick 18 instances from each type as testing data.
def _split_data():
    with open('Wine.csv', 'r', newline='') as f: 
        data = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))
    random.shuffle(data)
    a = [x for x in data if x[0] == 1]
    b = [x for x in data if x[0] == 2]
    c = [x for x in data if x[0] == 3]
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

# Read data and return features and label (list).
# The shape of features is (num_data, num_features_per_data_per_data)
def _read_data(filename):
    with open(filename, 'r', newline='') as f: 
        data = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))
    x = [row[1:] for row in data]
    y = [int(row[0]) for row in data]   
    return np.array(x), y

# Calculate likelihood distribution and and prior distribution for each label.
# Assuming all likelihoods are normal distribution, 
# and all priors are uniform distribution.
def _train(x, y):
    likelihood = {}
    priors = {}
    int_num_features = np.shape(x)[1]
    dict_num_instances = dict(zip(*np.unique(y, return_counts=True)))
    L = 0
    for label, count in dict_num_instances.items():
        R = L + count
        features = x[L:R].T
        L += count
        likelihood[label] = []
        for i in range(int_num_features):
            mean = np.mean(features[i])
            std = np.std(features[i])
            likelihood[label].append(stats.norm(mean, std))
        priors[label] = count / len(y)
    return likelihood, priors 

# Calculate postirior for each data in xt from given infos.
def _inference(xt, likelihood, priors, delta):
    int_num_features = np.shape(xt)[1]
    labels = np.unique(yt)                      # list of label
    posts = [{} for _ in range(np.shape(xt)[0])]
    for j, row in enumerate(xt):
        for label in labels:
            tmp = priors[label]
            for i in range(int_num_features):
                tmp *= integrate.quad(likelihood[label][i].pdf, 
                        row[i], row[i]+delta)[0]
            posts[j][label] = tmp
    return posts

def _decision(posts, yt):
    yy = [] 
    for post in posts:
        yy.append(max(post, key=post.get))
    return yy

if __name__ == '__main__':
    # Part 1
    # _split_data()
    
    # Part 2
    delta = 1e-6
    x, y = _read_data('train.csv')
    xt, yt = _read_data('test.csv')
    likelihood, priors = _train(x, y)
    posts = _inference(xt, likelihood, priors, delta)
    yy = _decision(posts, yt)

    correct = 0
    for i in range(len(yt)):
        if yy[i] == yt[i]:
            correct += 1
    print(correct/len(yt))

    # Part 3
    


