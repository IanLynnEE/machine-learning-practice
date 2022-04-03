# Determine 3 types of wines from 13 features.

import os

import random
import csv

import numpy as np
from scipy import stats
from scipy import integrate

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

# Read data and return 3 numpy array.
def _read_data(filename):
    with open(filename, 'r', newline='') as f: 
        data = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))
    a = [x[1:] for x in data if x[0] == 1]
    b = [x[1:] for x in data if x[0] == 2]
    c = [x[1:] for x in data if x[0] == 3]
    return np.array(a).T, np.array(b).T, np.array(c).T, len(a)+len(b)+len(c)

if __name__ == '__main__':
    # Part 1
    _split_data()
    # Part 2
    a, b, c, num_train = _read_data('train.csv')
    features = [a, b, c]
    num_train = 0 

    distribution = [ [], [], [] ]
    for label in range(3):
        for i in range(13):
            mean = np.mean(features[label][i])
            std = np.std(features[label][i])
            distribution[label].append(stats.norm(mean, std))
   
    priors = [0., 0., 0.]
    for i in range(3):
        priors[i] = np.shape(features[i])[1] / num_train

