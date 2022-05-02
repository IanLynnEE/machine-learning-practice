'''
NTHU EE Machine Learning HW2
Author: Ian Lin (林宗暐)
Student ID: 106061210
'''
import numpy as np
import pandas as pd
import math
import scipy.stats
import argparse

def phi_gbf(x, O1, O2, scale=None, center=None):
    phi = np.zeros([np.shape(x)[0], O1*O2])
    if scale is None:
        scale = (np.max(x,axis=0) - np.min(x,axis=0)) / np.array([O1-1, O2-1])
        center = np.zeros([2, O1*O2])
        for j in range(O2):
            for i in range(O1):
                k = O2*i+j
                center[:, k] = scale * np.array([i, j]) + np.min(x, axis=0)
                dis = x - center[:, k] 
                phi_before_exp = np.square(dis) / (2 * np.square(scale)).T
                phi[:, k] = np.exp(-np.sum(phi_before_exp, axis=1))
        return phi, scale, center
    for j in range(O2):
        for i in range(O1):
            k = O2*i+j
            dis = x - center[:, k] 
            phi_before_exp = np.square(dis) / (2 * np.square(scale)).T
            phi[:, k] = np.exp(-np.sum(phi_before_exp, axis=1))
    return phi 


def BLR(train_data, test_data_feature, O1=5, O2=5): 
    '''
    output: ndarray with size (length of test_data, )
    '''
    return
    return y_BLRprediction 


def MLR(x, xt, O1=5, O2=5):  
    '''
    input:
        x    : training data with label
        xt   : test_data_feature
    output: ndarray with size (length of test_data, )
    '''
    phi_train = np.zeros([np.shape(x)[0], O1*O2+2])
    phi_train[:, O1*O2] = x[:, 2]
    phi_train[:, O1*O2+1] = 1
    phi_train[:, :O1*O2], scale, center = phi_gbf(x[:, :2], O1, O2)
    phi_test = np.zeros([np.shape(xt)[0], O1*O2+2])
    phi_test[:, O1*O2] = xt[:, 2]
    phi_test[:, O1*O2+1] = 1
    phi_test[:, :O1*O2] = phi_gbf(xt[:, :2], O1, O2, scale, center)
    w = np.dot(np.linalg.pinv(phi_train), x[:, 3])
    y_MLLSprediction = np.dot(phi_test, w)
    return y_MLLSprediction 


def CalMSE(data, prediction):
    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean__squared_error = sum_squared_error/prediction.shape[0]
    return mean__squared_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O1', '--O_1', type=int, default=5)
    parser.add_argument('-O2', '--O_2', type=int, default=5)
    args = parser.parse_args()
    O_1 = args.O_1
    O_2 = args.O_2
    
    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy()
    data_test_feature = data_test[:, :3]
    data_test_label = data_test[:, 3]
    
    predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)

    print(f'MSE of MLR = {CalMSE(predict_MLR, data_test_label)}')
    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(
        e1=CalMSE(predict_BLR, data_test_label), 
        e2=CalMSE(predict_MLR, data_test_label)
    ))
    return 0

if __name__ == '__main__':
    main()
