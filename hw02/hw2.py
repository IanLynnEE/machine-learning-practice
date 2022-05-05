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


def _phi_GBF(x, O1, O2, scale=None, center=None):
    '''
    input:
        x      : ndarray with size (length of data, 2 features)
        O1     : int
        O2     : int
        scale  : (optional) ndarray with size (2, )
        center : (optional) ndarray with size (2, O1*O2)
    output:
        phi    : ndarray with size (length of data, O1*O2)
        scale  : ndarray with size (2, )
        center : ndarray with size (2, O1*O2)
    '''
    phi = np.zeros([np.shape(x)[0], O1*O2])
    if scale is None:
        scale = (np.max(x,axis=0) - np.min(x,axis=0)) / np.array([O1-1, O2-1])
        center = np.zeros([2, O1*O2])
        for i in range(O1):
            for j in range(O2):
                center[:, O2*i+j] = scale * np.array([i,j]) + np.min(x,axis=0)
    for i in range(O1):
        for j in range(O2):
            dis = x - center[:, O2*i+j]      # Somehow this is okey.
            dis_scaled = np.square(dis) / (2 * np.square(scale)).T
            phi[:, O2*i+j] = np.exp(-np.sum(dis_scaled, axis=1)) # -(a+b)
    return phi, scale, center


def _fit_BLR(x, y, alpha=1e-5, beta=1e-5, max_iter=100):
    '''
    input:
        x        : ndarray with size (length of data, number of features)
        y        : ndarray with size (length of data, )
        alpha    : (optional) float, initial alpha
        beta     : (optional) float, initial beta
        max_iter : (optional) int
    output:
        m_N      : ndarray with size (length of data, number of features)
    '''
    for i in range(max_iter):
        # Update S_N, m_N, gamma from previous alpha, beta.
        S_N_inv = alpha * np.eye(x.shape[1]) + beta * x.T.dot(x)
        S_N = np.linalg.inv(S_N_inv)
        m_N = beta * S_N.dot(x.T).dot(y)
        eigenvalues = np.linalg.eigvalsh(x.T.dot(x)) * beta
        gamma = np.sum(eigenvalues / (eigenvalues + alpha))
        # Update alpha, beta from above.
        alpha_new = gamma / np.sum(m_N.T.dot(m_N))
        beta_new = (np.shape(x)[0] - gamma) / np.sum(np.square(y - x.dot(m_N)))
        # Check convergence.
        if np.isclose(alpha, alpha_new) and np.isclose(beta, beta_new):
            print(f'Converge after {i + 1} iterations.')
            return m_N
        alpha = alpha_new; beta = beta_new
    print(f'Stop after {max_iter} iterations.')
    return m_N


def BLR(x, xt, O1=5, O2=5): 
    '''
    input:
        x  : ndarray with size (length of train_data, 3 features + 1 label)
        xt : ndarray with size (length of test_data, 3 features)
        O1 : int
        O2 : int
    output:
        prediction : ndarray with size (length of test_data, )
    '''
    phi_train = np.zeros([np.shape(x)[0], O1*O2+2])
    phi_train[:, O1*O2] = x[:, 2]
    phi_train[:, O1*O2+1] = 1
    phi_train[:, :O1*O2], scale, center = _phi_GBF(x[:, :2], O1, O2)
    m_N = _fit_BLR(phi_train, x[:, 3])
    phi_test = np.zeros([np.shape(xt)[0], O1*O2+2])
    phi_test[:, O1*O2] = xt[:, 2]
    phi_test[:, O1*O2+1] = 1
    phi_test[:, :O1*O2], _, _ = _phi_GBF(xt[:, :2], O1, O2, scale, center) 
    return phi_test.dot(m_N)


def MLR(x, xt, O1=5, O2=5):  
    '''
    input & output: same as above.
    '''
    phi_train = np.zeros([np.shape(x)[0], O1*O2+2])
    phi_train[:, O1*O2] = x[:, 2]
    phi_train[:, O1*O2+1] = 1
    phi_train[:, :O1*O2], scale, center = _phi_GBF(x[:, :2], O1, O2)
    w = np.linalg.pinv(phi_train).dot(x[:, 3])
    phi_test = np.zeros([np.shape(xt)[0], O1*O2+2])
    phi_test[:, O1*O2] = xt[:, 2]
    phi_test[:, O1*O2+1] = 1
    phi_test[:, :O1*O2], _, _ = _phi_GBF(xt[:, :2], O1, O2, scale, center)
    return phi_test.dot(w)

def _playground(x, y, xt, yt):
    return
    

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

    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(
        e1=CalMSE(predict_BLR, data_test_label), 
        e2=CalMSE(predict_MLR, data_test_label)
    ))
    return 0

if __name__ == '__main__':
    main()
