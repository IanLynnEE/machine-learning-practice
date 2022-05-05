# ML Homework 2

Predict admit chance by 3 features: GRE, TOEFL, Research Experience.

## Run

Place `Training_set.csv`, `Validation_set.csv`, and `hw2.py` in the same directory. Then,

``` shell
python3 hw2.py -O1 5 -O2 5
```

Default value of both O1 and O2 are 5, as the number of location used in the Gaussian basis function for GRE and TOEFL scores, respectively.

## Result




## Impact of O1 and O2



## Comparison




## Implementation

For forming the feature vector by the Gaussian basis function, I try to utilize the feature of numpy.

```python
# For training, i.e. scale and center are not given.
phi = np.zeros([np.shape(x)[0], O1*O2])
scale = (np.max(x,axis=0) - np.min(x,axis=0)) / np.array([O1-1, O2-1])
center = np.zeros([2, O1*O2])
for i in range(O1):
    for j in range(O2):
        center[:, O2*i+j] = scale * np.array([i,j]) + np.min(x,axis=0)
        dis = x - center[:, O2*i+j]
        dis_scaled = np.square(dis) / (2 * np.square(scale)).T
        phi[:, O2*i+j] = np.exp(-np.sum(dis_scaled, axis=1))
```

Then, the Research Experience and bias will be added to the last 2 elements of the feature vector. 

For Maximum Likelihood and Least Squares,
$$
w = \Phi^{\dagger} t, \ 
y = w^T \phi(x) \ .
$$

In out task, simply use:

```python
w = np.linalg.pinv(phi_train).dot(x[:, 3])
y = phi_test.dot(w)
```

Similarly, in Bayesian Linear Regression,

```python
y = phi_test.dot(m_N)
```

The problem will be how to calculate $m_N$. Let M be the number of features, N be the number of training data.
$$
S_N^{-1} = \alpha I + \beta\Phi^T\Phi,\\
m_N = \beta S_N \Phi^T t,\\
\gamma = \sum_{j=1}^M \frac{\lambda_j}{\alpha+\lambda_j}, \\
\alpha = \frac{\gamma}{m_N^T m_N}, \\
\beta = \frac{N-\gamma}{\sum_{i=1}^N (t_i - m_N^T \phi(x_i))^2} .
$$
That is:

1. Set initial value of $\alpha,\ \beta$ 
2. Calculate $S_N^{-1},\ m_N$ from $\alpha,\ \beta$
3. Calculate $\alpha,\ \beta$ from $S_N^{-1},\ m_N$
4. Go back to step 2 if the change in $\alpha,\ \beta$ still large.
