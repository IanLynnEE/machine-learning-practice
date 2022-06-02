# ML Homework 3

Neural Network from scratch for image classification.

- [Run](#run)
- [Result](#result)
- [Implementation](#implementation)
  - [Preprocessing](#preprocessing)
  - [Network](#network)
- [Decision Boundary](#decision-boundary)
  - [Increase Layers](#increase-layers)
  - [Increase Units](#increase-units)
  - [Surmise](#surmise)

## Run

Please make sure:
- the training and test dataset must have the same set of labels
- images should be black and white
- images must have the same dimensions.

Please put the following files in the working directory.

```
working_directory
├── Data_test
│   ├── Carambula
│   ├── Lychee
│   └── Pear
├── Data_train
│   ├── Carambula
│   ├── Lychee
│   └── Pear
├── main.py
└── model.py
```

Now, simply run `python3 main.py`. There are some options:

```bash
options:
  --layers LAYERS
  --hidden_units HIDDEN_UNITS
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --train_path TRAIN_PATH
  --test_path TEST_PATH
```

The `LAYERS` is set to be the number of hidden layers between input and output.

The `HIDDEN_UNIT` is set to be the same for all hidden layers, and a bias unit is added to each layer.

The following figure is said to be two layers with 5 hidden units.

![two_layers_five_hidden_units](images/two_layers_five_hidden_units.jpg)



## Result

Due to the randomness of splitting training and validation data and initial weights, the result differs in each run.

For the network above (2 hidden layers, 5 hidden units), the best accuracy is around 96%.

![2-5-1000-acc](images/2-5-1000/acc.png)

![2-5-1000-loss](images/2-5-1000/training_loss.png)

For the network (3 hidden layers, 5 hidden units), the best accuracy is around 96%, but the average result is better in my observation.

![3-5-1000-acc](images/3-5-1000/acc.png)

![3-5-1000-loss](images/3-5-1000/training_loss.png)



## Implementation

### Preprocessing

#### Load Images

All the images in the dataset are 32x32, and the alpha channel is not used for each image.

Therefore, 1024 pixels are taken as features of each image.

For larger images, please check [get_tiny_images.py](https://github.com/IanLynnEE/ImageClassifier/tree/main/part1).


#### PCA

Each feature is standardized before PCA.

```python
def reduce_dimension(x, xt):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + np.finfo(np.float64).eps
    pca = PCA(2)
    return pca.fit_transform((x-mean) / std), pca.transform((xt-mean) / std)
```

There are two ways of standardizing. If we standardize along axis 1, it's like we reset the exposure and dynamic of every image to be the same. For this task, however, the result indicates it's not a good idea.

The shape of an object is a more important factor for this task. The standard scaler, therefore, works along axis 0. It is set to enhance the "different" for each pixel in the same location. And the PCA can focus on comparing and transforming those differences.

The following code actually performs worse:

```python
def reduce_dimension(x: np.ndarray, xt: np.ndarray):
    exposure = np.mean(x, axis=1, keepdims=True)
    dynamic = np.std(x, axis=1, keepdims=True) + np.finfo(np.float64).eps
    x = (x - exposure) / dynamic
    exposure = np.mean(xt, axis=1, keepdims=True)
    dynamic = np.std(xt, axis=1, keepdims=True) + np.finfo(np.float64).eps
    xt = (xt - exposure) / dynamic
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + np.finfo(np.float64).eps
    pca = PCA(2)
    return pca.fit_transform((x-mean) / std), pca.transform((xt-mean) / std)
```


#### Split Training and Validation Data

To avoid overfitting, validation data is used to determine the number of epochs.

This function mimics the behavior of the function provided by sklearn.



### Network

I initially built a two hidden layers network with batch gradient descent. And then, I put a little tweak to make it SGD.

In the end, some modification can make the network works on 3 and more layers.

I will talk about the two layers network base on SGD, but BGD and the 3 layers network share the same idea.

#### Activation Function

For all layers except the output layer, the activation function is the sigmoid function.

```python
def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))
```

The output layer uses the softmax function[^1].

```python
def softmax(x):
    if x.ndim == 2:
        exps = np.exp(x - x.max(axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    exps = np.exp(x - x.max())
    return exps / np.sum(exps)
```


#### Loss Function

Cross-Entropy is used as it's a classification task.

```python
def cross_entropy(yt, yp):
    return -np.sum(yt * np.log(yp)) / yt.shape[0]
```


#### Training

For BGD, the training follows steps:

1. Feed forward
2. Backpropagation
3. Update weights
4. Evaluate model.

For SGD, it's the same except that only one data is fed into steps 1, 2, and 3. Repeat `batch_size` times, and it goes to step 4 to finish an epoch.

We need to use weights for feed forward, so weights are initialized[^2].

```python
# n_h: number of hidden units; n_f: number of features; n_o: number of labels
self.w[0] = np.random.rand(n_h, n_f + 1) * np.sqrt(1/n_h)
self.w[1] = np.random.rand(n_o, n_h + 1) * np.sqrt(1/n_o)
```

It's the same for BGD and SGD. In general,

```python
self.w[n] = np.random.rand(f_out, f_in) * np.sqrt(1/fan_out)
```

#### Feed Forward

For BGD, the input shape is (number of data, number of features). 

I first add a bias node to form the hidden layer 0:

```python
self.unit[0] = np.insert(x, 0, 1, axis=1)
```

And $A = Z W^T$.

```python
self.act[1] = self.unit[0] @ self.w[0].T
self.unit[1] = np.insert(sigmoid(self.act[1]), 0, 1, axis=1)
```

Following the definition in the textbook, the activations pass to the activation function to form the next layer. A bias node is added in the process. 

The activation function for the output layer (the last layer) is set to be the softmax function, and no bias node is needed.

```python
self.act[2] = self.unit[1] @ self.w[1].T
self.unit[2] = softmax(self.act[2])
```

For SGD, since the input shape is (number of features, ), simply switch to $A = W \cdot Z$.

For a deeper network, it follows the same feed-forward steps. Increase the number of repeated $A = Z W^T$.


#### Backpropagation

From right to left, we need to compute errors (5.56).

For BGD,

```python
# delta[max_idx] = output - y
delta[k] = delta[k+1] @ self.w[k][:, 1:] * sigmoid(self.act[k], True)
```

And the derivative can be obtained (5.53).

```python
dw[k-1] = delta[k].T @ self.unit[k-1] / current_batch_size
```

The index is confusing. Following is a tricky way to do this.

![backprop](images/backprop.png)

```python
delta = self.unit[self.n_l] - y
for i in range(self.n_l-1, -1, -1):
    dw[i] = np.outer(delta, self.unit[i])
    delta = self.w[i][:, 1:].T @ delta * sigmoid(self.act[i], True)
```

For SGD, it takes minor modifications to handle the dimension problem.


#### Update

For BGD and SGD, the optimize step is simple. 

```python
self.w[k] -= grad[k] * learning_rate
```

For SGD, if the validation loss gets smaller, the weight is saved as the best model.

For BGD, this is on the TODO list.



<div style="page-break-after: always;"></div>

## Decision Boundary

As mentioned above, the randomness of splitting training and validation data and initial weights makes it hard to analyze.

Following are two decision boundaries using the same parameters for training.

![2-5-1000-decision-compare](images/2-5-1000/decision_boundary_compare.png)

The shape of the decision region is relatively simple with few layers and units.


### Increase Layers

When we use 3 layers, the decision region seems to be more complicated. It does not guarantee a better result, but I find it usually performs better than the average case of 2 layers.

![3-5-1000-decision-compare](images/3-5-1000/decision_boundary_compare.png)

The better performance comes with the cost of the need for a larger dataset. The 3 layers network needs more epochs to train, and as I'm using SGD, the times of resuing data increase. This will raise the concern of overfitting.

![3-5-1000-overfitting](images/3-5-1000/overfitting.png)

The 3 layers network should be enough, but we can try the 4 layers to check the shape of the decision region. The accuracy can hit 96.6%, and the shape now has even sharper corners.

![4-5-1000-decision](images/4-5-1000/decision_boundary.png)


### Increase Units

Go back to the 2 layers network. If I used 6 hidden units, it seems the decision region is complicated as well.

![2-6-1000-decision](images/2-6-1000/decision_boundary.png)

In the training loss figure, even with the same epochs * batch_size, the overfitting is more presented in comparison to 5 units.

![2-6-1000-loss](images/2-6-1000/training_loss.png)

On the other hand, when using 3 units, the boundary will keep simple, even with 91% accuracy.

![2-3-1000-decision](images/2-3-1000/decision_boundary.png)


### Surmise

More layers and more hidden units make the network more complicated. As the result, the boundary region can be sharper or more complicated. Thus, it increases the risk of overfitting, but it can yield a better result as it "fits" better. Depending on the variance of training and test data, the conclusion varies.

For this task, I would choose the 3 layers network with 5 units.

Please note that the images in `images/2-5-1000` come from the best case. It does not represent the average case.



[^1]: https://cs231n.github.io/linear-classify/#softmax
[^2]: https://github.com/lionelmessi6410/Neural-Networks-from-Scratch