## Ch4 Training Models

- In many situations you don't really need to know the implementation details
- However, this can:
  - help you quickly home in on the appropriate model
  - find the right training algorithm
  - find the good set of hyperparameters
  - help you debug issue and performance error
  - will be essential to understanding, building, and training neural networks

## Linear Model

## SVD?

- Gradient Descent Technique is guaranteed to work out, if
  - wait long enough (vs. learning rate)
  - learning rate is not too fast

## Gradient Descent

Try to make sure that all features have a similar scale when using gradient descent, else it is going to take a much longer time to converge

### Batch Gradient Descent

A method to use the full training set at each gradient descent step: uses the whole batch of training data at every step

- Terribly slow on very large training set

### Stochastic Gradient Descent

Compute the gradient only on one single random instance in the training data set:

- much faster: requires very small amount of iteration
- much less regular
- final values are good, but not optimal
- because the cost function jumps around, this method has a better chance to jump out of local minima and find the global minimum compare to batch gradient descent
- Training Instances must be independent and identically distributed (IID)
  - Shuffle the instance can easily helps

### Mini-Batch Gradient Descent

Randomly pick subset of training instances to calculate the gradient

- harder to escape local minima than stochastic GD
- but bounce less and are more regular (less random)
- will not bounce too much around optimal points compares to stochastic GD



| Algorithm       | Large m (number of instances) | Out-of-core support | Large n (number of features) | Hyperparams | Scaling required | Scikit-Learn    |
| --------------- | ----------------------------- | ------------------- | ---------------------------- | ----------- | ---------------- | --------------- |
| Normal Equation | Fast                          | No                  | Slow                         | 0           | No               | N/A             |
| SVD             | Fast                          | No                  | Slow                         | 0           | No               | LinearRegressor |
| Batch GD        | Slow                          | No                  | Fast                         | 2           | Yes              | SGDRegressor    |
| Stochastic GD   | Fast                          | Yes                 | Fast                         | >=2         | Yes              | SGDRegressor    |
| Mini-batch GD   | Fast                          | Yes                 | Fast                         | >=2         | Yes              | SGDRegressor    |



## Polynomial Regression

<u>add powers of each feature as new features, then train a linear model on this extended set of features</u>

- Polynomial transforming usually add the combination between different features, too
  - so, polynomial regression is capable of finding relationships between features
- the `PolynomialFeatures(degree=d)` transforms an array containing *n* features into an array containing $(n+d)! / d!n!$ features: grows very quick



## Learning Curves:

plots of the model's performance on the training set and the validation set as a function of the training set size (or the training iteration)

- This curve helps check underfitting and overfitting more clearly than using cross-validation to check the performance.
- it usually should perform better on train set than validation set (at least a little bit)
- this curve might somehow suggests how many instances are suited for our training?

#### Underfitting

will shows the model does poorly both on training set and validation set

- adding more training examples will not help
- both curves have reached a plateu
- both curves are fairly high

#### Overfitting

will shows the model does nice in training set and poorly on validation set: a big gap

- the error on the training data is much lower
- a big gap between curves

#### Correct Situation

will shows the model does nice in training set and validation set

## The Bias/Variance Trade-Off

A model's generalization error can be expressed as the sum of three very different errors

### Bias

Due to wrong assumptions, such as assuming that the data is linear when it is actually quadratic. A high-bias model is most likely to underfit the training data.

### Variance

This part is due to the model's excessive sensitivity to small variations in the training data. A model with many degrees of freedom is likely to have variance and thus overfit the training data.

### Irreducible error

This part is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data

**increase a model's complexity will typically increase its variance and reduce its bias, and the opposite is true**



## Regularized Linear Models

*for a linear model, regularization is typically achieved by constraining the weights of the model*

For most of the regularized models, it is important to scale the data as they are sensitive to the scale of the input features (so e.g., using a `StandardScaler`)

### Ridge Regression

a *regularization term* equal to $\alpha\sum_i^n\theta_i^2$ is added to the cost function

- helps flat prediction
- reduce the model's variance but increasing



### Early Stopping

Another different way to regularize iterative learning algorithms is to stop training as soon as *the validation errors reaches a minimum* (the training is vs. epoch: the time you use the whole training set through the model)



## Logistic Regression

commonly used to estimate the probability that an instance belongs to a particular class

(it is a classifier model!)

### Defining Estimated Probability (vectorized form)

$$
\hat{p} = h_\mathbf{\theta}(\mathbf{x}) = \sigma(\mathbf{\theta}^T\mathbf{x})
$$

#### Sigma Function

$$
\sigma(t) = \frac{1}{1 + e^{-t}}
$$

- 0 if t < 0.5
- 1 if t >= 0.5

### Cost Function: *log loss*

$$
J(\mathbf{\theta}) = -\frac{1}{m}\sum^m_{i=1}[y^{(i)}\log (\hat{P^{(i)}})+(1-y^{(i)})\log(1-\hat{P}^{(i)})]
$$

- this cost function has no known closed-formed solution
- but it is convex that GD is guaranteed to find the global minimum

## Softmax Regression

*Multinomial Logistic Regression*: support multiple classes directly







# General Concepts

- Training a model means searching for a combination of  model parameters that minimizes a cost function
  - the more parameters a model has, the more dimensions this space has, and the harder the search is.