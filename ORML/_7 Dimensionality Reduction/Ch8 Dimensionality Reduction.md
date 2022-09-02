# Ch8 Dimensionality Reduction

## Main Approaches for Dimensionality Reduction

### Projection

Project the data into a subspace.

### Manifold

a *d*-dimensional manifold is a part of an *n*-dimensional space (where $d<n$) that locally resembles a *d*-dimensional hyperplane 

- Reducing the dimensionality of the training set before training a model will usually speed up training
- but it may not always lead to a better or simpler solution: it all depends on the dataset

## PCA (Principal Component Analysis)

first it identifies the hyperplane that lies closest to the data, and then it projects the data onto it.

- Preserving the maximum variance

### Principal Components

- First principal component is the axis that maximize the amount of variance
- the remaining i^th orthogonal axis is the i^th principal component  
- PCA assumes that the dataset is centered around the origin, so if you want to implement PCA yourself without using SKLearn, don't forget to center the data first

#### SVD (Singular Value Decomposition)

Decompose one matrix into three matrix
$$
\matrix{X}=\matrix{U\Sigma V^T}
$$
which $\matrix{V}$ is the matrix that contains the unit vectors that define all the principal components

#### Project Down to d Dimensions

if you want to project the original dataset into a reduced dataset $X_{d-proj}$:

- defining $W_d$ as the matrix containing the first d columns of $V$

$$
X_{d-proj} = XW_d
$$

#### Explained Variance Ratio

<u>The ratio indicates the proportion of the dataset's variance that lies along each principal component</u>

#### Choosing the Right Number of Dimensions

- Choose the number of dimensions that add up to a sufficiently large portion of the variance (e.g., 95%)

#### PCA for Compression

PCA helps reduce the space used to train the data

- For the `MNIST` dataset, when trying to preserve 95% of the variance, PCA helps compress the dataset from 784 features to 150 features: allows the new feature to be only 20% of the original set
- you can decompress the data back into 784 features, but it will likely to loss some information (within 5 of variance): those error are called <u>reconstruction error</u>

### Randomized PCA

`svd_solver='randomized'`, SKLearn uses a stochastic algorithm called <u>randomized PCA</u> that quickly finds the **approximation of the first *d* principal components**

- It is dramatically faster than full SVD when *d* is much smaller than *n* (number of features/dimensions)
- by default, if m or n is greater than 500 and d is less than 80% of m or n, SKLearn will automatically uses randomized PCA, you can set `svd_solver="full"` to force full SVD

### Incremental PCA

PCA - online: allow you to split and train on small part of the data incrementally

#### Deal with Large Data:

- You can also use NumPy's `memmap` class: this allows you to manipulate a large array stored in a binary file on disk as if it were entirely in memory
- Since `IncrementalPCA` only uses a small part of the data at one time, this allows the transformer to use usual `fit` method on a large local file/data

```python
X_mm = np.memmap(filename, dtype"float32", mode="readonly", shape=(m, n))

batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)
```

### Kernel PCA

The Kernel trick: allows a linear decision boundary in the high-dimensional feature space, which corresponds to complex nonlinear decision boundary in the *original space*

- Also able to be used in PCA that gives a special axis: a complex nonlinear projections for dimensionality reduction
- Good at preserving clusters of instances after projects
- or sometimes even unrolling datasets that lie close to a twisted manifold

#### **Different Kernels works for different Datasets**

##### Use **grid search**

- However,  kPCA is an unsupervised learning algorithm, there is no obvious performance measure to help:
  - So that we have to use a model and grid search to help decide what will be the kernel to use.

##### Measure lowest reconstruction error

- It is not the exact point, but still measurable: (???)
  - possible to find a point in the original space that would map close to the reconstructed point: *pre-image*

## LLE (Locally Linear Embedding)

1. first measuring how each training instance linearly relates to its closest neighbors, 

2. and then looking for a low-dimensional representation of the training set where these local relationships are best preserved

Another powerful <u>nonlinear dimensionality reduction</u> (NLDR):

- A Manifold Learning Technique that does not rely on projections
- particularly good at unrolling twisted manifolds: especially when there is not too much noise
- scale poorly to very large datasets (bad time complexity)

## Other Dimensionality Reduction Techniques

### Random Projections

`sklearn.random_preojections`

- quality of the dimensionality depends on the number of instances and the target dimensionality
- but not on the initial dimensionality

### Multidimensional Scaling (MDS)

Reduces dimensionality while trying to preserve the distances between the instances

### Isomap

Creates a graph by connecting each instance to its nearest neighbors, then reduces dimensionality while trying to preserve the <u>geodesic distances</u> between the instances

- geodesic distances: the number of nodes on the shortest path between the nodes

### t-Distributed Stochastic Neighbor Embedding (t-SNE)

- trying to keep similar instances close
- keep dissimilar instances apart
- mostly used for visualization: particular to visualize clusters of instances in high-dimensional space

### Linear Discriminant Analysis (LDA)

- Learns most discriminative axes between the classes
  - which used to define a hyperplane onto which to project the data
- this projection will keep classes as far apart as possible
- LDA is a good technique to reduce technique to reduce dimensionality before running another classification algorithm such as an SVM classifier