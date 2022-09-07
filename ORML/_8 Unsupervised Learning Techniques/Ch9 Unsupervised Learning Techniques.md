# Ch9 Unsupervised Learning Techniques

## Overview

<u>Clustering</u>:

- Group similar instances together into clusters.
- great tool for data analysis, customer segmentation, recommender systems, search engines, image segmentation, semi-supervised learning, dimensionality reduction and more.

<u>Anomaly detection</u>:

- what "normal" data looks like
- then use that to detect abnormal instances
  - such as defective items on a production line or a new trend in a time series

<u>Density estimation</u>:

- estimating the *probability density function* (PDF) of the random process that generated the dataset
- commonly used for anomaly detection:
  - instances located in very low-density regions are likely to be anomalies
- also useful for data analysis and visualizations

## Clustering

<u>the task of identifying similar instances and assigning them to *clusters*, or groups of similar instances.</u>

- <u>hard clustering</u>: assigning each instance to a single cluster
- <u>soft clustering</u>: give each instance a score per cluster
  - the score can be the distance toward the centroid
  - can also similarity score (affinity) such as the Gaussian RBF

### K-Means

- K-Means algorithm does not behave very well when the blobs have very different diameters (since it only cares the distance toward the centroid)
- remember to scale the input features before using K-Means: else it will perform worse

#### Algorithm

Recursion 3 steps until stop changing or reach maximum loop

#### Accelerated K-Means

This is the algorithm the `KMeans` class uses by default (by Charles Elkan, with triangle inequality, track of bounds...)

- you can force it to use the original K-Means algorithm by setting `algorithm="full"`

#### Mini-batch K-Means

Instead of using the full dataset at each iteration, the algorithm is capable of using mini-batches, moving the centroids just slightly at each iteration

- speeds up the algorithm typically by a factor of three or four
- with large k: bad inertia performance
  - *inertia*: the mean squared distance between each instance and its closest centroid



#### Finding The Optimal Number of Clusters

if k is too small: separate clusters get merged

if k is too large: some clusters get chopped into multiple pieces

**Method**:

- Plot the graph of inertia as a function of k, and find the point of elbow
- Plot the graph of silhouette score as a function of k:
  - <u>silhouette score</u>: is the mean *silhouette coefficient* over all the instances
    - silhouette coefficient: $\frac{(b-1)}{\max(a, b)}$
      - a: mean distance to the other instances in the same cluster (the mean intra-cluster distance)
      - b: mean nearest-cluster distance (the mean distance to the instances of the next closest cluster, defined as the one that minimizes b, excluding the instance's own cluster)
    - very between -1 and +1
      - close to +1 means the instance is well inside its own cluster and far from other clusters
      - close to 0 means it is close to a cluster boundary
      - close to -1 means that the instance may have been assigned to the wrong cluster
  - More computational expensive

- Plot the <u>silhouette diagram</u>:
  - plot every instance's silhouette coefficient, sorted by the cluster they are assigned to and by the value of the coefficient
    - Height: indicates the number of instances in the cluster
    - Width: represent each instances (sorted) silhouette coefficient in the cluster\
    - We want many of the instances Silhouette coefficient be higher than the dashed line (the Silhouette Score: average of all Silhouette coefficient)
    - Sometimes it might be good to have clusters with similar size (?)

#### Limits of K-Means

- Merits: fast and scalable

- Limits:
  - necessary to run the algorithm several times to avoid suboptimal solutions
  - need to specify the number of clusters
  - does not behave very well when the clusters have varying sizes, different densities, or nonspherical shapes

**Depending on the data, different clustering algorithms may perform better**

#### Clustering For Preprocessing Sometimes can improve the performance of model

### DBSCAN (Density-based spatial clustering of applications with noise)

<u>algorithm defines clusters as continuous regions of high density</u>

- DBSCAN can't be used to predict a new instance's cluster: you will need to use KNN on the top of DBSCAN
- very simple algorithm
- capable of identifying any number of clusters of any shape
- robust to outliers
- only two hyperparameters
- can't deal with dataset with density varies significantly across clusters
- time complexity: $O(m\log m)$
- SKLearn space complexity: $O(m^2)$

### Other Clustering Algorithms

#### Agglomerative Clustering

- a hierarchy of clusters is built from the bottom up
- each iteration, agglomerative clustering connects the nearest pair of clusters
- scales very well to large numbers of instances or clusters
  - but must provide a connectivity matrix
- can capture clusters of various shapes
- produces a flexible and informative cluster tree

#### BIRCH

- Balanced Iterative Reducing and Clustering yusing Hierarchies
- built specifically for large datasets
- faster when the number of features is not too large (<20)

#### Mean-Shift

- a bit similar to DBSCAN
- but like to chop clusters into pieces
- bad computational complexity

#### Affinity propagation

- using voting system
- can detect any number of clusters of different sizes
- a computational complexity of $O(m^2)$: bad

#### Spectral clustering

- creates a low dimensional embedding from a similarity matrix
- use another clustering algorithm in this low dimensional space
- can capture complex cluster structures, and it can also be used to cut graphs
- does not scale very well to large number of instances
- does not behave well when the clusters have very different size

## Gaussian Mixtures

A <u>Gaussian Mixtures Model</u> (GMM) is a probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown.

- the algorithm is very close the K-Means but it every time tries to find the parameters rather than calculating the distance
- EM can end up converging to poor solutions, so it needs to be run several times and keeping only the best solution
- <u>generative model</u>: you can sample new instances from it
- density: by estimates the log of the <u>probability density function</u> (PDF) at that location.
  - greater the score, higher the density

### Constraint model to converge for complex data

- set `covariance_type`:
  - `spherical`: all clusters must be spherical (but can have different diameters) (i.e., different variances)
  - `diag`: clusters can take on any ellipsoidal shape of any size, but the ellipsoid's axes must be parallel to the coordinate axes (i.e., the covariance matrices must be diagonal)
  - `tied`: all clusters must have the same ellipsoidal shape, size, and orientation (i.e., all clusters share the same covariance matrix)

### Complexity

k: number of clusters, m: number of instances, n: dimensions/number of features

- `spherical` or `diag` method: $O(kmn)$
- `tied` or `full` method: $O(kmn^2+kn^3)$
  - does not scale well to large number of features

### Anomaly Detection 

<u>Anomalies</u>, or <u>outliers</u> are the instances that deviate strongly from the norm

- Gaussian Mixtures can been used to detect anomalies by comparing the density (probability) and a threshold
- when outliers is too many, it might bias model's view of "normality"
  - so run and delete extreme outliers at first
  - and then run the algorithms some times later 

####  Novelty Detection

- assumes it is trained on a "clean" dataset, uncontaminated by outliers
- outlier detection is often used to clean up a dataset

### Decide the Number of Clusters

Use the number the minimizes a <u>theoretical information criterion</u>:

- <u>Bayesian information criterion</u> (BIC)
  - tends to be simpler but to not fit the data quite as well
    - especially for larger datasets
- <u>Akaike information criterion</u> (AIC)

### Bayesian Gaussian Mixture Models

the model that automatically discard the unnecessary clusters (no need to manually decide cluster numbers)

## Other Algorithms for Anomaly and Novelty Detection

### PCA 

And other dimensionality reduction techniques with an `inverse_transfrom()` method

- the reconstruction error of a anomaly instance is much larger than the reconstruction error of a normal instance

### Fast-MCD (minimum covariance determinant)

- assumes the data are generated from a single Gaussian distribution
- and all outliers are not generated from this Gaussian distribution
- gives a better estimation of the elliptic envelope and thus makes the algorithm better at identifying the outliers

### Isolation Forest

- efficient algorithm for high-dimensional datasets
- use random forest to find anomalies

### Local Outlier Factor

- compares the density of instances around a given instance to the density around its neighbors
- an anomaly is often more isolated than its k nearest neighbors

### One-class SVM

- better suited for novelty detection
- Use SVM to map the data into lower dimension
- find the data around a small compact region (all outsiders are outliers)
- works great for high-dimensional datasets
- but does not scale to large datasets
