# Ch5 Support Vector Machines

- particularly well suited for classification of complex small or medium sized datasets

<u>Not only separates the two classes but also stay as far away from the closest training instances as possible</u>: **large margin classification**

- SVM is sensitive toward feature scaling: remember to do them

## Soft Margin Classification

allows the existence of misclassifications:

- keep the street as large as possible
- and limit the margin violations

## Parameters

### C:

if the SVM Model is overfitting, you can try regularizing it by reducing C

## Nonlinear SVM Classification: Kernels

One idea to use SVM on the non-separatable data is to generate more features from itself to make it separable.

### Polynomial Kernels

Using the kernel tricks, we can avoid to transform the whole datasets into huge number of features, it can directly calculate them out.

#### Parameters

##### coef0

controls how much the model is influenced by high-degree polynomials versus low-degree polynomials

### Similarity Features

add features computed using a **similarity function**: 

- measures how much each instance resembles a particular landmark

Check the graph at P160!

### Some Other Kernels

<u>String Kernels</u>: when classifying text documents or DNA sequences

...

### How to Choose the Kernel?

- Should always try the linear kernel first, especially when the training set is very large or if it has plenty of features
  - `LinearSVC` is much faster than `SVC(kernel="linear")`

- If the training set is not too large, should also try the Gaussian RBF kernel
- when have enough time and computing power, experiment with a few other kernels, using cross-validation and grid search. (especially if there are kernels specialized for your training set's data structure)

## SVC TIME Complexity

SVM has a very bad time complexity against instances numbers

(don't use it on a dataset with many instances, but it is efficient against features numbers)

- this is caused by those kernel tricks



## SVM Regression

<u>Rather than trying to give more space, we wish our streets (margins) to contain as many instances as possible while limiting margin violations</u>