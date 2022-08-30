# Ch6 Decision Trees

## Principle of Decision Trees

#### Gini Impurity: *how pure is it?*

$$
G_i = 1 - \sum^{n}_{k = 1}p_{i, k}^2
$$

If this node ends in a relative specific group (not mixed, messed), then it has a higher impurity

### CART Cost Function:

$$
J(k, t_k) = \frac{m_l}{m}G_l + \frac{m_r}{m}G_r
$$

(weighted sum of impurity)

- CART is a greedy algorithm (deciding one step does not consider how it will influence steps after)
- Not a greedy algorithm which finds the optimal tree (guaranteed) is known to be an *NP-Complete* problem: $O(\exp(m))$ 

#### Entropy

concept from physics
$$
H_i = - \sum^n_{k=1, p_{i, k}\ne1}p_{i, k}\log_2(p_{i, k})
$$

- Gini impurity is slightly faster to compute
- most of time Gini impurity is similar trees with entropy
- but when they differ, Gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees

## Qualities of DT

- They require very little data preparation: they don't require feature scaling or centering at all.

- Decision Trees are intuitive, and their decisions are easy to interpret: a <u>white box model</u>
  - <u>black box model</u> are the models that its decisions can be hardly explained, like random forest or Neural Networks

- Decision Trees generally are approximately balanced: computational complexity of $O(\log_2(m))$ and independent of number of features: predictions are very fast!
- training algorithm $O(n\times m \log_2(m))$
  - for a small training sets (less than a few thousand instances), the training process can be speeded up by presorting the data (set `presort=True`) 

## Regularization Hyperparameters

- Decision Trees are *nonparametric model*:
  - they make very few assumptions about the training data
  - don't have pre-determined amount of parameters
  - have the tendency to overfit the data



**Reduce the freedom during training!**

- `max_depth`: the maximum depth of the Decision Tree
- `min_samples_split`: the minimum number of samples a node must have before it can split
- `min_samples_leaf`: the minimum number of samples a leaf node must have
- `min_weight_fraction_leaf`: `min_samples_leaf` in fraction of the total number of weighted instance
- `max_leaf_nodes`: the maximum number of leaf nodes
- `max_features`: the maximum number of features that are evaluated for splitting at each node

increasing `min_*` hyperparameters or reducing `max_*` hyperparameters will regularize the model



### Also <u>pruning</u> unnecessary nodes can regularize the decision tree



## Instability

- very sensitive to training set rotation!
  - So use Principal Component Analysis: often results in a better orientation of the training data
- very sensitive to small variations in the training data
  - Since the training data used by SKLearn is stochastic, it is possible to get very different models even on the same training data

