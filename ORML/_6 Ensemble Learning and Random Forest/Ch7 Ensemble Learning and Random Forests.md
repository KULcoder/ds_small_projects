# Ch7 Ensemble Learning and Random Forests

<u>Ensemble Learning</u>: using a group of predictors

<u>Ensemble method</u>: an Ensemble Learning algorithm



*you will use ensemble methods near the end of a project, once you have already built a few good predictors, to combine them into an even better predictor*



#### <u>Hard Voting Classifier</u>: 

predict the class that gets the most votes from different predictors

- this method often achieves a higher accuracy than the best classifier in the ensemble
- even if each classifier is a <u>weak learner</u> (meaning it does only slightly better than random guessing), the ensemble can still be a <u>strong learner</u>, provided there are a *sufficient number* of weak learners and they are *sufficiently diverse*
- ensemble methods works well for different predictors:
  - either they are stochastic
  - or they are very different algorithms

#### <u>Soft Voting Classifier</u>:

if all the classifiers are able to estimate class probabilities (i.e., all have`predict_proba()` method), predict the highest class probability, averages over all the individual classifiers.

- Often perform better than hard voting classifier, since they are able to weight highly confident predictions more



## Popular Ensemble Methods

### Bagging (*bootstrap aggregating*)

<u>use the **same predictor** but train them on **different random subsets** of the training set, and the sampling is performed **with** replacement</u>

- when sampling is performed without replacement, it is called <u>pasting</u>
- generally, the net result is that the ensemble has a similar bias but a lower variance than a single predictor trained on the original training set
- parallel method: scale very well
- SKLearn `BaggingClassifier` automatically use soft voting when `predict_proba()` method exist
- overall, bagging often results in better models than pasting (why?)

#### Out-of-Bag Evaluation

<u>Use the sample that is not used by a single predictor to evaluate that predictor's performance</u>

#### Random Patches

<u>Both sampling the instances and the features</u>

#### Random Subspace

<u>keeping all training instances but sampling features</u>

**Sampling features results in even more predictor diversity, trading a bit more bias for a lower variance**

### Boosting

<u>Any ensemble method that can combine several weak learners into a strong learner</u>:

- general idea is train predictors sequentially, each trying to correct its predecessor.

#### AdaBoost (Adaptive Boosting)

pay more attention on the underfit instances (focus more and more on the hard cases)

- by giving weight toward the misclassified instances by the previous predictor
- hard to be parallelized: does not scale as well as bagging
- when your `AdaBoost` ensemble is overfitting the training set, you can try to reducing the number of estimators or more strongly regularizing the base estimator.

#### Gradient Boosting

tries to fit the new predictor to the <u>residual errors</u> made by the previous predictor.

- To find the optimal number of trees, you can use early stopping:

### Stacking

<u>Instead of voting, use another ML model to decide the final result from all other models</u>

- remember to split the training set such that the instance used to train final ml model is clean.
- use `DESlib` as an open source choice
  - https://github.com/Menelau/DESlib



## Random Forest

- With a few exceptions, a `RandomForestClassifier` has all the hyperparameters of a `DecisionTreeClassifier` and of a `BaggingClassifier`

- In additional to the random forest, you can even make the thresholds for tree building to be random: *extremely randomized tree* ensemble (or <u>Extra-Trees</u> for short):
  - trades more bias for a lower variance
  - use `ExtraTreesClassifier` in `sklearn.ensemble`

- Random Forest allows a easy way to measure the relative importance of each feature
  - how much the tree nodes use that feature reduce impurity on average