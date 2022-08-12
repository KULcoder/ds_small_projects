# Ch3 Classification

### General Structure of SKLearn Datasets

- A `DESCR` key describing the dataset
- A `data` key containing an array with one row per instance and one column per feature
- A `target` key containing an array with the labels

### Stochastic Gradient Descent (SGD)

- Being capable of handling very large datasets efficiently.



## Classifier Performance

### Accuracy

Not preferred for classifiers!

- Especially for *skewed datasets* (i.e., when some classes are much more frequent than others)

### Confusion Matrix

Show the number of **true negatives, false negatives, false positives, true positives**

### Precision

*out of all positive predictions (TP + FP), what is the proportion of correctly prediction (TP)*
$$
\text{precision} = \frac{TP}{TP+FP}
$$

### Recall / Sensitivity / True Positive Rate(TPR)

*out of all truly positive objects (TP + FN) , what is the proportion of correctly predicted* (TP)
$$
\text{recall} = \frac{TP}{TP+FN}
$$

### F1 Score

A simply way to combine Precision and Recall, which evaluate them as the same importance
$$
F_1 = \frac{2}{\frac{1}{\text{precision}} + \frac{1}{recall}} = 2\times\frac{\text{precision}\times\text{recall}}{\text{precision + precision}} = \frac{TP}{TP+\frac{FN+FP}{2}}
$$

- The F1 score favors classifiers that have similar precision and recall



*Increasing precision reduces recall, and vice versa*



### Precision/Recall Trade-Off

- `sklearn` can't let us directly change model's threshold, but we can use `model.decision_function(x)` to obtain the score of this `x` in the model, and then, we can compare it to our manually set threshold

### 

### ROC Curve

***receiver operating characteristic*** (ROC)

*true positive rate(recall)* against *false positive rate*
$$
FPR = \frac{FN}{FN + TN}=1 - TNR=1-specificity
$$


### ROC Curve vs PR Curve

- Prefer PR when
  - positive class is rare
  - we care more about the false positives than the false negatives (care about precision more?)











```python
# this is a code for a customized cross validation
from sklearn.model_selection import StratfiedFold
from sklearn.base import clone

skfolds = StrateKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train):
    clone_clf = clone(some_clf) # some_clf can be replaced by any model
    X_train_folds = X_train[train_index]
    y_train_folds = y_train[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
```



## Multiclass Classification

*multiclass classifiers (also called multinomial classifiers)* can distinguish between more than two classes

##### Following algorithms supports multiple classes natively:

- Logistic Regression Classifiers
- Random Forest Classifiers
- Naïve Bayes Classifiers
- ...

##### Following algorithms are strictly binary classifiers

- SGD Classifiers
- Support Vector Machine Classifiers
- ...

### How to use the strictly binary classifiers for multiclass classification tasks?

#### Method1: *one-versus-the-rest* (OvR) / *one-versus-all*

- Mimic the multiclass classification tasks into multi binary classification task on yes or no

- To decide, compare the decision score (choose the highest score)
- have to train $N$ models
- preferred by most binary classification algorithms

#### Method2: *one-versus-one* (OvO)

- turn the multiclass classification tasks into multi binary classification tasks between different categories
- To decide, check for the model which wins the most duels
- have to train $N\times (N-1)/2$ models
- benefit: each classifier only needs to be trained on the part of the training set for the two classes that it must distinguish
- preferred by SVM

### Those two methods are automatically decided by SKLearn when you try  to use a binary classification algorithm for a multiclass classification task



## Multilabel Classification

<u>outputs multiple binary tags</u>

- evaluate by computing average f1 score for different tags (`average='macro'`)
- or by they proportions (`average='weighted'`)

## Multioutput Classification

multilabel classification where each label can be multiclass

