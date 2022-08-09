# Ch2 End-to-End Machine Learning Project

**Will only left some interesting findings here, the whole process will be left on the jupyter notebook**

## Look at the Big Picture



## Scikit-Learn Design

*Consistency*

​	All objects have a consistent and simple interface:



​	*Estimators*

​		Any object that can estimate some parameters based on a dataset is called an *estimator* (e.g., an `imputer `is an estimator). The estimation itself is performed by the `fit()` method, and it takes only a dataset as a parameter (or two for supervised learning algorithms; the second dataset contains the labels). Any other parameter needed to guide the estimation process is considered a hyperparameter (such as an imputer's strategy), and it must be set as an instance variable (generally via a constructor parameter).

​	*Transformers*

​		Some estimators (such as an imputer) can also transform a dataset; these are called *transformers*. Once again, the API is simple: the transformation is performed by the `transform()` method with the dataset to transform as a parameter. It returns the transformed dataset. This transformation generally relies on the learned parameter, as is the case for an `imputer`. All transformers also have a convenience method called `fit_transform()` that is equivalent to calling `fit()` and then `transform()` (but sometimes `fit_transform()` is optimized and runs much faster).

​	*Predictors*

​		Finally, some estimators, given a dataset, are capable of making predictions: they are called *predictors*. For example, the `LinearRegression` model in the previous chapter was a predictor: given a country's GDP per capita, it predicted life satisfaction. A predictor has a `predict()` method that takes a dataset of new instances and returns a dataset of corresponding predictions. It also has a `score()` method that measures the quality of the predictions, given a test set (and the corresponding labels, in the case of supervised learning algorithms)



*Inspection*

​	All the estimator's hyperparameters are accessible directly via public instance variables (e.g., `imputer .strategy`), and all the estimator's learned parameters are accessible via public instance variables with an underscore  suffix (e.g., `imputer.statistics_`)

*Nonproliferation of classes*

​	Datasets are represented as NumPy or SciPy sparse matrices, instead of homemade classes. Hyperparameters are just regular Python strings or numbers.

*Composition*

​	Existing building blocks are reused as much as possible. For example, it is easy to create a `Pipeline` estimator from an arbitrary sequence of transformers followed by a final estimator, as we will see.

*Sensible defaults*

​	Scikit-Learn provides reasonable default values for most parameters, making it easy to quickly create a baseline working system.



### For Categorical Data with a large number of possible categories:

- try to replace them with useful numerical features related to the categories
- or try to replace each category with a learnable, low-dimensional vector called *embedding*



### See p68 for how to create a custom transformer



### Feature Scaling

with few exceptions, Machine Learning algorithms **don't preform well** when the input numerical attributes have **very different scales**.

#### Two Scaling Methods

- Mix-Max Scaling:
  - for neural networks, it always expect the input value to range from 0 to 1
- Standardization:

you need to **fit the scaler with train method** and use it to transform all data



### Dealing with Underfitting

- A more powerful model
- feed the algorithm with better features
- reduce the constraints on the model



### Dealing with Overfitting

- Simply the model
- constrain the model (regularize it)
- get a lot more training data



### Launch

need to get your solution ready for production (e.g., polish the code, write documentation and tests, and so on)

#### For maintain and Retrain

you need to automate the training process as much as you can to save time:

- Collect fresh data regularly and label it (e.g., using human raters)
- Write a script to train the model and fine-tune the hyperparameters automatically. This script could run automatically, for example every day or every week, depending on your needs
- Write another script that will evaluate both the new model and the previous model on the updated test set, and deploy model to production if the performance has not decreased (if it did, make sure you investigate why)



## Small Notes

<u>ensemble learning</u>: building a model on top of many other models

#### How to save a model?

```python
import joblib

joblib.dump(my_model, "my_model.pkl")
# and later
my_model_loaded = joblib.load("my_model.pkl")
```



#### To Analyze Attributes

If you want to analyze the importance of attributes, it might be a good idea to maintain a list recording the sequence of attributes with names. 

Example can be found in p79 ORML



### Generalization toward Unknown Data is a Key Point for Data Science Project

Never use test set to tweak the hyperparameters: they are meant to be unknown!



### ML involves quite a lot of infrastructure!
