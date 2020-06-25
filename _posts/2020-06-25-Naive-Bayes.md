---
layout: post
title: Naive Bayes Classifier
gh-repo: jcs-lambda/CS-Unit1-Build
gh-badge: [star, fork, follow]
tags: [lambda, data science, machine learning]
comments: false
---

For my first build project in the Computer Science unit of Lambda School, I chose to implement a Gaussian Naive Bayes classifier.  
The completed project repository is [hosted on Github](https://github.com/jcs-lambda/CS-Unit1-Build).

## Naive Bayes Classification

A [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) is an algorithm based on [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem).
Bayes' Theorom attempts to describe the probability of an event occuring based on prior knowledge of conditions pertaining to that event.
A Naive Bayes classifier is considered 'naive' in that it assumes no dependence between the features in order to reduce computational complexity.

## Implementation

I built my classifier using Scikit-Learn's [BaseEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator)
and [ClassifierMixin](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html#sklearn.base.ClassifierMixin) as base classes
for a familiar starting point.
I used several sklearn [utility functions](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.utils) to handle input validation.
I also used some functions from Python's [math module](https://docs.python.org/3/library/math.html) and [numpy](https://pypi.org/project/numpy).

```python
import math

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
```

The basic class definition contains the public methods I am planning to implement: `fit` and `predict`.

```python
class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self
        
    def predict(self, X):
        predictions = []
        return np.array(predictions)
```

I decided to put all my input validation logic in one function that is called from both the `fit` and `predict` methods.

```python
    def _validate_input(self, X, y=None):
        if y is not None:
            # fitting the model, validate X and y
            return check_X_y(X, y)
        else:
            # predicting, validate X
            check_is_fitted(self, ['num_features_', 'feature_summaries_'])
            X = check_array(X)
            if X.shape[1] != self.num_features_:
                raise(ValueError('unexpected input shape: (x, {X.shape[1]}); must be (x, {self.num_features_})'))
            return X
```

The `fit` method implements two operations: splitting the training data by its class label,
and storing the mean and standard deviation of each feature according to its label.
I stored the information in a dictionary as a simple way to keep the information pertaining to
each class separate. Using the class label as keys in the dictionary allows this implementation
to accept any hashable type as the class label. This way I do not have to do any converting or
tracking of the labels - I can use them as is.

```python
        # create dictionary containing input data separated by class label
        data_by_class = {}
        for i in range(len(X)):
            features = X[i]
            label = y[i]
            if label not in data_by_class:
                # first occurence of label, create empty list in dictionary
                data_by_class[label] = []
            data_by_class[label].append(features)
```

After separating the data by class, I store the mean and standard deviation of each feature as an instance variable.

```python
        self.feature_summaries_ = {}
        for label, features in data_by_class.items():
            self.feature_summaries_[label] = [
                (np.mean(column), np.std(column))
                for column in zip(*features)
            ]
```

Once the training data has been fit, the classifier is ready to make predictions.
Predictions are made in two steps for each row of input data. The first step accumulates the log
probabilites of each of the features, for each class. I used the sum of the log of each probability instead
of the product of the probabilities themselves to avoid possible floating point issues that can arise
when multiplying several values that are expected to be close to zero. This is not an issue for small
datasets, but can become a problem when calculating for a large amount of features.

```python
        for x in X:
            # get cumulative log probabilites for each class for this row
            probabilities = {}
            for label, features in self.feature_summaries_.items():
                probabilities[label] = 0
                for i in range(len(features)):
                    mean, stdev = features[i]
                    probabilities[label] += math.log2(
                        self._liklihood(x[i], mean, stdev)
                    )
```

Once I have the probabilites for each class, I loop though those results and select the class
with the highest probability.

```python
            # find class with highest probability
            best_label, best_prob = None, -1
            for label, probability in probabilities.items():
                if best_label is None or probability > best_prob:
                    best_prob = probability
                    best_label = label

            # prediction for this row
            predictions.append(best_label)
```

Although this classifier works just as well as sklearn's GaussianNB on a small, clean dataset
(I tested on [sklearn's wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine)),
it is not ready for use in a production environment. Dirty data, large data, and who knows what else will break this implementation.

![Results vs sklearn's GaussianNB](https://github.com/jcs-lambda/jcs-lambda.github.io/raw/master/img/gnb_vs_nb.png)

## Resources

Naive Bayes Classification
- https://en.wikipedia.org/wiki/Naive_Bayes_classifier
- https://en.wikipedia.org/wiki/Bayes%27_theorem
- https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
- https://www.geeksforgeeks.org/naive-bayes-classifiers/
- https://alexn.org/blog/2012/02/09/howto-build-naive-bayes-classifier.html
- https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/
- https://dzone.com/articles/naive-bayes-tutorial-naive-bayes-classifier-in-pyt
- https://www.edureka.co/blog/naive-bayes-tutorial/
- https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn

SciKit-Learn
- https://scikit-learn.org/stable/developers/develop.html
- https://scikit-learn.org/stable/developers/utilities.html
- https://scikit-learn.org/stable/modules/classes.html
- https://scikit-learn.org/stable/modules/naive_bayes.html
