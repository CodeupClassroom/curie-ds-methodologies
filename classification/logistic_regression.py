import itertools as it
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
plt.style.use('ggplot')
plt.ion()

df = sns.load_dataset('titanic')
df = df[['survived', 'pclass', 'age', 'fare', 'sex']].dropna()
df.shape



model = smf.logit('survived ~ pclass + age + fare + sex', df).fit()

model.summary()

df['prediction'] = model.predict(df)

# Visualizing the predicted vs actual values
step = .05
bins = np.arange(0, 1 + step, step)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.hist(df.query('survived == 1').prediction, color='orange', bins=bins)
ax1.set(title='Survived')
ax2.hist(df.query('survived == 0').prediction, bins=bins)
ax2.set(xlabel='P(survived)', title='Died')

# Create a synthetic dataframe of X values to illustrate how the predictions are
# working.
df = (
    pd.DataFrame(
        it.product([1, 2, 3], [15, 25, 35, 45], [50], ["male", "female"]),
        columns=["pclass", "age", "fare", "sex"],
    )
    .assign(prediction=lambda df: model.predict(df))
    .assign(pclass=lambda df: df.pclass.map({1: "first", 2: "second", 3: "third"}))
    .drop(columns="fare")
)

df.sort_values(by=["pclass", "age"])

# the difference changing sex from male to female makes in the predicted
# probability, holding pclass and age constant
(
    df.groupby(["pclass", "age"]).prediction.apply(
        lambda df: df.diff().abs().dropna().values[0]
    )
    .rename('delta p(survived) for 1 change in sex')
    .reset_index()
)

df.sort_values(by=['pclass', 'sex']).assign(
    delta_prediction=lambda df: df.prediction.diff()
)

sns.lineplot(data=df, y='prediction', style='pclass', hue='sex', x='age')

sns.relplot(data=sns.load_dataset('titanic'), y='age', x='fare')

# ------------------------------------------------------------

# import sys
# from io import StringIO
# class SwallowOutput:
#     def __init__(self):
#         self._out = StringIO()
#         self._err = StringIO()
#     def __enter__(self):
#         sys.stdout = self._out
#         sys.stderr = self._err
#     def __exit__(self, *args):
#         sys.stdout = sys.__stdout__
#         sys.stderr = sys.__stderr__
#         self._out.seek(0)
#         self._err.seek(0)
#         self.stdout = self._out.read()
#         self.stderr = self._err.read()
# o = SwallowOutput()
# with o:
#     exec('print(1)')
# o.stdout
# o.stderr

# ------------------------------------------------------------

import sklearn.metrics

iris = sns.load_dataset('iris')

model = smf.mnlogit('species ~ sepal_length + petal_length + petal_width', iris).fit()

model.summary()

iris['predicted'] = (
    model.predict(iris)
    .set_axis(sorted(iris.species.unique()), axis=1)
    .idxmax(axis=1)
)

print(sklearn.metrics.classification_report(iris.species, iris.predicted))

# ------------------------------------------------------------

#: # Logistic Regression
#:
#: 0. Validate Split
#: 1. What is logistic regression?
#:
#:    * :math:`1 / 1 + e^{-\sum{\beta_ix_i}}`
#:    * putting our ols formula into a transformation such that the outcome is a
#:      number between 0 and 1
#:    * Pros: fast to train and predict, probabilities, more interpretable than
#:      some models
#:    * Cons: not as interpretable as, e.g. dtrees; assumes feature independence
#:    * Good choice for a baseline to compare other models against
#:
#: 2. Analyzing our model
#: 3. Choosing the best threshold

#: ## Simple Example
#:
#: Using just one predictor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
plt.ion()

np.random.seed(123)
df = pd.DataFrame({'macbook': np.random.choice([0, 1], 40)})
df['coolness'] = np.where(
    df.macbook == 1,
    np.random.normal(80, 10, 40),
    np.random.normal(40, 10, 40),
)
plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8.5))
df.query('macbook == 1').coolness.plot.hist(ax=ax1, bins=15, alpha=.8, title='macbook')
df.query('macbook == 0').coolness.plot.hist(ax=ax2, bins=15, alpha=.8, title='no macbook')
ax2.set(xlabel='coolness')
fig.tight_layout()

model.summary()

model = smf.logit('macbook ~ coolness', df).fit()
df['prediction'] = model.predict(df)
ax = df.set_index('coolness').sort_index().prediction.plot()
df.plot.scatter(y='macbook', x='coolness', ax=ax, label='Actual')
ax.set(ylabel='P(macbook)')
plt.tight_layout()

#: We define a **threshold** for predicting the positive class. If the predicted
#: probability is above the threshold, then we predict 1, else 0.
#:
#: - default .5
#: - threshold == 1, always predict negative (perfect precision)
#: - threshold == 0, always predict positive (perfect recall)
#:
#: ## Mini Exercise
#:
#: 1. Load the titanic dataset that you've put together from previous lessons.
#: 2. Split your data into train and test datasets. Further split your training
#:    data into train and validate sets.
#: 3. Fit a logistic regression model on your training data using sklearn's
#:    linear_model.LogisticRegression class. Use fare and pclass as the
#:    predictors.
#: 4. Use the model's `.predict` method. What is the output?
#: 5. Use the model's `.predict_proba` method. What is the output? Why do you
#:    think it is shaped like this?
#: 6. Evaluate your model's predictions on the validate data set. How accurate
#:    is the mode? How does changing the threshold affect this?
#:
#: ## More Complex Example

df = sns.load_dataset('titanic')[['survived', 'age', 'pclass', 'sex']].dropna()
train, test = train_test_split(df, random_state=14, train_size=.85)
train, validate = train_test_split(train, random_state=14, train_size=.85)
print('   train: %d rows x %d columns' % train.shape)
print('validate: %d rows x %d columns' % validate.shape)
print('   test: %d rows x %d columns' % test.shape)

model = smf.logit('survived ~ age + pclass + sex', train).fit()

model.summary()

dataset = validate
t = .5
probs = model.predict(dataset)
y = dataset.survived
yhat = (probs > t).astype(int)

precision_score(y, yhat, average=None)
recall_score(y, yhat, average=None)
accuracy_score(y, yhat)

from importlib import reload
import logistic_regression_util
reload(logistic_regression_util)

logistic_regression_util.plot_true_by_probs(y, probs)

logistic_regression_util.plot_true_by_probs(y, probs, subplots=True)

plt.style.use('ggplot')
logistic_regression_util.plot_metrics_by_thresholds(y, probs)

import sklearn.linear_model
y, X = df.survived, df[['age', 'pclass']]
model = sklearn.linear_model.LogisticRegression().fit(X, y)
model.predict(X)
model.predict_proba(X).sum(axis=1)

#: ## Exercise
#:
#: In this exercise, we'll continue working with the titanic dataset and
#: building logistic regression models. Throughout this exercise, be sure you
#: are training and comparing models on the train and validate datasets. The
#: test dataset should only be used for your final model.
#:
#: For all of the models you create, choose a threshold that optimizes for
#: accuracy.
#:
#: 1. Create another model that includes age in addition to fare and pclass.
#:    Does this model perform better than your previous one?
#: 2. Include sex in your model as well. Note that you'll need to encode this
#:    feature before including it in a model.
#: 3. Try out other combinations of features and models.
#: 4. Choose you best model and evaluate it on the test dataset. Is it overfit?
#: 5. **Bonus** How do different strategies for handling the missing values in
#:    the age column affect model performance?
#: 6. **Bonus**: How do different strategies for encoding sex affect model
#:    performance?
#: 7. **Bonus**: `scikit-learn`'s `LogisticRegression` classifier is actually
#:    applying [a regularization penalty to the coefficients][1] by default.
#:    This penalty causes the magnitude of the coefficients in the resulting
#:    model to be smaller than they otherwise would be. This value can be
#:    modified with the `C` hyper parameter. Small values of `C` correspond to
#:    a larger penalty, and large values of `C` correspond to a smaller penalty.
#:
#:     Try out the following values for `C` and note how the coefficients and
#:     the model's performance on both the dataset it was trained on and on the
#:     validate split are affected.
#:
#: [1]: https://en.wikipedia.org/wiki/Regularized_least_squares
