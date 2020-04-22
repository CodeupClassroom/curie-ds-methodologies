#: # K Nearest Neghbors
#:
#: Make predictions based off of the closes observations.
#:
#: - distance metric
#:
#: Pros:
#:
#: - Fast to train
#: - Intuitive
#: - can pick up on arbitrary patterns (unline logit or dtrees)
#: - one assumption: closer points are more similar
#:
#: Cons:
#:
#: - k is unknown
#: - Model parameter is the entire training dataset
#: - Prediction can be expensive
#: - Because distance is used, scaling is important
#:
#: ## Plan
#:
#: 1. Demo KNN in a single dimension
#: 2. Demo in 2 dimensions
#: 3. See the need for scaling
#: 4. Compare model performance on unscaled vs scaled data
#: 5. How to choose `k` ? Visualize model performance
#:
#: ## Demo

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.ion()

#: Generate the fake data and split it
np.random.seed(123)
df = pd.DataFrame(
    {
        "flavor": np.random.choice(["pistachio", "strawberry", "chocolate"], 100),
        "pints": np.random.normal(10, 1, 100),
        "n_sprinkles": np.random.normal(1000, 100, 100).round(),
    }
)
df.pints = np.where(df.flavor == "pistachio", np.random.normal(15, 1, 100), df.pints)
df.pints = np.where(df.flavor == "strawberry", np.random.normal(5, 1, 100), df.pints)
df.n_sprinkles = np.where(
    df.flavor == "pistachio", np.random.normal(1080, 100, 100), df.n_sprinkles
)
df.n_sprinkles = np.where(
    df.flavor == "strawberry", np.random.normal(920, 100, 100), df.n_sprinkles
)
colors = {"pistachio": "darkgreen", "strawberry": "red", "chocolate": "brown"}
train, test = train_test_split(df, train_size=0.9)

#: Visualize KNN with just pints
def plot_pints(plot_test=False):
    for flavor, pints in train.groupby("flavor").pints:
        plt.scatter(pints, [0] * pints.size, label=flavor, c=colors[flavor], alpha=0.5)
    plt.xlabel("Pints Consumed")
    plt.ylim(-0.1, 0.2)
    plt.yticks([])
    plt.ylabel("")
    plt.gca().spines["left"].set_visible(False)
    if plot_test:
        plt.scatter(
            test.pints, [0.05] * test.pints.size, c="black", marker="x", label="???"
        )
    plt.legend()
plot_pints(True)

#: Visualize KNN with pints and sprinkles, demo why scaling matters
def plot_scatter(plot_test=False, same_xy_scale=False):
    for flavor, subset in train.groupby("flavor"):
        plt.scatter(
            subset.pints, subset.n_sprinkles, label=flavor, c=colors[flavor], alpha=0.5
        )
    plt.xlabel("Pints Consumed")
    plt.ylabel("# Sprinkles")
    if plot_test:
        plt.scatter(
            test.pints, test.n_sprinkles, c="black", marker="x", label="???", s=128
        )
    plt.legend()
    if same_xy_scale:
        plt.xlim(-300, 300)
plot_scatter(True, True)

#: Performance on unscaled data -- not great
X_train, y_train = train.drop(columns='flavor'), train.flavor
X_test, y_test = test.drop(columns='flavor'), test.flavor
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

#: Performance on scaled data -- great!
scaler = MinMaxScaler()
X_train, y_train = train.drop(columns='flavor'), train.flavor
X_test, y_test = test.drop(columns='flavor'), test.flavor
knn = KNeighborsClassifier(n_neighbors=2)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)

#: TODO: improve this example, viz k vs accuracy by train-test
#:
#: Generate the fake data; this time class boundaries aren't so clear
np.random.seed(123)
n = 1000
df = pd.DataFrame(
    {
        "flavor": np.random.choice(["pistachio", "strawberry", "chocolate"], n),
        "pints": np.random.normal(10, 1, n),
        "n_sprinkles": np.random.normal(1000, 100, n).round(),
    }
)
df.pints = np.where(df.flavor == "pistachio", np.random.normal(12, .5, n), df.pints)
df.pints = np.where(df.flavor == "strawberry", np.random.normal(8, .8, n), df.pints)
df.n_sprinkles = np.where(
    df.flavor == "pistachio", np.random.normal(1100, 50, n), df.n_sprinkles
)
df.n_sprinkles = np.where(
    df.flavor == "strawberry", np.random.normal(900, 50, n), df.n_sprinkles
)
colors = {"pistachio": "darkgreen", "strawberry": "red", "chocolate": "brown"}
train, test = train_test_split(df)
plot_scatter(True)

#: Performance on scaled data -- great!
scaler = MinMaxScaler()
X_train, y_train = train.drop(columns='flavor'), train.flavor
X_test, y_test = test.drop(columns='flavor'), test.flavor
knn = KNeighborsClassifier(n_neighbors=2)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)

def eval_knn_model(k):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train_scaled, y_train)
    return {
        'k': k,
        'train_accuracy': knn.score(X_train_scaled, y_train),
        'test_accuracy': knn.score(X_test_scaled, y_test),
    }
pd.DataFrame([eval_knn_model(k) for k in range(1, 31)]).set_index('k').plot()
