"""
Utilities for delivering a lesson on logistic regression. Built for binary
classifiers, ymmv for multi-class.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

plt.rc("axes.spines", top=False, right=False)
plt.rc("font", size=13)
plt.rc("figure", figsize=(12, 8.5))
plt.rc("axes", grid=True)
plt.rc("grid", lw=0.8, color="grey", ls=":", alpha=0.7)


def evaluate_threshold(t, y, probs):
    yhat = (probs > t).astype(int)
    return {
        "threshold": t,
        "precision": precision_score(y, yhat),
        "recall": recall_score(y, yhat),
        "accuracy": accuracy_score(y, yhat),
    }


def evaluate_thresholds(y, probs):
    return pd.DataFrame(
        [evaluate_threshold(t, y, probs) for t in np.arange(0, 1.01, 0.01)]
    )


def plot_true_by_probs(y, probs, subplots=False):
    if subplots:
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12, 8.5))
        axs[0].hist(probs[y == 0], bins=25)
        axs[0].set(title="y = 0")
        axs[1].hist(probs[y == 1], bins=25)
        axs[1].set(
            title="y = 1", xlabel="P(y = 1)", xlim=(0, 1), xticks=np.arange(0, 1.1, 0.1)
        )
    else:
        probs.groupby(y).plot.hist(bins=25, alpha=0.6)
        plt.xlabel("P(y = 1)")
        plt.legend(title=y.name)


def plot_metrics_by_thresholds(y, probs, subplots=False):
    evaluation = evaluate_thresholds(y, probs)
    axs = (
        evaluation.query("precision > 0")
        .set_index("threshold")
        .plot(subplots=subplots, sharex=True, sharey=True, figsize=(12, 8.5))
    )
    (axs[-1] if subplots else axs).set_xticks(np.arange(0, 1.05, 0.05))
    plt.tight_layout()


def get_macbook_data():
    np.random.seed(123)
    df = pd.DataFrame({"macbook": np.random.choice([0, 1], 40)})
    df["coolness"] = np.where(
        df.macbook == 1, np.random.normal(80, 10, 40), np.random.normal(40, 10, 40)
    )
    return df
