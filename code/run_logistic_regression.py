from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os, sys
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import matplotlib.gridspec as gridspec

# user modules
from helpers import *
import clean_data


def logistic_regression_on_df(
    target_df,
    feature_array,
    train_amount,
    motor_rescue_dfs,
    post_jaws_dfs,
    post_npas_dfs,
    jaws_post_times,
    npas_post_times,
    is_medial,
    show=False,
):
    feature_df = target_df.iloc[:, feature_array]

    X_train, X_test, y_train, y_test = clean_data.split_data(
        feature_df, target_df, train_amount, seed=1
    )

    X_train_norm, X_test_norm = clean_data.normalize_data(
        X_train, X_test, min_max=False
    )

    clf = LogisticRegression(
        class_weight="balanced",
        solver="newton-cholesky",
        max_iter=1000,
    )
    clf.fit(X_train_norm, y_train)

    # fig, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    fig = plt.figure(figsize=(12, 8), dpi=300, tight_layout=True)
    gs = gridspec.GridSpec(2, 3)
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:, 0])
    axes = [fig.add_subplot(gs0[0])]
    gs0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:, 1:], hspace=0.3)
    for i in range(4):
        axes.append(fig.add_subplot(gs0[i]))

    # axes = [ax[i, j] for i in range(2) for j in range(2)]
    axes[0].barh(np.arange(len(feature_array)), clf.coef_[0], height=0.8, color="b")
    axes[0].hlines(
        np.arange(0 - 0.5, len(feature_array) + 0.5, 1),
        -np.max(np.abs(clf.coef_[0])),
        np.max(np.abs(clf.coef_[0])),
        color="gray",
        lw=0.5,
        linestyle="dotted",
    )
    axes[0].vlines(
        0, -0.5, len(feature_array) - 0.5, color="k", lw=0.5, linestyle="dashed"
    )
    axes[0].set_yticks(np.arange(len(feature_array)))
    axes[0].set_yticklabels(target_df.columns[feature_array], rotation=0)
    axes[0].set_title(f"-1 favors DD, 1 favors Naive", fontsize=10)
    axes[0].set_ylim([-0.5, len(feature_array) - 0.5])

    # fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300, tight_layout=True)
    i = 0
    axes[1].plot(
        np.arange(-6, 6, 0.01),
        1 / (1 + np.exp(-np.arange(-6, 6, 0.01))),
        c="k",
        lw=0.5,
        ls="dashed",
    )
    axes[1].vlines(
        0,
        0,
        1,
        color="gray",
        ls="dotted",
        lw=0.25,
    )
    for index, row in X_train_norm.iterrows():
        xx = np.dot(clf.coef_[0], X_train_norm.iloc[i, :])
        axes[1].scatter(
            xx,
            1 / (1 + np.exp(-xx)),
            c="b" if y_train.iloc[i] == 0 else "r",
            s=6,
            zorder=10,
        ),
        i += 1

    i = 0
    axes[2].plot(
        np.arange(-6, 6, 0.01),
        1 / (1 + np.exp(-np.arange(-6, 6, 0.01))),
        c="k",
        lw=0.5,
        ls="dashed",
    )
    for index, row in X_test_norm.iterrows():
        xx = np.dot(clf.coef_[0], X_test_norm.iloc[i, :])
        # axes[2].scatter(xx, y_test.iloc[i], c="b" if y_test.iloc[i] == 0 else "r", s=4),
        axes[2].scatter(
            xx,
            1 / (1 + np.exp(-xx)),
            c="b" if y_test.iloc[i] == 0 else "r",
            s=6,
            zorder=10,
        )
        i += 1
    axes[2].vlines(
        0,
        0,
        1,
        color="gray",
        ls="dotted",
        lw=0.25,
    )

    accuracy_train = accuracy_score(y_train, clf.predict(X_train_norm))
    accuracy_test = accuracy_score(y_test, clf.predict(X_test_norm))
    axes[1].set_title(
        f"Training (n={len(X_train_norm)}), Accuracy: {accuracy_train:.02f}",
        fontsize=10,
    )
    axes[2].set_title(
        f"Testing (n={len(X_test_norm)}), Accuracy: {accuracy_test:.02f}", fontsize=10
    )

    cm = confusion_matrix(y_train, clf.predict(X_train_norm), labels=clf.classes_)
    cm = cm / cm.sum()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_train, clf.predict(X_train_norm)
    )
    disp.plot(ax=axes[3], im_kw={"aspect": "auto"})
    axes[3].set_title(
        f"Accuracy: {accuracy_train:.02f}\nPrecision: {precision[1]:.02f}, {precision[0]:.02f}\nRecall: {recall[1]:.02f}, {recall[0]:.02f}",
        fontsize=10,
    )
    axes[3].set_xticks([0, 1])
    axes[3].set_xticklabels(["DD", "Naive"])
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(["DD", "Naive"])

    cm = confusion_matrix(y_test, clf.predict(X_test_norm), labels=clf.classes_)
    cm = cm / cm.sum()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_test, clf.predict(X_test_norm)
    )

    disp.plot(ax=axes[4], im_kw={"aspect": "auto"})
    axes[4].set_title(
        f"Accuracy: {accuracy_test:.02f}\nPrecision: {precision[1]:.02f}, {precision[0]:.02f}\nRecall: {recall[1]:.02f}, {recall[0]:.02f}",
        fontsize=10,
    )
    axes[4].set_xticks([0, 1])
    axes[4].set_xticklabels(["DD", "Naive"])
    axes[4].set_yticks([0, 1])
    axes[4].set_yticklabels(["DD", "Naive"])

    makeNice(axes, labelsize=8)
    fig.savefig(
        "data/logistic_regression/logistic_regression_weights.pdf", bbox_inches="tight"
    )
    plt.close()

    if show:
        run_cmd("open data/logistic_regression/logistic_regression_weights.pdf")

    return [accuracy_train, accuracy_test]
