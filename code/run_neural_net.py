from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import matplotlib.gridspec as gridspec
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import sys
import scipy.stats
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# user modules
import clean_data
from helpers import *


def find_best_params_grid_search(X_train, y_train):
    parameter_space = {
        "hidden_layer_sizes": [(200, 100), (6, 3), (12, 6, 3), (12, 6), (12), (6)],
        "activation": ["tanh", "relu", "identity"],
        "solver": ["sgd", "adam", "lbfgs"],
        "learning_rate": ["constant", "adaptive"],
        "learning_rate_init": [1e-3, 1e-2],
    }

    mlp = MLPClassifier(
        random_state=1,
        max_iter=50000,
        alpha=1e-5,
        max_fun=50000,
    )

    clf = GridSearchCV(mlp, parameter_space, n_jobs=5, cv=3, scoring="accuracy")
    clf.fit(X_train, y_train)

    return clf.best_params_


def run_grid_search(
    target_df,
    feature_array,
    train_amount,
    motor_rescue_dfs,
    show=False,
):

    feature_df = target_df.iloc[:, feature_array]
    use_pre_ephys_training = True
    if use_pre_ephys_training:
        feature_df = pd.concat(
            [
                feature_df,
                motor_rescue_dfs[0].iloc[:, feature_array],
                motor_rescue_dfs[2].iloc[:, feature_array],
            ],
        )
        target_df = pd.concat(
            [
                target_df,
                pd.DataFrame(
                    np.zeros(len(motor_rescue_dfs[0])),
                    columns=["Type"],
                    index=motor_rescue_dfs[0].iloc[:, feature_array].index,
                ),
                pd.DataFrame(
                    np.zeros(len(motor_rescue_dfs[2])),
                    columns=["Type"],
                    index=motor_rescue_dfs[2].iloc[:, feature_array].index,
                ),
            ],
        )

    X_train, X_test, y_train, y_test = clean_data.split_data(
        feature_df, target_df, train_amount
    )

    X_train_norm, X_test_norm = clean_data.normalize_data(
        X_train, X_test, min_max=False
    )

    parameter_space = {
        "hidden_layer_sizes": [
            (10, 10),
            (
                10,
                10,
                10,
            ),
            (50,),
            (50, 50),
            (200, 100),
            (500,),
        ],
        "activation": ["tanh", "relu", "identity"],
        "solver": ["sgd", "adam", "lbfgs"],
        "learning_rate": ["constant", "adaptive"],
        "learning_rate_init": [1e-3, 1e-2],
    }

    mlp = MLPClassifier(
        random_state=1,
        max_iter=30000,
        alpha=1e-5,
        max_fun=30000,
    )

    clf = GridSearchCV(mlp, parameter_space, n_jobs=5, cv=4, scoring="accuracy")
    clf.fit(X_train_norm, y_train)

    print(
        "Best params: ",
        clf.best_params_,
        "Acc Score: ",
        clf.best_score_,
    )

    clf_final = MLPClassifier(
        hidden_layer_sizes=clf.best_params_["hidden_layer_sizes"],
        activation=clf.best_params_["activation"],
        solver=clf.best_params_["solver"],
        learning_rate=clf.best_params_["learning_rate"],
        learning_rate_init=clf.best_params_["learning_rate_init"],
        random_state=1,
        max_iter=30000,
        alpha=1e-5,
        max_fun=30000,
    )
    clf_final.fit(X_train_norm, y_train)
    accuracy_train = accuracy_score(y_train, clf_final.predict(X_train_norm))
    accuracy_test = accuracy_score(y_test, clf_final.predict(X_test_norm))
    print(accuracy_train, accuracy_test)

    """
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    """

    return clf.best_params_

    clf_final = MLPClassifier(
        hidden_layer_sizes=clf.best_params_["hidden_layer_sizes"],
        activation=clf.best_params_["activation"],
        solver=clf.best_params_["solver"],
        learning_rate=clf.best_params_["learning_rate"],
        random_state=1,
        max_iter=30000,
        alpha=1e-5,
        max_fun=30000,
    )
    clf_final.fit(X_train_norm, y_train)
    accuracy_train = accuracy_score(y_train, clf.predict(X_train_norm))
    accuracy_test = accuracy_score(y_test, clf.predict(X_test_norm))
    print(accuracy_train, accuracy_test)
    sys.exit()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300, tight_layout=True)
    axes = [ax[i] for i in range(2)]

    accuracy_train = accuracy_score(y_train, clf.predict(X_train))
    accuracy_test = accuracy_score(y_test, clf.predict(X_test))

    cm = confusion_matrix(y_train, clf.predict(X_train), labels=clf.classes_)
    cm = cm / cm.sum()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_train, clf.predict(X_train)
    )

    disp.plot(ax=axes[0], im_kw={"aspect": "auto"})
    axes[0].set_title(
        f"Accuracy: {accuracy_train:.02f}\nPrecision: {precision[1]:.02f}, {precision[0]:.02f}\nRecall: {recall[1]:.02f}, {recall[0]:.02f}",
        fontsize=10,
    )

    cm = confusion_matrix(y_test, clf.predict(X_test), labels=clf.classes_)
    cm = cm / cm.sum()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_test, clf.predict(X_test)
    )

    disp.plot(ax=axes[1], im_kw={"aspect": "auto"})
    axes[1].set_title(
        f"Accuracy: {accuracy_test:.02f}\nPrecision: {precision[1]:.02f}, {precision[0]:.02f}\nRecall: {recall[1]:.02f}, {recall[0]:.02f}",
        fontsize=10,
    )

    for i in range(2):
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["DD", "Naive"])
        axes[i].set_yticks([0, 1])
        axes[i].set_yticklabels(["DD", "Naive"])

    plt.suptitle(
        f"Naive/DD Train: {len(y_train[y_train==1])}/{len(y_train[y_train==0])}, Naive/DD Test: {len(y_test[y_test==1])}/{len(y_test[y_test==0])}"
    )

    fig.savefig(f"../data/neural_net/grid_search_conf_mat.pdf", bbox_inches="tight")
    plt.close()
    if show:
        run_cmd("open ../data/neural_net/grid_search_conf_mat.pdf")
    return [accuracy_train, accuracy_test]


def feature_importance(
    target_df,
    feature_array,
    train_amount,
    seeds=10,
    show=False,
):
    seeds = np.arange(seeds)
    feature_df = target_df.iloc[:, feature_array]
    train_test_accuracy = np.zeros((len(feature_array) + 1, 2))

    xlabels = list(target_df.columns[feature_array])
    xlabels.insert(0, "All")

    acc_scores = np.zeros((len(feature_array) + 1, len(seeds)))

    feature_count = 1
    for entry in feature_array:
        copy_features = feature_array.copy()
        copy_features.remove(entry)

        tmp_feature_df = pd.concat(
            [
                target_df.iloc[:, copy_features],
            ],
        )

        tmp_target_df = pd.concat(
            [
                target_df,
            ],
        )

        for i in seeds:

            np.random.seed(i)
            X_train, X_test, y_train, _ = clean_data.split_data(
                tmp_feature_df, tmp_target_df, train_amount, seed=i
            )

            X_train_norm, _ = clean_data.normalize_data(X_train, X_test, min_max=False)
            clf = MLPClassifier(
                hidden_layer_sizes=(200, 100),
                activation="relu",
                solver="adam",
                random_state=i,
                max_iter=50000,
                alpha=1e-5,
                max_fun=50000,
                learning_rate_init=1e-2,
                learning_rate="adaptive",
                epsilon=1e-8,
                shuffle=True,
                early_stopping=False,
                verbose=False,
                tol=1e-4,
                n_iter_no_change=10,
            )

            clf.fit(X_train_norm, y_train)
            acc_scores[feature_count, i] = accuracy_score(
                y_train, clf.predict(X_train_norm)
            )
        feature_count += 1

    for i in seeds:

        np.random.seed(i)
        X_train, X_test, y_train, _ = clean_data.split_data(
            feature_df, target_df, train_amount, seed=i
        )

        X_train_norm, _ = clean_data.normalize_data(X_train, X_test, min_max=False)

        clf = MLPClassifier(
            hidden_layer_sizes=(200, 100),
            activation="relu",
            solver="adam",
            random_state=i,
            max_iter=50000,
            alpha=1e-5,
            max_fun=50000,
            learning_rate_init=1e-2,
            learning_rate="adaptive",
            epsilon=1e-8,
            shuffle=True,
            early_stopping=False,
            verbose=False,
            tol=1e-4,
            n_iter_no_change=10,
        )

        clf.fit(X_train_norm, y_train)
        acc_scores[0, i] = accuracy_score(y_train, clf.predict(X_train_norm))

    for i in range(1, acc_scores.shape[0]):
        print(
            scipy.stats.ttest_ind(
                acc_scores[0, :], acc_scores[i, :], equal_var=False
            ).pvalue
        )

    clf.fit(X_train_norm, y_train)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300, tight_layout=True)
    ax.errorbar(
        np.arange(train_test_accuracy[:, 0].shape[0]),
        np.mean(acc_scores, axis=1),
        yerr=scipy.stats.sem(acc_scores, axis=1),
        color="k",
        capsize=4,
        marker="o",
    )
    ax.legend()
    ax.set_xticks(np.arange(0, len(feature_array) + 1))
    ax.set_xticklabels(xlabels, rotation=35, fontsize=6)
    ax.grid(visible=True, which="both")
    ax.set_ylabel("Accuracy")
    makeNice(ax, labelsize=6)
    fig.savefig(
        f"../data/neural_net/feature_removal_performance.pdf", bbox_inches="tight"
    )
    plt.close()
    if show:
        run_cmd("open ../data/neural_net/feature_removal_performance.pdf")


def feature_importance_selected(
    target_df,
    feature_array,
    train_amount,
    seeds=10,
    show=False,
):
    seeds = np.arange(seeds)
    feature_df = target_df.iloc[:, feature_array]
    train_test_accuracy = np.zeros((len(feature_array) + 1, 2))

    xlabels = list(target_df.columns[feature_array])
    xlabels.insert(0, "All")

    acc_scores = np.zeros((len(feature_array) + 1, len(seeds)))
    acc_scores_test = np.zeros((len(feature_array) + 1, len(seeds)))
    acc_scores_naive = np.zeros((len(feature_array) + 1, len(seeds)))
    acc_scores_dd = np.zeros((len(feature_array) + 1, len(seeds)))
    probs_scores_naive = np.zeros((len(feature_array) + 1, len(seeds)))
    probs_scores_dd = np.zeros((len(feature_array) + 1, len(seeds)))

    corr_level = 0.8
    df_corr = feature_df.corr()

    feature_removals = {}
    ditches = np.zeros((len(feature_array), len(feature_array)))
    for i in range(len(feature_array)):
        append_vals = []
        for j in range(len(feature_array)):
            if np.abs(df_corr.iloc[i, j]) >= corr_level and i != j:
                append_vals.append(j)
                ditches[i, j] = 1
        ditches[i, i] = 1
        if i == 70:
            ditches[i, 2] = 1
            append_vals.append(2)
        if i == 60:
            ditches[i, 1] = 1
            append_vals.append(1)

        feature_removals[feature_array[i]] = append_vals

    feature_count = 1
    for entry in feature_array:
        copy_features = feature_array.copy()
        removals = [entry]
        if len(feature_removals[entry]) > 0:
            removals.extend(feature_removals[entry])
        copy_features.remove(entry)
        copy_features = [x for x in copy_features if x not in removals]

        tmp_feature_df = pd.concat(
            [
                target_df.iloc[:, copy_features],  # target_df.iloc[:, [entry]],  #
            ],
        )

        tmp_target_df = pd.concat(
            [
                target_df,
            ],
        )

        for i in seeds:
            np.random.seed(i)
            X_train, X_test, y_train, y_test = clean_data.split_data(
                tmp_feature_df, tmp_target_df, train_amount, seed=i
            )

            X_train_norm, X_test_norm = clean_data.normalize_data(
                X_train, X_test, min_max=False
            )
            clf = MLPClassifier(
                hidden_layer_sizes=(200, 100),
                activation="relu",
                solver="adam",
                random_state=i,
                max_iter=50000,
                alpha=1e-5,
                max_fun=50000,
                learning_rate_init=1e-2,
                learning_rate="adaptive",
                epsilon=1e-8,
                shuffle=True,
                early_stopping=False,
                verbose=False,
                tol=1e-4,
                n_iter_no_change=10,
            )

            clf.fit(X_train_norm, y_train)
            acc_scores[feature_count, i] = accuracy_score(
                y_train, clf.predict(X_train_norm)
            )
            acc_scores_test[feature_count, i] = accuracy_score(
                y_test, clf.predict(X_test_norm)
            )

            matrix = confusion_matrix(y_test, clf.predict(X_test_norm))

            acc_scores_dd[feature_count, i], acc_scores_naive[feature_count, i] = (
                matrix.diagonal() / matrix.sum(axis=1)
            )

            probs = clf.predict_proba(X_test_norm)

            probs_scores_naive[feature_count, i] = np.mean(probs[y_test == 1, 1])
            probs_scores_dd[feature_count, i] = np.mean(probs[y_test == 0, 0])

        feature_count += 1

    for i in seeds:

        np.random.seed(i)
        X_train, X_test, y_train, y_test = clean_data.split_data(
            feature_df, target_df, train_amount, seed=i
        )

        X_train_norm, X_test_norm = clean_data.normalize_data(
            X_train, X_test, min_max=False
        )

        clf = MLPClassifier(
            hidden_layer_sizes=(200, 100),
            activation="relu",
            solver="adam",
            random_state=i,
            max_iter=50000,
            alpha=1e-5,
            max_fun=50000,
            learning_rate_init=1e-2,
            learning_rate="adaptive",
            epsilon=1e-8,
            shuffle=True,
            early_stopping=False,
            verbose=False,
            tol=1e-4,
            n_iter_no_change=10,
        )

        clf.fit(X_train_norm, y_train)
        acc_scores[0, i] = accuracy_score(y_train, clf.predict(X_train_norm))
        acc_scores_test[0, i] = accuracy_score(y_test, clf.predict(X_test_norm))
        matrix = confusion_matrix(y_test, clf.predict(X_test_norm))

        acc_scores_dd[0, i], acc_scores_naive[0, i] = matrix.diagonal() / matrix.sum(
            axis=1
        )

        probs = clf.predict_proba(X_test_norm)

        probs_scores_naive[0, i] = np.mean(probs[y_test == 1, 1])
        probs_scores_dd[0, i] = np.mean(probs[y_test == 0, 0])

    for i in range(1, acc_scores.shape[0]):

        print(f"Removal of {feature_df.columns[i-1]}:")
        print(
            f"\tTrain p-val:",
            scipy.stats.ttest_ind(
                acc_scores[0, :], acc_scores[i, :], equal_var=False
            ).pvalue,
        )
        print(
            f"\tTest p-val:",
            scipy.stats.ttest_ind(
                acc_scores_test[0, :], acc_scores_test[i, :], equal_var=False
            ).pvalue,
        )
        print()

    indices = list(np.mean(acc_scores_test, axis=1).argsort().astype(int))
    indices.pop(indices.index(0))
    indices.insert(0, 0)
    print(indices)
    probs_scores_naive = probs_scores_naive[indices]
    probs_scores_dd = probs_scores_dd[indices]
    acc_scores_test = acc_scores_test[indices]
    acc_scores = acc_scores[indices]
    ditches_indices = [x - 1 for x in indices[1:]]
    ditches = ditches[ditches_indices]
    xlabels = [xlabels[i] for i in indices]
    print(xlabels)
    # sys.exit()

    corr_mat = feature_df.corr()
    # corr_mat = corr_mat.iloc[ditches_indices, :]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(2) for j in range(2)]
    sns.heatmap(
        corr_mat,  # feature_df.corr(),
        cmap="coolwarm",
        ax=axes[0],
        vmin=-1,
        vmax=1,
        fmt=".2f",
        annot=True,
        annot_kws={"fontsize": 4},
        # mask=np.triu(feature_df.corr()),
        cbar_kws={"shrink": 0.5},
        cbar=False,
    )
    sns.heatmap(ditches, vmin=0, vmax=1, cmap="RdYlGn_r", cbar=False, ax=axes[1])
    """sns.heatmap(
        np.identity(len(feature_array)) == 0,
        vmin=0,
        vmax=1,
        cmap="RdYlGn_r",
        cbar=False,
        ax=axes[1],
    )"""
    axes[1].set_xticks(np.arange(0.5, 12.5))
    axes[1].set_xticklabels(feature_df.columns, rotation=90)
    axes[1].set_yticks(np.arange(0.5, 12.5))
    axes[1].set_yticklabels(xlabels[1:], rotation=0)
    axes[1].set_xlabel("Correlated Feature Removed")
    axes[1].set_ylabel("Feature Removed")

    axes[2].errorbar(
        np.arange(train_test_accuracy[:, 0].shape[0]),
        np.mean(acc_scores, axis=1),
        yerr=scipy.stats.sem(acc_scores, axis=1),
        color="k",
        capsize=4,
        marker="o",
        lw=0.5,
        markersize=4,
        label="Train",
    )
    axes[2].errorbar(
        np.arange(train_test_accuracy[:, 0].shape[0]),
        np.mean(acc_scores_test, axis=1),
        yerr=scipy.stats.sem(acc_scores_test, axis=1),
        color="gray",
        capsize=4,
        marker="o",
        lw=0.5,
        markersize=4,
        label="Test",
    )
    axes[3].errorbar(
        np.arange(train_test_accuracy[:, 0].shape[0]),
        np.mean(probs_scores_naive, axis=1),
        yerr=scipy.stats.sem(probs_scores_naive, axis=1),
        color="r",
        capsize=4,
        marker="o",
        lw=0.5,
        markersize=4,
        label="Naive",
    )
    axes[3].errorbar(
        np.arange(train_test_accuracy[:, 0].shape[0]),
        np.mean(probs_scores_dd, axis=1),
        yerr=scipy.stats.sem(probs_scores_dd, axis=1),
        color="b",
        capsize=4,
        marker="o",
        lw=0.5,
        markersize=4,
        label="DD",
    )
    axes[2].legend()
    axes[2].set_xticks(np.arange(0, len(feature_array) + 1))
    axes[2].set_xticklabels(xlabels, rotation=90, fontsize=6)
    axes[2].grid(visible=True, which="both")
    axes[2].set_ylabel("Accuracy")
    axes[3].legend()
    axes[3].set_xticks(np.arange(0, len(feature_array) + 1))
    axes[3].set_xticklabels(xlabels, rotation=90, fontsize=6)
    axes[3].grid(visible=True, which="both")
    axes[3].set_ylabel("Probability")

    add_fig_labels(axes)
    makeNice(axes, labelsize=6)
    fig.savefig(
        f"../data/neural_net/feature_removal_performance_selected.pdf",
        bbox_inches="tight",
    )
    plt.close()
    if show:
        run_cmd("open ../data/neural_net/feature_removal_performance_selected.pdf")


def feature_removal_probabilities(
    target_df, feature_array, train_amount, seeds=[0, 1, 2, 3, 4], show=False
):
    feature_df = target_df.iloc[:, feature_array]
    features = feature_df.columns

    fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]

    fig2, ax2 = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
    axes2 = [ax2[i, j] for i in range(4) for j in range(3)]

    axis_plot = 0
    for feature in features:
        naive_data = []
        naive_data_removed = []
        dd_data = []
        dd_data_removed = []

        for seed in seeds:
            np.random.seed(seed)
            X_train, X_test, y_train, y_test = clean_data.split_data(
                feature_df, target_df, train_amount, seed=seed
            )

            X_train_norm, X_test_norm = clean_data.normalize_data(
                X_train, X_test, min_max=False
            )

            X_train_removed = X_train_norm.drop(feature, axis=1)
            X_test_removed = X_test_norm.drop(feature, axis=1)

            X_train_removed = X_train_norm.copy()
            X_train_removed[feature] = 0

            X_test_removed = X_test_norm.copy()
            X_test_removed[feature] = 0

            clf = MLPClassifier(
                hidden_layer_sizes=(200, 100),
                activation="relu",
                solver="adam",
                random_state=seed,
                max_iter=50000,
                alpha=1e-5,
                max_fun=50000,
                learning_rate_init=1e-2,
                learning_rate="adaptive",
                epsilon=1e-8,
                shuffle=True,
                early_stopping=False,
                verbose=False,
                tol=1e-4,
                n_iter_no_change=10,
            )

            clf.fit(X_train_norm, y_train)
            # clf_removed.fit(X_train_removed, y_train)

            probs = clf.predict_proba(X_test_norm)
            # probs_removed = clf_removed.predict_proba(X_test_removed)
            probs_removed = clf.predict_proba(X_test_removed)

            naive_data.extend(probs[y_test == 1, 1])
            naive_data_removed.extend(probs_removed[y_test == 1, 1])
            dd_data.extend(probs[y_test == 0, 0])
            dd_data_removed.extend(probs_removed[y_test == 0, 0])

        dd_data = np.asarray(dd_data)
        naive_data = np.asarray(naive_data)
        dd_data_removed = np.asarray(dd_data_removed)
        naive_data_removed = np.asarray(naive_data_removed)

        acc_dd = len(dd_data[dd_data > 0.5]) / len(dd_data)
        acc_removed_dd = len(dd_data_removed[dd_data_removed > 0.5]) / len(
            dd_data_removed
        )

        acc_naive = len(naive_data[naive_data > 0.5]) / len(naive_data)
        acc_removed_naive = len(naive_data_removed[naive_data_removed > 0.5]) / len(
            naive_data_removed
        )

        axes2[axis_plot].scatter(acc_dd, acc_removed_dd, color="k")
        axes2[axis_plot].scatter(acc_naive, acc_removed_naive, color="gray")

        sns.regplot(
            x=dd_data[(dd_data > 0.5) & (dd_data_removed > 0.5)],
            y=dd_data_removed[(dd_data > 0.5) & (dd_data_removed > 0.5)],
            color="b",
            scatter_kws={"s": 2},
            ax=axes[axis_plot],
        )
        sns.regplot(
            x=naive_data[(naive_data > 0.5) & (naive_data_removed > 0.5)],
            y=naive_data_removed[(naive_data > 0.5) & (naive_data_removed > 0.5)],
            color="r",
            scatter_kws={"s": 2},
            ax=axes[axis_plot],
        )
        axes[axis_plot].set_xlabel("All", fontsize=6)
        axes[axis_plot].set_ylabel(f"{feature}", fontsize=6)
        axes[axis_plot].plot([0.5, 1], [0.5, 1], color="k", lw=0.5, ls="dashed")
        axes[axis_plot].set_xlim([0.5, 1])
        axes[axis_plot].set_ylim([0.5, 1])
        xlims = axes2[axis_plot].get_xlim()
        ylims = axes2[axis_plot].get_ylim()
        axes2[axis_plot].plot(
            [np.min([xlims[0], ylims[0]]), np.max([xlims[1], ylims[1]])],
            [np.min([xlims[0], ylims[0]]), np.max([xlims[1], ylims[1]])],
            color="k",
            ls="dashed",
            lw=0.5,
        )
        axes2[axis_plot].set_xlim(
            [np.min([xlims[0], ylims[0]]), np.max([xlims[1], ylims[1]])]
        )
        axes2[axis_plot].set_ylim(
            [np.min([xlims[0], ylims[0]]), np.max([xlims[1], ylims[1]])]
        )
        axis_plot += 1
    makeNice(axes, labelsize=6)
    fig.savefig(f"../data/feature_removal_probabilities.pdf", bbox_inches="tight")

    makeNice(axes2, labelsize=6)
    fig2.savefig(
        f"../data/feature_removal_probabilities_summary.pdf", bbox_inches="tight"
    )

    plt.close()
    if show:
        run_cmd(f"open ../data/feature_removal_probabilities_summary.pdf")
        run_cmd(f"open ../data/feature_removal_probabilities.pdf")


def predict_motor_rescue(
    target_df,
    feature_array,
    train_amount,
    motor_rescue_dfs,
    post_jaws_dfs,
    post_npas_dfs,
    jaws_post_times,
    npas_post_times,
    is_medial,
    seeds=[0, 1, 2, 3, 4],
    show=False,
):
    feature_df = target_df.iloc[:, feature_array]

    jaws_tracked_probs = np.zeros((23, 76, 2))
    npas_tracked_probs = np.zeros((10, 76, 2))

    x_train_probs = np.zeros(
        (
            810,
            2,
        )
    )

    pre_prediction_prob = [
        np.zeros((len(motor_rescue_dfs[0]), 2)),
        np.zeros((len(motor_rescue_dfs[2]), 2)),
        np.zeros((len(post_jaws_dfs[0]), 2)),
        np.zeros((len(post_jaws_dfs[1]), 2)),
        np.zeros((len(post_jaws_dfs[2]), 2)),
        np.zeros((len(post_jaws_dfs[3]), 2)),
        np.zeros((len(post_npas_dfs[0]), 2)),
        np.zeros((len(post_npas_dfs[1]), 2)),
        np.zeros((len(post_npas_dfs[2]), 2)),
        np.zeros((len(post_npas_dfs[3]), 2)),
    ]

    jaws_prediction_prob = [[] for x in range(len(post_jaws_dfs) + 1)]
    npas_prediction_prob = [[] for x in range(len(post_npas_dfs) + 1)]

    jaws_medial_prediction_prob = [[] for x in range(len(post_jaws_dfs) + 1)]
    npas_medial_prediction_prob = [[] for x in range(len(post_npas_dfs) + 1)]

    jaws_non_medial_prediction_prob = [[] for x in range(len(post_jaws_dfs) + 1)]
    npas_non_medial_prediction_prob = [[] for x in range(len(post_npas_dfs) + 1)]

    post_jaws_medial_dfs = [x[x["Medial"] == 1] for x in post_jaws_dfs]
    post_npas_medial_dfs = [x[x["Medial"] == 1] for x in post_npas_dfs]

    post_jaws_non_medial_dfs = [x[x["Medial"] == 0] for x in post_jaws_dfs]
    post_npas_non_medial_dfs = [x[x["Medial"] == 0] for x in post_npas_dfs]

    post_jaws_dfs = [
        x.drop(columns=["Post-Time", "Medial", "mouse", "folder", "name"])
        for x in post_jaws_dfs
    ]
    post_npas_dfs = [
        x.drop(columns=["Post-Time", "Medial", "mouse", "folder", "name"])
        for x in post_npas_dfs
    ]

    post_jaws_medial_dfs = [
        x.drop(columns=["Post-Time", "Medial", "mouse", "folder", "name"])
        for x in post_jaws_medial_dfs
    ]
    post_npas_medial_dfs = [
        x.drop(columns=["Post-Time", "Medial", "mouse", "folder", "name"])
        for x in post_npas_medial_dfs
    ]

    post_jaws_non_medial_dfs = [
        x.drop(columns=["Post-Time", "Medial", "mouse", "folder", "name"])
        for x in post_jaws_non_medial_dfs
    ]
    post_npas_non_medial_dfs = [
        x.drop(columns=["Post-Time", "Medial", "mouse", "folder", "name"])
        for x in post_npas_non_medial_dfs
    ]

    accuracy_runs = np.zeros((len(seeds), 2))  # train test
    precision = np.zeros((len(seeds), 4))  # train dd naive test dd naive
    recall = np.zeros((len(seeds), 4))  # train dd naive test dd naive

    motor_predictions_percent = np.zeros(
        (len(seeds), 4)
    )  # jaws pre post, npas pre post
    motor_predictions_post_times_percent = np.zeros(
        (len(seeds), (len(post_jaws_dfs) + 1) * 2)
    )
    medial_predictions_percent = np.zeros(
        (len(seeds), (len(post_jaws_dfs) + 1) * 2)
    )  # jaws pre post, npas pre post

    non_medial_predictions_percent = np.zeros(
        (len(seeds), (len(post_jaws_dfs) + 1) * 2)
    )  # jaws pre post, npas pre post

    motor_predictions_amount = np.zeros((len(seeds), 4))  # jaws pre post, npas pre post
    motor_predictions_post_times_amount = np.zeros(
        (len(seeds), (len(post_jaws_dfs) + 1) * 4)
    )
    medial_predictions_amount = np.zeros(
        (len(seeds), (len(post_jaws_dfs) + 1) * 4)
    )  # jaws pre post, npas pre post

    non_medial_predictions_amount = np.zeros(
        (len(seeds), (len(post_jaws_dfs) + 1) * 4)
    )  # jaws pre post, npas pre post

    cm_train = np.zeros((2, 2))
    cm_test = np.zeros((2, 2))

    predict_jaws_all = [np.zeros((len(x), 2)) for x in post_jaws_dfs]
    predict_jaws_all.insert(0, np.zeros((len(motor_rescue_dfs[0]), 2)))

    predict_npas_all = [np.zeros((len(x), 2)) for x in post_npas_dfs]
    predict_npas_all.insert(0, np.zeros((len(motor_rescue_dfs[2]), 2)))

    run_count = 0
    for seed in seeds:
        np.random.seed(seed)
        X_train, X_test, y_train, y_test = clean_data.split_data(
            feature_df, target_df, train_amount, seed=seed
        )

        X_train_norm, X_test_norm = clean_data.normalize_data(
            X_train, X_test, min_max=False
        )

        motor_feature_dfs = [x.iloc[:, feature_array] for x in motor_rescue_dfs]
        norm_motor_dfs = [
            clean_data.normalize_data(X_train, x, min_max=False)[1]
            for x in motor_feature_dfs
        ]

        norm_jaws_post_dfs = [
            clean_data.normalize_data(X_train, x, min_max=False)[1]
            for x in post_jaws_dfs
        ]

        norm_npas_post_dfs = [
            clean_data.normalize_data(X_train, x, min_max=False)[1]
            for x in post_npas_dfs
        ]

        norm_jaws_medial_post_dfs = [
            clean_data.normalize_data(X_train, x, min_max=False)[1] if len(x) > 0 else x
            for x in post_jaws_medial_dfs
        ]

        norm_npas_medial_post_dfs = [
            clean_data.normalize_data(X_train, x, min_max=False)[1] if len(x) > 0 else x
            for x in post_npas_medial_dfs
        ]

        norm_jaws_non_medial_post_dfs = [
            clean_data.normalize_data(X_train, x, min_max=False)[1] if len(x) > 0 else x
            for x in post_jaws_non_medial_dfs
        ]

        norm_npas_non_medial_post_dfs = [
            clean_data.normalize_data(X_train, x, min_max=False)[1] if len(x) > 0 else x
            for x in post_npas_non_medial_dfs
        ]

        clf = MLPClassifier(
            hidden_layer_sizes=(200, 100),
            activation="relu",
            solver="adam",
            random_state=seed,
            max_iter=50000,
            alpha=1e-5,
            max_fun=50000,
            learning_rate_init=1e-2,
            learning_rate="adaptive",
            epsilon=1e-8,
            shuffle=True,
            early_stopping=False,
            verbose=False,
            tol=1e-4,
            n_iter_no_change=10,
        )

        grid_search = False
        if grid_search:
            best_params = find_best_params_grid_search(X_train_norm, y_train)
            clf = MLPClassifier(
                hidden_layer_sizes=best_params["hidden_layer_sizes"],
                activation=best_params["activation"],
                solver=best_params["solver"],
                learning_rate=best_params["learning_rate"],
                random_state=seed,
                max_iter=50000,
                learning_rate_init=best_params["learning_rate_init"],
                alpha=1e-5,
                max_fun=50000,
                epsilon=1e-8,
                shuffle=True,
                early_stopping=False,
                verbose=False,
            )

        scores = cross_val_score(clf, X_train_norm, y_train, cv=4)
        print("CV: ", scores.mean(), scores.std())
        clf.fit(X_train_norm, y_train)

        save_x_train = X_train.copy()
        save_x_train["Type"] = y_train
        save_x_train.to_csv(f"../data/neural_net/x_train_values_{seed}.csv")

        np.savetxt(
            f"../data/neural_net/x_train_probs_seed_{seed}.txt",
            clf.predict_proba(X_train_norm),
            newline="\n",
            delimiter=" ",
            fmt="%f",
        )

        """# Calculate permutation importance
        result = permutation_importance(
            clf, X_train_norm, y_train, n_repeats=300, random_state=42, n_jobs=5
        )
        print(result)

        # Plot feature importance
        feature_names = np.array([f"Feature {i}" for i in range(X_train.shape[1])])
        sorted_idx = result.importances_mean.argsort()

        plt.figure(figsize=(10, 6))
        plt.boxplot(
            result.importances[sorted_idx].T,
            vert=False,
            labels=X_train.columns[sorted_idx],
        )
        plt.title("Permutation Importance of Each Feature")
        """
        # Explain the model predictions using SHAP
        """explainer = shap.KernelExplainer(clf.predict_proba, X_train_norm)
        shap_values = explainer.shap_values(X_test_norm, nsamples=100)

        # Plot the SHAP summary plot
        shap.summary_plot(shap_values, X_test_norm)"""

        predicts_train = clf.predict(X_train_norm)
        predicts_test = clf.predict(X_test_norm)
        accuracy_train = accuracy_score(y_train, predicts_train)
        accuracy_test = accuracy_score(y_test, predicts_test)

        predictions = [clf.predict(x) for x in norm_motor_dfs]
        predictions_percent = [
            np.sum(clf.predict(x)) / x.shape[0] for x in norm_motor_dfs
        ]

        ### predict tracked cells
        """for i in range(jaws_tracked_probs.shape[0]):
            jaws_tracked_cell = pd.read_csv(
                f"../data/tracked_units/jaws_treatment_../data/jaws_treatment_cell_{i+1}.csv"
            )
            jaws_tracked_cell = jaws_tracked_cell.iloc[:, 1:]
            jaws_tracked_cell.fillna(0, inplace=True)

            norm_jaws_tracked_cell = clean_data.normalize_data(
                X_train, jaws_tracked_cell, min_max=False
            )[1]
            jaws_tracked_probs[i, :, :] += clf.predict_proba(
                norm_jaws_tracked_cell
            ) / len(seeds)
        for i in range(npas_tracked_probs.shape[0]):
            npas_tracked_cell = pd.read_csv(
                f"../data/tracked_units/npas_treatment_../data/npas_treatment_cell_{i+1}.csv"
            )
            npas_tracked_cell = npas_tracked_cell.iloc[:, 1:]
            npas_tracked_cell.fillna(0, inplace=True)
            norm_npas_tracked_cell = clean_data.normalize_data(
                X_train, npas_tracked_cell, min_max=False
            )[1]
            npas_tracked_probs[i, :, :] += clf.predict_proba(
                norm_npas_tracked_cell
            ) / len(seeds)"""

        ### end predict tracked cells

        # determine probabilities
        plot_thres_probs = 0.0
        predict_jaws_pre = clf.predict_proba(norm_motor_dfs[0])
        predict_jaws_30 = clf.predict_proba(norm_jaws_post_dfs[0])
        predict_jaws_60 = clf.predict_proba(norm_jaws_post_dfs[1])
        predict_jaws_90_120 = clf.predict_proba(norm_jaws_post_dfs[2])
        predict_jaws_150_180_210 = clf.predict_proba(norm_jaws_post_dfs[3])

        predict_jaws_all[0] += predict_jaws_pre / len(seeds)
        predict_jaws_all[1] += predict_jaws_30 / len(seeds)
        predict_jaws_all[2] += predict_jaws_60 / len(seeds)
        predict_jaws_all[3] += predict_jaws_90_120 / len(seeds)
        predict_jaws_all[4] += predict_jaws_150_180_210 / len(seeds)

        predict_npas_pre = clf.predict_proba(norm_motor_dfs[2])
        predict_npas_30 = clf.predict_proba(norm_npas_post_dfs[0])
        predict_npas_60 = clf.predict_proba(norm_npas_post_dfs[1])
        predict_npas_90_120 = clf.predict_proba(norm_npas_post_dfs[2])
        predict_npas_150_180_210 = clf.predict_proba(norm_npas_post_dfs[3])

        predict_npas_all[0] += predict_npas_pre / len(seeds)
        predict_npas_all[1] += predict_npas_30 / len(seeds)
        predict_npas_all[2] += predict_npas_60 / len(seeds)
        predict_npas_all[3] += predict_npas_90_120 / len(seeds)
        predict_npas_all[4] += predict_npas_150_180_210 / len(seeds)

        # get naive probabilities of all seeds for Jaws
        jaws_prediction_prob[0].extend(
            predict_jaws_pre[predict_jaws_pre[:, 1] > plot_thres_probs, 1]
        )
        jaws_prediction_prob[1].extend(
            predict_jaws_30[predict_jaws_30[:, 1] > plot_thres_probs, 1]
        )
        jaws_prediction_prob[2].extend(
            predict_jaws_60[predict_jaws_60[:, 1] > plot_thres_probs, 1]
        )
        jaws_prediction_prob[3].extend(
            predict_jaws_90_120[predict_jaws_90_120[:, 1] > plot_thres_probs, 1]
        )
        jaws_prediction_prob[4].extend(
            predict_jaws_150_180_210[
                predict_jaws_150_180_210[:, 1] > plot_thres_probs, 1
            ]
        )

        # get naive probabilities of all seeds for Npas
        npas_prediction_prob[0].extend(
            predict_npas_pre[predict_npas_pre[:, 1] > plot_thres_probs, 1]
        )
        npas_prediction_prob[1].extend(
            predict_npas_30[predict_npas_30[:, 1] > plot_thres_probs, 1]
        )
        npas_prediction_prob[2].extend(
            predict_npas_60[predict_npas_60[:, 1] > plot_thres_probs, 1]
        )
        npas_prediction_prob[3].extend(
            predict_npas_90_120[predict_npas_90_120[:, 1] > plot_thres_probs, 1]
        )
        npas_prediction_prob[4].extend(
            predict_npas_150_180_210[
                predict_npas_150_180_210[:, 1] > plot_thres_probs, 1
            ]
        )

        # get naive probabilites of all seeds for medial Jaws
        predict_jaws_pre_medial = clf.predict_proba(
            norm_motor_dfs[0][is_medial[0] == 1]
        )
        if len(norm_jaws_medial_post_dfs[0] > 0):
            predict_jaws_30_medial = clf.predict_proba(norm_jaws_medial_post_dfs[0])
        predict_jaws_60_medial = clf.predict_proba(norm_jaws_medial_post_dfs[1])
        predict_jaws_90_120_medial = clf.predict_proba(norm_jaws_medial_post_dfs[2])
        predict_jaws_150_180_210_medial = clf.predict_proba(
            norm_jaws_medial_post_dfs[3]
        )

        jaws_medial_prediction_prob[0].extend(
            predict_jaws_pre_medial[predict_jaws_pre_medial[:, 1] > plot_thres_probs, 1]
        )

        if len(norm_jaws_medial_post_dfs[0]) > 0:
            jaws_medial_prediction_prob[1].extend(
                predict_jaws_30_medial[
                    predict_jaws_30_medial[:, 1] > plot_thres_probs, 1
                ]
            )
        jaws_medial_prediction_prob[2].extend(
            predict_jaws_60_medial[predict_jaws_60_medial[:, 1] > plot_thres_probs, 1]
        )
        jaws_medial_prediction_prob[3].extend(
            predict_jaws_90_120_medial[
                predict_jaws_90_120_medial[:, 1] > plot_thres_probs, 1
            ]
        )
        jaws_medial_prediction_prob[4].extend(
            predict_jaws_150_180_210_medial[
                predict_jaws_150_180_210_medial[:, 1] > plot_thres_probs, 1
            ]
        )

        # get naive probabilites of all seeds for medial Npas
        predict_npas_pre_medial = clf.predict_proba(
            norm_motor_dfs[2][is_medial[2] == 1]
        )
        predict_npas_30_medial = clf.predict_proba(norm_npas_medial_post_dfs[0])
        predict_npas_60_medial = clf.predict_proba(norm_npas_medial_post_dfs[1])
        predict_npas_90_120_medial = clf.predict_proba(norm_npas_medial_post_dfs[2])
        predict_npas_150_180_210_medial = clf.predict_proba(
            norm_npas_medial_post_dfs[3]
        )

        npas_medial_prediction_prob[0].extend(
            predict_npas_pre_medial[predict_npas_pre_medial[:, 1] > plot_thres_probs, 1]
        )

        if len(norm_npas_medial_post_dfs[0]) > 0:
            npas_medial_prediction_prob[1].extend(
                predict_npas_30_medial[
                    predict_npas_30_medial[:, 1] > plot_thres_probs, 1
                ]
            )
        npas_medial_prediction_prob[2].extend(
            predict_npas_60_medial[predict_npas_60_medial[:, 1] > plot_thres_probs, 1]
        )
        npas_medial_prediction_prob[3].extend(
            predict_npas_90_120_medial[
                predict_npas_90_120_medial[:, 1] > plot_thres_probs, 1
            ]
        )
        npas_medial_prediction_prob[4].extend(
            predict_npas_150_180_210_medial[
                predict_npas_150_180_210_medial[:, 1] > plot_thres_probs, 1
            ]
        )

        # get naive probabilites of all seeds for non medial Jaws
        predict_jaws_pre_non_medial = clf.predict_proba(
            norm_motor_dfs[0][is_medial[0] != 1]
        )
        predict_jaws_30_non_medial = clf.predict_proba(norm_jaws_non_medial_post_dfs[0])
        predict_jaws_60_non_medial = clf.predict_proba(norm_jaws_non_medial_post_dfs[1])
        predict_jaws_90_120_non_medial = clf.predict_proba(
            norm_jaws_non_medial_post_dfs[2]
        )
        predict_jaws_150_180_210_non_medial = clf.predict_proba(
            norm_jaws_non_medial_post_dfs[3]
        )

        jaws_non_medial_prediction_prob[0].extend(
            predict_jaws_pre_non_medial[
                predict_jaws_pre_non_medial[:, 1] > plot_thres_probs, 1
            ]
        )

        if len(norm_jaws_non_medial_post_dfs[0]) > 0:
            jaws_non_medial_prediction_prob[1].extend(
                predict_jaws_30_non_medial[
                    predict_jaws_30_non_medial[:, 1] > plot_thres_probs, 1
                ]
            )
        jaws_non_medial_prediction_prob[2].extend(
            predict_jaws_60_non_medial[
                predict_jaws_60_non_medial[:, 1] > plot_thres_probs, 1
            ]
        )
        jaws_non_medial_prediction_prob[3].extend(
            predict_jaws_90_120_non_medial[
                predict_jaws_90_120_non_medial[:, 1] > plot_thres_probs, 1
            ]
        )
        jaws_non_medial_prediction_prob[4].extend(
            predict_jaws_150_180_210_non_medial[
                predict_jaws_150_180_210_non_medial[:, 1] > plot_thres_probs, 1
            ]
        )

        # get naive probabilites of all seeds for non medial Npas
        predict_npas_pre_non_medial = clf.predict_proba(
            norm_motor_dfs[2][is_medial[2] != 1]
        )
        predict_npas_30_non_medial = clf.predict_proba(norm_npas_non_medial_post_dfs[0])
        predict_npas_60_non_medial = clf.predict_proba(norm_npas_non_medial_post_dfs[1])
        predict_npas_90_120_non_medial = clf.predict_proba(
            norm_npas_non_medial_post_dfs[2]
        )
        predict_npas_150_180_210_non_medial = clf.predict_proba(
            norm_npas_non_medial_post_dfs[3]
        )

        npas_non_medial_prediction_prob[0].extend(
            predict_npas_pre_non_medial[
                predict_npas_pre_non_medial[:, 1] > plot_thres_probs, 1
            ]
        )

        if len(norm_npas_non_medial_post_dfs[0]) > 0:
            npas_non_medial_prediction_prob[1].extend(
                predict_npas_30_non_medial[
                    predict_npas_30_non_medial[:, 1] > plot_thres_probs, 1
                ]
            )
        npas_non_medial_prediction_prob[2].extend(
            predict_npas_60_non_medial[
                predict_npas_60_non_medial[:, 1] > plot_thres_probs, 1
            ]
        )
        npas_non_medial_prediction_prob[3].extend(
            predict_npas_90_120_non_medial[
                predict_npas_90_120_non_medial[:, 1] > plot_thres_probs, 1
            ]
        )
        npas_non_medial_prediction_prob[4].extend(
            predict_npas_150_180_210_non_medial[
                predict_npas_150_180_210_non_medial[:, 1] > plot_thres_probs, 1
            ]
        )

        pre_prediction_prob[0] += clf.predict_proba(norm_motor_dfs[0]) / len(seeds)
        pre_prediction_prob[1] += clf.predict_proba(norm_motor_dfs[2]) / len(seeds)

        # Jaws post
        pre_prediction_prob[2] += clf.predict_proba(norm_jaws_post_dfs[0]) / len(seeds)
        pre_prediction_prob[3] += clf.predict_proba(norm_jaws_post_dfs[1]) / len(seeds)
        pre_prediction_prob[4] += clf.predict_proba(norm_jaws_post_dfs[2]) / len(seeds)
        pre_prediction_prob[5] += clf.predict_proba(norm_jaws_post_dfs[3]) / len(seeds)

        # npas post
        pre_prediction_prob[6] += clf.predict_proba(norm_npas_post_dfs[0]) / len(seeds)
        pre_prediction_prob[7] += clf.predict_proba(norm_npas_post_dfs[1]) / len(seeds)
        pre_prediction_prob[8] += clf.predict_proba(norm_npas_post_dfs[2]) / len(seeds)
        pre_prediction_prob[9] += clf.predict_proba(norm_npas_post_dfs[3]) / len(seeds)

        ### percents
        motor_predictions_post_times_percent[run_count, 0] = predictions_percent[0]
        motor_predictions_post_times_percent[run_count, len(post_jaws_dfs) + 1] = (
            predictions_percent[2]
        )
        motor_predictions_post_times_percent[run_count, 1 : len(post_jaws_dfs) + 1] = [
            np.sum(clf.predict(x)) / x.shape[0] for x in norm_jaws_post_dfs
        ]
        motor_predictions_post_times_percent[run_count, len(post_jaws_dfs) + 2 :] = [
            np.sum(clf.predict(x)) / x.shape[0] for x in norm_npas_post_dfs
        ]

        ### amounts in naive
        motor_predictions_post_times_amount[run_count, 0] = np.sum(predictions[0])
        motor_predictions_post_times_amount[run_count, len(post_jaws_dfs) + 1] = np.sum(
            predictions[2]
        )

        motor_predictions_post_times_amount[run_count, 1 : len(post_jaws_dfs) + 1] = [
            np.sum(clf.predict(x)) for x in norm_jaws_post_dfs
        ]
        motor_predictions_post_times_amount[
            run_count,
            len(post_jaws_dfs) + 2 : motor_predictions_post_times_amount.shape[1] // 2,
        ] = [np.sum(clf.predict(x)) for x in norm_npas_post_dfs]

        ### amounts in dd
        motor_predictions_post_times_amount[
            run_count, motor_predictions_post_times_amount.shape[1] // 2
        ] = np.count_nonzero(predictions[0] == 0)
        motor_predictions_post_times_amount[
            run_count,
            len(post_jaws_dfs) + 1 + motor_predictions_post_times_amount.shape[1] // 2,
        ] = np.count_nonzero(predictions[2] == 0)

        motor_predictions_post_times_amount[
            run_count,
            motor_predictions_post_times_amount.shape[1] // 2
            + 1 : len(post_jaws_dfs)
            + 1
            + motor_predictions_post_times_amount.shape[1] // 2,
        ] = [np.count_nonzero(clf.predict(x) == 0) for x in norm_jaws_post_dfs]
        motor_predictions_post_times_amount[
            run_count,
            motor_predictions_post_times_amount.shape[1] // 2
            + len(post_jaws_dfs)
            + 2 :,
        ] = [np.count_nonzero(clf.predict(x) == 0) for x in norm_npas_post_dfs]

        ### precision
        precision[run_count, :2], recall[run_count, :2], _, _ = (
            precision_recall_fscore_support(y_train, predicts_train)
        )
        precision[run_count, 2:], recall[run_count, 2:], _, _ = (
            precision_recall_fscore_support(y_test, predicts_test)
        )

        accuracy_runs[run_count] = [accuracy_train, accuracy_test]
        motor_predictions_percent[run_count, :] = predictions_percent

        medial_predictions = [
            predictions[x][is_medial[x] == 1] for x in range(len(is_medial))
        ]
        non_medial_predictions = [
            predictions[x][is_medial[x] == 0] for x in range(len(is_medial))
        ]

        jaws_medial_predictions = [
            clf.predict(x) if len(x) > 0 else -1 for x in norm_jaws_medial_post_dfs
        ]
        npas_medial_predictions = [
            clf.predict(x) if len(x) > 0 else -1 for x in norm_npas_medial_post_dfs
        ]

        jaws_non_medial_predictions = [
            clf.predict(x) if len(x) > 0 else -1 for x in norm_jaws_non_medial_post_dfs
        ]
        npas_non_medial_predictions = [
            clf.predict(x) if len(x) > 0 else -1 for x in norm_npas_non_medial_post_dfs
        ]

        medial_predictions_percent[run_count, 0] = (
            np.sum(medial_predictions[0]) / medial_predictions[0].shape[0]
        )
        medial_predictions_percent[run_count, len(post_jaws_dfs) + 1] = (
            np.sum(medial_predictions[2]) / medial_predictions[2].shape[0]
        )
        medial_predictions_percent[run_count, 1 : len(post_jaws_dfs) + 1] = [
            np.sum(x) / x.shape[0] if isinstance(x, np.ndarray) else -1
            for x in jaws_medial_predictions
        ]
        medial_predictions_percent[run_count, len(post_jaws_dfs) + 2 :] = [
            np.sum(x) / x.shape[0] if isinstance(x, np.ndarray) else -1
            for x in npas_medial_predictions
        ]

        non_medial_predictions_percent[run_count, 0] = (
            np.sum(non_medial_predictions[0]) / non_medial_predictions[0].shape[0]
        )
        non_medial_predictions_percent[run_count, len(post_jaws_dfs) + 1] = (
            np.sum(non_medial_predictions[2]) / non_medial_predictions[2].shape[0]
        )
        non_medial_predictions_percent[run_count, 1 : len(post_jaws_dfs) + 1] = [
            np.sum(x) / x.shape[0] if isinstance(x, np.ndarray) else -1
            for x in jaws_non_medial_predictions
        ]
        non_medial_predictions_percent[run_count, len(post_jaws_dfs) + 2 :] = [
            np.sum(x) / x.shape[0] if isinstance(x, np.ndarray) else -1
            for x in npas_non_medial_predictions
        ]

        """ medial_predictions_percent[run_count] = [
            np.sum(x) / x.shape[0] for x in medial_predictions
        ]

        non_medial_predictions_percent[run_count] = [
            np.sum(x) / x.shape[0] for x in non_medial_predictions
        ]"""
        # print(clf.predict_proba(X_train_norm))
        # print(clf.predict(X_train_norm))

        print("Acc Train Test: ", accuracy_train, accuracy_test)
        print()
        cm_train += confusion_matrix(y_train, predicts_train, labels=clf.classes_)
        cm_test += confusion_matrix(y_test, predicts_test, labels=clf.classes_)
        run_count += 1

    # print(motor_predictions_post_times_amount)

    """for i in range(jaws_tracked_probs.shape[0]):
        np.savetxt(
            f"../data/tracked_units/jaws_treatment_../data/jaws_treatment_cell_{i+1}_probabilities.txt",
            jaws_tracked_probs[i, :, :],
            newline="\n",
            delimiter=" ",
            fmt="%f",
        )

    for i in range(npas_tracked_probs.shape[0]):
        np.savetxt(
            f"../data/tracked_units/npas_treatment_../data/npas_treatment_cell_{i+1}_probabilities.txt",
            npas_tracked_probs[i, :, :],
            newline="\n",
            delimiter=" ",
            fmt="%f",
        )"""

    """pd.DataFrame(
        motor_predictions_post_times_amount,
        columns=[
            "Jaws Pre Naive",
            "Jaws Post 30 Naive",
            "Jaws Post 60 Naive",
            "Jaws Post 90-120 Naive",
            "Jaws Post 150-210 Naive",
            "Npas Pre Naive",
            "Npas Post 30 Naive",
            "Npas Post 60 Naive",
            "Npas Post 90-120 Naive",
            "Npas Post 150-210 Naive",
            "Jaws Pre DD",
            "Jaws Post 30 DD",
            "Jaws Post 60 DD",
            "Jaws Post 90-120 DD",
            "Jaws Post 150-210 DD",
            "Npas Pre DD",
            "Npas Post 30 DD",
            "Npas Post 60 DD",
            "Npas Post 90-120 DD",
            "Npas Post 150-210 DD",
        ],
        index=["Seed 1", "Seed 2", "Seed 3", "Seed 4", "Seed 5"],
    ).to_csv(f"../data/neural_net/cells_predicted_jaws_dd.csv")

    np.savetxt(
        f"../data/neural_net/pre_jaws_probabilities.txt",
        pre_prediction_prob[0],
        newline="\n",
        fmt="%f",
    )
    np.savetxt(
        f"../data/neural_net/pre_npas_probabilities.txt",
        pre_prediction_prob[1],
        newline="\n",
        fmt="%f",
    )

    np.savetxt(
        f"../data/neural_net/post_30_jaws_probabilities.txt",
        pre_prediction_prob[2],
        newline="\n",
        fmt="%f",
    )
    np.savetxt(
        f"../data/neural_net/post_60_jaws_probabilities.txt",
        pre_prediction_prob[3],
        newline="\n",
        fmt="%f",
    )
    np.savetxt(
        f"../data/neural_net/post_90_120_jaws_probabilities.txt",
        pre_prediction_prob[4],
        newline="\n",
        fmt="%f",
    )
    np.savetxt(
        f"../data/neural_net/post_150_180_210_jaws_probabilities.txt",
        pre_prediction_prob[5],
        newline="\n",
        fmt="%f",
    )

    np.savetxt(
        f"../data/neural_net/post_30_npas_probabilities.txt",
        pre_prediction_prob[6],
        newline="\n",
        fmt="%f",
    )
    np.savetxt(
        f"../data/neural_net/post_60_npas_probabilities.txt",
        pre_prediction_prob[7],
        newline="\n",
        fmt="%f",
    )
    np.savetxt(
        f"../data/neural_net/post_90_120_npas_probabilities.txt",
        pre_prediction_prob[8],
        newline="\n",
        fmt="%f",
    )
    np.savetxt(
        f"../data/neural_net/post_150_180_210_npas_probabilities.txt",
        pre_prediction_prob[9],
        newline="\n",
        fmt="%f",
    )"""

    print(
        [len(x[x[:, 0] > 0.9, 0]) / len(x[x[:, 0] > 0.5, 0]) for x in predict_jaws_all]
    )
    print(
        [len(x[x[:, 0] > 0.9, 0]) / len(x[x[:, 0] > 0.5, 0]) for x in predict_npas_all]
    )

    print(
        [len(x[x[:, 1] > 0.9, 1]) / len(x[x[:, 1] > 0.5, 1]) for x in predict_jaws_all]
    )
    print(
        [len(x[x[:, 1] > 0.9, 1]) / len(x[x[:, 1] > 0.5, 1]) for x in predict_npas_all]
    )

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300, tight_layout=True)
    ax.plot(
        np.arange(5),
        [len(x[x[:, 0] > 0.9, 0]) / len(x[x[:, 0] > 0.5, 0]) for x in predict_jaws_all],
        marker="o",
        color="b",
        label="Jaws",
    )
    ax.plot(
        np.arange(5),
        [len(x[x[:, 0] > 0.9, 0]) / len(x[x[:, 0] > 0.5, 0]) for x in predict_npas_all],
        marker="o",
        color="r",
        label="Npas",
    )
    ax.legend()
    ax.set_ylabel("% DD Prob > 0.9 ")
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(["Pre-Stim", "30min", "60min", "90-120min", "150-210min"])
    makeNice(ax)
    fig.savefig(f"../data/neural_net/prob_conf.pdf", bbox_inches="tight")
    plt.close()

    plot_predict_cont_mat(
        norm_motor_dfs,
        motor_predictions_post_times_percent,
        medial_predictions_percent,
        non_medial_predictions_percent,
        medial_predictions,
        non_medial_predictions,
        accuracy_runs,
        cm_train,
        cm_test,
        precision,
        recall,
        jaws_prediction_prob,
        npas_prediction_prob,
        jaws_medial_prediction_prob,
        npas_medial_prediction_prob,
        jaws_non_medial_prediction_prob,
        npas_non_medial_prediction_prob,
        show=show,
    )
    """sys.exit()

    print("Acc Train Test: ", accuracy_train, accuracy_test)
    print("Predict %: ", predictions_percent)
    print("Medial Predict %: ", medial_predictions_percent)
    print("Non Medial Predict %: ", non_medial_predictions_percent)

    print("Naive Actual: ", np.sum(y_train) / y_train.shape[0])
    print("DD Actual: ", 1 - np.sum(y_train) / y_train.shape[0])
    print("Naive Predicted: ", np.sum(predicts_train) / predicts_train.shape[0])
    print("DD Predicted: ", 1 - np.sum(predicts_train) / predicts_train.shape[0])
    print(
        "Incorrects Naive: ",
        predicts_train[(predicts_train != y_train) & (predicts_train == 0)].shape[0]
        / predicts_train.shape[0],
    )
    print(
        "Incorrects DD: ",
        predicts_train[(predicts_train != y_train) & (predicts_train == 1)].shape[0]
        / predicts_train.shape[0],
    )

    predictions = [clf.predict(x) for x in norm_motor_dfs]
    predictions_percent = [np.sum(clf.predict(x)) / x.shape[0] for x in norm_motor_dfs]

    predicts_train = clf.predict(X_train_norm)
    accuracy_train = accuracy_score(y_train, predicts_train)

    medial_predictions = [
        predictions[x][is_medial[x] == 1] for x in range(len(is_medial))
    ]
    non_medial_predictions = [
        predictions[x][is_medial[x] == 0] for x in range(len(is_medial))
    ]

    medial_predictions_percent = [np.sum(x) / x.shape[0] for x in medial_predictions]

    non_medial_predictions_percent = [
        np.sum(x) / x.shape[0] for x in non_medial_predictions
    ]"""

    """
    max_time = 250
    fig, ax = plt.subplots(4, 3, figsize=(12, 8), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]

    norm_motor_dfs[1] = norm_motor_dfs[1][jaws_post_times <= max_time]
    norm_motor_dfs[3] = norm_motor_dfs[3][npas_post_times <= max_time]
    is_medial[1] = is_medial[1][jaws_post_times <= max_time]
    is_medial[3] = is_medial[3][npas_post_times <= max_time]

    axis_count = 0
    for col in target_df.columns[feature_array]:
        if col != "Type":
            jaws_pre = motor_feature_dfs[0][col]
            jaws_post = motor_feature_dfs[1][col]
            jaws_post = jaws_post[jaws_post_times <= max_time]
            npas_pre = motor_feature_dfs[2][col]
            npas_post = motor_feature_dfs[3][col]
            npas_post = npas_post[npas_post_times <= max_time]

            jaws_pval = scipy.stats.ttest_ind(
                jaws_pre, jaws_post, equal_var=False
            ).pvalue
            npas_pval = scipy.stats.ttest_ind(
                npas_pre, npas_post, equal_var=False
            ).pvalue

            axes[axis_count].plot(
                [0, 1], [np.mean(jaws_pre), np.mean(jaws_post)], color="b", marker="o"
            )
            axes[axis_count].plot(
                [2, 3], [np.mean(npas_pre), np.mean(npas_post)], color="r", marker="o"
            )

            axes[axis_count].errorbar(
                0,
                np.mean(jaws_pre),
                yerr=scipy.stats.sem(jaws_pre),
                color="b",
                capsize=2,
                lw=0.5,
            )
            axes[axis_count].errorbar(
                2,
                np.mean(npas_pre),
                yerr=scipy.stats.sem(npas_pre),
                color="r",
                capsize=2,
                lw=0.5,
            )

            axes[axis_count].errorbar(
                1,
                np.mean(jaws_post),
                yerr=scipy.stats.sem(jaws_post),
                color="b",
                capsize=2,
                lw=0.5,
            )
            axes[axis_count].errorbar(
                3,
                np.mean(npas_post),
                yerr=scipy.stats.sem(npas_post),
                color="r",
                capsize=2,
                lw=0.5,
            )

            jaws_post_y = np.mean(jaws_post) + scipy.stats.sem(jaws_post)
            jaws_pre_y = np.mean(jaws_pre) + scipy.stats.sem(jaws_pre)
            jaws_bracket_y = jaws_pre_y if jaws_pre_y > jaws_post_y else jaws_post_y
            plot_data.plot_bracket(
                axes[axis_count],
                0,
                1,
                jaws_bracket_y,
                jaws_bracket_y * 1.05,
                f"T-test: {jaws_pval:.03f}",
                color="k",
                lw=0.5,
            )

            npas_post_y = np.mean(npas_post) + scipy.stats.sem(npas_post)
            npas_pre_y = np.mean(npas_pre) + scipy.stats.sem(npas_pre)
            npas_bracket_y = npas_pre_y if npas_pre_y > npas_post_y else npas_post_y
            plot_data.plot_bracket(
                axes[axis_count],
                2,
                3,
                npas_bracket_y,
                npas_bracket_y * 1.05,
                f"T-test: {npas_pval:.03f}",
                color="k",
                lw=0.5,
            )

            axes[axis_count].set_ylabel(col)
            axes[axis_count].set_xticks([0, 1, 2, 3])
            axes[axis_count].set_xticklabels(
                [
                    f"Jaws-Pre\nn={jaws_pre.shape[0]}",
                    f"Jaws-Post\nn={jaws_post.shape[0]}",
                    f"Npas-Pre\nn={npas_pre.shape[0]}",
                    f"Npas-Post\nn={npas_post.shape[0]}",
                ]
            )

            axis_count += 1

    makeNice(axes, labelsize=8)
    fig.savefig(
        f"../data/neural_net/motor_rescue_prediction_pre_post_{max_time}.pdf",
        bbox_inches="tight",
    )
    plt.close()
    if show:
        run_cmd(f"open ../data/neural_net/motor_rescue_prediction_pre_post_{max_time}.pdf")
    """


def plot_predict_cont_mat(
    norm_motor_dfs,
    predictions_percent,
    medial_predictions_percent,
    non_medial_predictions_percent,
    medial_predictions,
    non_medial_predictions,
    accuracy_runs,
    cm_train,
    cm_test,
    precision,
    recall,
    jaws_predictions_prob,
    npas_predictions_prob,
    jaws_medial_prediction_prob,
    npas_medial_prediction_prob,
    jaws_non_medial_prediction_prob,
    npas_non_medial_prediction_prob,
    show=False,
):
    jaws_predictions_percent = predictions_percent[
        :, : predictions_percent.shape[1] // 2
    ]

    npas_predictions_percent = predictions_percent[
        :, predictions_percent.shape[1] // 2 :
    ]

    jaws_medial_predictions_percent = medial_predictions_percent[
        :, : predictions_percent.shape[1] // 2
    ]
    npas_medial_predictions_percent = medial_predictions_percent[
        :, predictions_percent.shape[1] // 2 :
    ]

    jaws_non_medial_predictions_percent = non_medial_predictions_percent[
        :, : predictions_percent.shape[1] // 2
    ]
    npas_non_medial_predictions_percent = non_medial_predictions_percent[
        :, predictions_percent.shape[1] // 2 :
    ]

    jaws_x_medial = np.asarray(
        [
            i
            for i in range(len(jaws_medial_predictions_percent))
            if jaws_medial_predictions_percent[0, i] != -1
        ]
    )
    jaws_y_medial = jaws_medial_predictions_percent[:, jaws_x_medial]

    npas_x_medial = np.asarray(
        [
            i
            for i in range(len(npas_medial_predictions_percent))
            if npas_medial_predictions_percent[0, i] != -1
        ]
    )
    npas_y_medial = npas_medial_predictions_percent[:, npas_x_medial]

    jaws_x_non_medial = np.asarray(
        [
            i
            for i in range(len(jaws_non_medial_predictions_percent))
            if jaws_non_medial_predictions_percent[0, i] != -1
        ]
    )
    jaws_y_non_medial = jaws_non_medial_predictions_percent[:, jaws_x_non_medial]

    npas_x_non_medial = np.asarray(
        [
            i
            for i in range(len(npas_non_medial_predictions_percent))
            if npas_non_medial_predictions_percent[0, i] != -1
        ]
    )
    npas_y_non_medial = npas_non_medial_predictions_percent[:, npas_x_non_medial]

    accuracy_train = accuracy_runs[:, 0]
    accuracy_test = accuracy_runs[:, 1]
    precision_train = precision[:, :2]
    precision_test = precision[:, 2:]
    recall_train = recall[:, :2]
    recall_test = recall[:, 2:]

    fig, ax = plt.subplots(3, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(3) for j in range(2)]

    # cm = confusion_matrix(y_train, predicts_train, labels=clf.classes_)
    # cm = cm / cm.sum()
    # cm_train = cm_train / cm_train.sum()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=[0, 1])
    # precision, recall, _, _ = precision_recall_fscore_support(y_train, predicts_train)

    disp.plot(ax=axes[0], im_kw={"aspect": "auto"}, values_format=".0f")
    axes[0].set_title(
        f"Accuracy: {np.mean(accuracy_train,axis=0):.02f}\nPrecision: {np.mean(precision_train[:,1]):.02f}, {np.mean(precision_train[:,0]):.02f}\nRecall: {np.mean(recall_train[:,1]):.02f}, {np.mean(recall_train[:,0]):.02f}",
        fontsize=10,
    )

    # cm = confusion_matrix(y_test, predicts_test, labels=clf.classes_)
    # cm = cm / cm.sum()
    # cm_test = cm_test / cm_test.sum()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=[0, 1])
    # precision, recall, _, _ = precision_recall_fscore_support(y_test, predicts_test)

    disp.plot(ax=axes[1], im_kw={"aspect": "auto"}, values_format=".0f")
    axes[1].set_title(
        f"Accuracy: {np.mean(accuracy_test,axis=0):.02f}\nPrecision: {np.mean(precision_test[:,1]):.02f}, {np.mean(precision_test[:,0]):.02f}\nRecall: {np.mean(recall_test[:,1]):.02f}, {np.mean(recall_test[:,0]):.02f}",
        fontsize=10,
    )

    axes[2].errorbar(
        np.arange(jaws_predictions_percent.shape[1]),
        np.mean(jaws_predictions_percent, axis=0),
        yerr=scipy.stats.sem(jaws_predictions_percent, axis=0),
        marker="o",
        color="b",
        lw=0.5,
        markersize=4,
        capsize=4,
        label="Jaws",
    )

    axes[2].errorbar(
        np.arange(jaws_predictions_percent.shape[1]),
        np.mean(npas_predictions_percent, axis=0),
        yerr=scipy.stats.sem(npas_predictions_percent, axis=0),
        marker="o",
        color="r",
        lw=0.5,
        markersize=4,
        capsize=4,
        label="Npas",
    )
    axes[2].legend(fancybox=False, frameon=False)

    ### PLOT MEDIAL ###

    axes[3].errorbar(
        jaws_x_medial,
        np.mean(jaws_y_medial, axis=0),
        yerr=scipy.stats.sem(jaws_y_medial, axis=0),
        marker="o",
        color="b",
        ls="dashed",
        lw=0.5,
        markersize=4,
        capsize=4,
        # label=f"JAWS All,{np.mean(predictions_percent[:,0])*100:.2f}% to {np.mean(predictions_percent[:,1])*100:.2f}%",
    )

    axes[3].errorbar(
        np.arange(
            jaws_medial_predictions_percent.shape[1],
            (npas_x_medial[-1] + 1) * 2,
        ),
        np.mean(npas_y_medial, axis=0),
        yerr=scipy.stats.sem(npas_y_medial, axis=0),
        marker="o",
        color="r",
        ls="dashed",
        lw=0.5,
        markersize=4,
        capsize=4,
        # label=f"Npas All,{np.mean(predictions_percent[:,2])*100:.2f}% to {np.mean(predictions_percent[:,3])*100:.2f}%",
    )

    ### PLOT NON MEDIAL

    axes[3].errorbar(
        np.arange(jaws_non_medial_predictions_percent.shape[1]),
        np.mean(jaws_non_medial_predictions_percent, axis=0),
        yerr=scipy.stats.sem(jaws_non_medial_predictions_percent, axis=0),
        marker="o",
        color="b",
        lw=0.5,
        ls="dotted",
        markersize=4,
        capsize=4,
        # label=f"JAWS All,{np.mean(predictions_percent[:,0])*100:.2f}% to {np.mean(predictions_percent[:,1])*100:.2f}%",
    )

    axes[3].errorbar(
        np.arange(
            jaws_non_medial_predictions_percent.shape[1],
            npas_non_medial_predictions_percent.shape[1] * 2,
        ),
        np.mean(npas_non_medial_predictions_percent, axis=0),
        yerr=scipy.stats.sem(npas_non_medial_predictions_percent, axis=0),
        marker="o",
        color="r",
        ls="dotted",
        lw=0.5,
        markersize=4,
        capsize=4,
        # label=f"Npas All,{np.mean(predictions_percent[:,2])*100:.2f}% to {np.mean(predictions_percent[:,3])*100:.2f}%",
    )

    # axes[3].set_xticklabels([f"Jaws-Pre", f"Jaws-Post", f"Npas-Pre", f"Npas-Post"])

    axes[2].legend(fancybox=False, frameon=False, fontsize=6)
    axes[3].legend(fancybox=False, frameon=False, fontsize=6, ncols=2)

    for i in range(2):
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["DD", "Naive"])
        axes[i].set_yticks([0, 1])
        axes[i].set_yticklabels(["DD", "Naive"])

    """plt.suptitle(
        f"Naive/DD Train: {len(y_train[y_train==1])}/{len(y_train[y_train==0])}, Naive/DD Test: {len(y_test[y_test==1])}/{len(y_test[y_test==0])}"
    )"""

    axes[4].errorbar(
        x=np.arange(5),
        y=[np.mean(x) for x in jaws_predictions_prob],
        yerr=[scipy.stats.sem(x) for x in jaws_predictions_prob],
        color="b",
        marker="o",
        lw=0.5,
        markersize=4,
        capsize=4,
    )

    axes[4].errorbar(
        x=np.arange(5),
        y=[np.mean(x) for x in npas_predictions_prob],
        yerr=[scipy.stats.sem(x) for x in npas_predictions_prob],
        color="r",
        marker="o",
        lw=0.5,
        markersize=4,
        capsize=4,
    )

    axes[4].set_ylabel("Naive Probability")

    axes[4].set_xticks(np.arange(5))
    axes[4].set_xticklabels(
        [
            f"Pre",
            30,
            60,
            "90-120",
            "150-210",
        ]
    )
    axes[4].grid(lw=0.1, alpha=0.5)

    ### PLOT MEDIAL ###
    axes[5].errorbar(
        jaws_x_medial,
        y=[np.mean(jaws_medial_prediction_prob[x]) for x in jaws_x_medial],
        yerr=[scipy.stats.sem(jaws_medial_prediction_prob[x]) for x in jaws_x_medial],
        marker="o",
        color="b",
        ls="dashed",
        lw=0.5,
        markersize=4,
        capsize=4,
    )

    axes[5].errorbar(
        np.arange(
            jaws_medial_predictions_percent.shape[1],
            (npas_x_medial[-1] + 1) * 2,
        ),
        y=[np.mean(x) for x in npas_medial_prediction_prob],
        yerr=[scipy.stats.sem(x) for x in npas_medial_prediction_prob],
        marker="o",
        color="r",
        ls="dashed",
        lw=0.5,
        markersize=4,
        capsize=4,
    )

    ### PLOT NON MEDIAL

    axes[5].errorbar(
        np.arange(jaws_non_medial_predictions_percent.shape[1]),
        y=[np.mean(x) for x in jaws_non_medial_prediction_prob],
        yerr=[scipy.stats.sem(x) for x in jaws_non_medial_prediction_prob],
        marker="o",
        color="b",
        lw=0.5,
        ls="dotted",
        markersize=4,
        capsize=4,
    )

    axes[5].errorbar(
        np.arange(
            jaws_non_medial_predictions_percent.shape[1],
            npas_non_medial_predictions_percent.shape[1] * 2,
        ),
        y=[np.mean(x) for x in npas_non_medial_prediction_prob],
        yerr=[scipy.stats.sem(x) for x in npas_non_medial_prediction_prob],
        marker="o",
        color="r",
        ls="dotted",
        lw=0.5,
        markersize=4,
        capsize=4,
    )

    for i in [2, 3, 5]:
        axes[i].set_ylabel("Predicted Naive %" if i != 5 else "Naive Probability")
        axes[i].set_xticks([0, 1, 2, 3])
        xlim = axes[i].get_xlim()
        if i == 3 or i == 2:
            axes[i].hlines(
                1 - np.mean(precision_train[:, 0]),
                xlim[0],
                xlim[1],
                color="k",
                lw=0.5,
                ls="dashed",
            )
            axes[i].hlines(
                1 - np.mean(recall_train[:, 0]),
                xlim[0],
                xlim[1],
                color="k",
                lw=0.5,
                ls="dotted",
            )
            axes[i].set_xlim(xlim)

        if i == 3 or i == 5:
            axes[i].set_xticks(np.arange(predictions_percent.shape[1]))
            axes[i].set_xticklabels(
                [
                    f"Jaws-Pre",
                    30,
                    60,
                    "90-120",
                    "150-210",
                    f"Npas-Pre",
                    30,
                    60,
                    "90-120",
                    "150-210",
                ]
            )
        else:
            axes[i].set_xticks(np.arange(predictions_percent.shape[1] // 2))
            axes[i].set_xticklabels(
                [
                    f"Pre",
                    30,
                    60,
                    "90-120",
                    "150-210",
                ]
            )
        axes[i].grid(lw=0.1, alpha=0.5)

    makeNice(axes[2:], labelsize=6)
    match_axis(axes[2:4], type="y")
    match_axis(axes[4:], type="y")
    add_fig_labels(axes)
    fig.savefig(
        f"../data/neural_net/motor_rescue_conf_mat.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()

    jaws_predictions_prob = [np.asarray(x) for x in jaws_predictions_prob]
    npas_predictions_prob = [np.asarray(x) for x in npas_predictions_prob]
    print([len(x[x > 0.9]) / len(x) for x in jaws_predictions_prob])
    print([len(x[x > 0.9]) / len(x) for x in npas_predictions_prob])

    fig2, ax2 = plt.subplots(2, 3, figsize=(8, 6), dpi=300, tight_layout=True)
    axes2 = [ax2[i, j] for i in range(2) for j in range(3)]
    bins = np.arange(0.0, 1.1, 0.05)
    for i in range(5):
        sns.histplot(
            x=jaws_predictions_prob[i],
            kde=True,
            color="b",
            edgecolor="w",
            bins=bins,
            stat="probability",
            ax=axes2[i],
        )
        sns.histplot(
            x=npas_predictions_prob[i],
            kde=True,
            color="r",
            edgecolor="w",
            bins=bins,
            stat="probability",
            ax=axes2[i],
        )
        ylims = axes2[i].get_ylim()
        axes2[i].vlines(
            np.mean(jaws_predictions_prob[i]),
            ylims[0],
            ylims[1],
            ls="dashed",
            lw=0.5,
            color="b",
        )
        axes2[i].vlines(
            np.mean(npas_predictions_prob[i]),
            ylims[0],
            ylims[1],
            ls="dashed",
            lw=0.5,
            color="r",
        )
        axes2[i].set_ylim(ylims)
    makeNice(axes2)
    fig2.savefig(
        "../data/neural_net/motor_rescue_probs_hist.pdf", bbox_inches="tight"
    )
    plt.close()

    labels = ["Pre", "30 min", "60 min", "90-120 min", "150-210 min"]
    fig3, ax3 = plt.subplots(1, 5, figsize=(8, 2), dpi=300, tight_layout=True)
    axes3 = [ax3[i] for i in range(5)]
    for i in range(5):
        sns.ecdfplot(
            x=jaws_predictions_prob[i],
            color="b",
            stat="proportion",
            ax=axes3[i],
            legend=False,
            lw=0.5,
        )
        sns.ecdfplot(
            x=npas_predictions_prob[i],
            color="r",
            stat="proportion",
            ax=axes3[i],
            legend=False,
            lw=0.5,
        )
        axes3[i].vlines(
            np.mean(jaws_predictions_prob[i]), 0, 1, ls="dashed", color="b", lw=0.5
        )
        axes3[i].vlines(
            np.median(jaws_predictions_prob[i]), 0, 1, ls="dotted", color="b", lw=0.5
        )
        axes3[i].vlines(
            np.mean(npas_predictions_prob[i]), 0, 1, ls="dashed", color="r", lw=0.5
        )
        axes3[i].vlines(
            np.median(npas_predictions_prob[i]), 0, 1, ls="dotted", color="r", lw=0.5
        )
        axes3[i].set_ylim([0, 1])
        axes3[i].set_xlabel(f"Naive Prob {labels[i]}", fontsize=6)
        axes3[i].set_ylabel(f"Proportion", fontsize=6)
    makeNice(axes3, labelsize=6)
    fig3.savefig(
        "../data/neural_net/motor_rescue_probs_ecdf.pdf", bbox_inches="tight"
    )
    plt.close()
    if show:
        run_cmd("open ../data/neural_net/motor_rescue_conf_mat.pdf")
        run_cmd("open ../data/neural_net/motor_rescue_probs_hist.pdf")
        run_cmd("open ../data/neural_net/motor_rescue_probs_ecdf.pdf")


def plot_osc_over_time(
    feature_array,
    motor_rescue_dfs,
    post_jaws_dfs,
    post_npas_dfs,
    motor_rescue_dfs_osc,
    jaws_post_times,
    post_jaws_osc,
    npas_post_times,
    post_npas_osc,
    is_medial,
    show=True,
):

    motor_feature_dfs = [x.iloc[:, feature_array] for x in motor_rescue_dfs]
    norm_motor_dfs = [clean_data.normalize_data(x) for x in motor_feature_dfs]

    jaws_pre_post = [
        np.sum(motor_rescue_dfs_osc[0]) / motor_rescue_dfs_osc[0].shape[0],
        np.sum(motor_rescue_dfs_osc[1]) / motor_rescue_dfs_osc[1].shape[0],
    ]

    npas_pre_post = [
        np.sum(motor_rescue_dfs_osc[2]) / motor_rescue_dfs_osc[2].shape[0],
        np.sum(motor_rescue_dfs_osc[3]) / motor_rescue_dfs_osc[3].shape[0],
    ]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=300, tight_layout=True)
    axes = [ax[i] for i in range(3)]

    ### ALL
    axes[0].plot([0, 1], jaws_pre_post, marker="o", color="b", lw=0.5, markersize=4)
    axes[0].plot([2, 3], npas_pre_post, marker="o", color="r", lw=0.5, markersize=4)

    axes[0].set_ylabel("Delta Osc %")
    axes[0].set_xticks([0, 1, 2, 3])
    axes[0].set_xticklabels(
        [
            f"Jaws-Pre\nn={motor_rescue_dfs_osc[0].shape[0]}",
            f"Jaws-Post\nn={motor_rescue_dfs_osc[1].shape[0]}",
            f"Npas-Pre\nn={motor_rescue_dfs_osc[2].shape[0]}",
            f"Npas-Post\nn={motor_rescue_dfs_osc[3].shape[0]}",
        ]
    )

    jaws_osc = [np.sum(x) / x.shape[0] for x in post_jaws_osc]
    npas_osc = [np.sum(x) / x.shape[0] for x in post_npas_osc]
    jaws_osc.insert(
        0, np.sum(motor_rescue_dfs_osc[0]) / motor_rescue_dfs_osc[0].shape[0]
    )
    npas_osc.insert(
        0, np.sum(motor_rescue_dfs_osc[2]) / motor_rescue_dfs_osc[2].shape[0]
    )

    unique_jaws = np.unique(jaws_post_times)
    unique_npas = np.unique(npas_post_times)

    unique_jaws = np.insert(unique_jaws, 0, 0)
    unique_npas = np.insert(unique_npas, 0, 0)

    axes[1].plot(unique_jaws, jaws_osc, marker="o", color="b", lw=0.5, markersize=4)
    axes[1].plot(unique_npas, npas_osc, marker="o", color="r", lw=0.5, markersize=4)

    xlabels = [x for x in unique_jaws[1:]]
    xlabels.insert(0, "Pre-Stim")

    axes[1].set_ylabel("Delta Osc %")
    axes[1].set_xticks(unique_jaws)
    axes[1].set_xticklabels(xlabels)
    axes[1].set_xlabel("Post-Stim Time (min)")

    clean_data.plot_column_comparison_over_time(
        axes[2],
        "Delta Power",
        motor_feature_dfs[0],
        motor_feature_dfs[2],
        jaws_post_times,
        post_jaws_dfs,
        npas_post_times,
        post_npas_dfs,
    )

    makeNice(axes, labelsize=8)
    fig.savefig(
        f"../data/neural_net/motor_rescue_prediction_osc.pdf", bbox_inches="tight"
    )
    plt.close()
    if show:
        run_cmd("open ../data/neural_net/motor_rescue_prediction_osc.pdf")
