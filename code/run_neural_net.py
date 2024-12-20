"""
run_neural_net.py

This script trains a neural network to predict motor rescue from a set of features and tests what features contribute the most.

Author: John E. Parker
"""

# python modules
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import scipy.stats
import matplotlib as mpl
import pickle

# set plotting parameters
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# user modules
import clean_data
from helpers import *


def predict_motor_rescue(
    target_df,
    feature_array,
    outlier_dict,
    jaws_npas_pre_post,
    train_amount=0.8,
    seeds=np.arange(15),
    show=False,
    test_outliers=False,
    min_max_scale=False,
):
    """
    Generates a neural network to predict motor rescue from a set of features to determine binary classification. Creates plots to display results.

    Parameters
    ----------
    \t target_df (dataframe): dataframe containing the target data

    \t feature_array (list): list of columns to use as features

    \t outlier_dict (dict): dictionary containing the outlier thresholds for features

    \t jaws_npas_pre_post (list): list of dataframes containing the JAWS and NPAS data

    \t train_amount=0.8 (float): proportion of data to use for training

    \t seeds=np.arange(15) (list): list of random seeds to use for training

    \t show=False (bool): whether to show the plots

    \t test_outliers=False (bool): whether to test the outliers

    \t min_max_scale=False (bool): whether to use min-max scaling
    """

    # Remove outliers
    target_df_outliers_removed = clean_data.remove_outliers_by_group_zscore_independent(
        target_df[target_df["Type"] == 1],
        target_df[target_df["Type"] == 0],
        outlier_dict,
    )

    # Get features
    feature_df = target_df_outliers_removed.iloc[:, feature_array]

    # Remove outliers
    target_df_outliers = target_df.loc[
        ~target_df.index.isin(target_df_outliers_removed.index)
    ]

    # Get outliers
    y_outliers = target_df_outliers["Type"]
    target_df_outliers = target_df_outliers.drop(columns="Type")

    # Drop columns
    jaws_npas_pre_post_dropped = [
        x.drop(columns=["mouse", "folder", "name", "Post-Time", "Medial"])
        for x in jaws_npas_pre_post
    ]

    # Initialize variables
    jaws_npas_pre_post_predicts = [
        np.zeros((len(seeds), 4)) for x in jaws_npas_pre_post
    ]  # jaws predicted dd, jaws predicted naive, npas predicted dd, npas predicted naive

    jaws_npas_pre_post_predicts_medial = [
        np.zeros((len(seeds), 4)) for x in jaws_npas_pre_post
    ]  # jaws predicted dd, jaws predicted naive, npas predicted dd, npas predicted naive

    jaws_npas_pre_post_predicts_non_medial = [
        np.zeros((len(seeds), 4)) for x in jaws_npas_pre_post
    ]  # jaws predicted dd, jaws predicted naive, npas predicted dd, npas predicted naive

    jaws_npas_pre_post_probs = [
        np.zeros((len(seeds), len(x), 2)) for x in jaws_npas_pre_post
    ]  # dd naive probabilities

    outlier_probabilities = np.zeros((len(target_df_outliers), 2))

    accuracies = np.zeros((len(seeds), 2))  # train test
    precisions = np.zeros((len(seeds), 4))  # train train test test
    recalls = np.zeros((len(seeds), 4))  # train train test test
    cm_train = np.zeros((2, 2))
    cm_test = np.zeros((2, 2))

    # Initialize histogram counts
    conf_thres = 0.75
    nbins = 20
    histogram_counts_dd = np.zeros((len(feature_df.columns), nbins))
    histogram_counts_naive = np.zeros((len(feature_df.columns), nbins))

    # Loop through seeds
    run = 0
    for seed in seeds:
        np.random.seed(seed)

        # Split data
        X_train, X_test, y_train, y_test = clean_data.split_data(
            feature_df, target_df_outliers_removed, train_amount, seed=seed
        )

        # Add outliers to test set
        if test_outliers:
            X_test = pd.concat([X_test, target_df_outliers])
            y_test = pd.concat([y_test, y_outliers])

        # Normalize data
        outliers_norm = clean_data.normalize_data(
            X_train, target_df_outliers, min_max=min_max_scale
        )[1]

        X_train_norm, X_test_norm = clean_data.normalize_data(
            X_train, X_test, min_max=min_max_scale
        )

        jaws_npas_pre_post_dropped_norm = [
            clean_data.normalize_data(
                X_train, jaws_npas_pre_post_dropped[i], min_max=min_max_scale
            )[1]
            for i in range(len(jaws_npas_pre_post_dropped))
        ]

        # Initialize network
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

        # Train network
        clf.fit(X_train_norm, y_train)

        # Save network
        with open(f"../data/neural_net/MLP_seed_{seed:02d}.pkl", "wb") as f:
            pickle.dump(clf, f)

        # Save training data
        X_train.to_csv(f"../data/neural_net/X_train_seed_{seed:02d}.csv")

        # Get network statistics
        predicts_train = clf.predict(X_train_norm)
        predicts_test = clf.predict(X_test_norm)
        probs_train = clf.predict_proba(X_train_norm)
        probs_test = clf.predict_proba(X_test_norm)

        # Update histograms
        col_count = 0
        for col in feature_df.columns:
            bins = np.histogram_bin_edges(target_df_outliers_removed[col], nbins)
            histogram_counts_dd[col_count, :] += np.histogram(
                X_train[(probs_train[:, 0] > conf_thres) & (y_train == 0)], bins=bins
            )[0] / len(seeds)
            histogram_counts_naive[col_count, :] += np.histogram(
                X_train[(probs_train[:, 1] > conf_thres) & (y_train == 1)], bins=bins
            )[0] / len(seeds)
            col_count += 1

        # Update outlier probabilities
        outlier_probabilities += clf.predict_proba(outliers_norm) / len(seeds)

        # Update accuracies
        accuracies[run, 0] = accuracy_score(y_train, predicts_train)
        accuracies[run, 1] = accuracy_score(y_test, predicts_test)
        precisions[run, :2], recalls[run, :2], _, _ = precision_recall_fscore_support(
            y_train, predicts_train
        )
        precisions[run, 2:], recalls[run, 2:], _, _ = precision_recall_fscore_support(
            y_test, predicts_test
        )

        # Update collective confusion matrix
        cm_train += confusion_matrix(y_train, predicts_train, labels=clf.classes_)
        cm_test += confusion_matrix(y_test, predicts_test, labels=clf.classes_)

        # Update JAWS and NPAS datasets
        jaws_npas_predicts = [clf.predict(x) for x in jaws_npas_pre_post_dropped_norm]
        jaws_npas_probs = [
            clf.predict_proba(x) for x in jaws_npas_pre_post_dropped_norm
        ]

        # Update JAWS and NPAS datasets by time chunk
        for i in range(len(jaws_npas_pre_post_dropped)):
            ### FIND PROBABILITIES FOR ALL CELLS
            jaws_npas_pre_post_probs[i][run, :, :] = jaws_npas_probs[i]

            ### FIND PREDICTED PROPORTIONS OF CLASSES
            # jaws predicted DD %
            jaws_npas_pre_post_predicts[i][run, 0] = len(
                jaws_npas_predicts[i][
                    (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                    & (jaws_npas_predicts[i] == 0)
                ]
            ) / len(jaws_npas_pre_post[i][jaws_npas_pre_post[i]["mouse"] == "JAWS"])
            # jaws predicted naive %
            jaws_npas_pre_post_predicts[i][run, 1] = len(
                jaws_npas_predicts[i][
                    (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                    & (jaws_npas_predicts[i] == 1)
                ]
            ) / len(jaws_npas_pre_post[i][jaws_npas_pre_post[i]["mouse"] == "JAWS"])
            # npas predicted DD %
            jaws_npas_pre_post_predicts[i][run, 2] = len(
                jaws_npas_predicts[i][
                    (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                    & (jaws_npas_predicts[i] == 0)
                ]
            ) / len(jaws_npas_pre_post[i][jaws_npas_pre_post[i]["mouse"] == "NPAS"])
            # npas predicted naive %
            jaws_npas_pre_post_predicts[i][run, 3] = len(
                jaws_npas_predicts[i][
                    (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                    & (jaws_npas_predicts[i] == 1)
                ]
            ) / len(jaws_npas_pre_post[i][jaws_npas_pre_post[i]["mouse"] == "NPAS"])

            ### FIND PREDICTED PROPORTIONS OF CLASSES BY MEDIAL
            jaws_npas_pre_post_predicts_medial[i][run, 0] = (
                len(
                    jaws_npas_predicts[i][
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_predicts[i] == 0)
                        & (jaws_npas_pre_post[i]["Medial"] == 1)
                    ]
                )
                / len(
                    jaws_npas_pre_post[i][
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_pre_post[i]["Medial"] == 1)
                    ]
                )
                if len(
                    jaws_npas_pre_post[i][
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_pre_post[i]["Medial"] == 1)
                    ]
                )
                > 0
                else 0
            )
            # jaws precicted naive %
            jaws_npas_pre_post_predicts_medial[i][run, 1] = (
                len(
                    jaws_npas_predicts[i][
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_predicts[i] == 1)
                        & (jaws_npas_pre_post[i]["Medial"] == 1)
                    ]
                )
                / len(
                    jaws_npas_pre_post[i][
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_pre_post[i]["Medial"] == 1)
                    ]
                )
                if len(
                    jaws_npas_pre_post[i][
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_pre_post[i]["Medial"] == 1)
                    ]
                )
                > 0
                else 0
            )
            # npas precicted DD %
            jaws_npas_pre_post_predicts_medial[i][run, 2] = len(
                jaws_npas_predicts[i][
                    (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                    & (jaws_npas_predicts[i] == 0)
                    & (jaws_npas_pre_post[i]["Medial"] == 1)
                ]
            ) / len(
                jaws_npas_pre_post[i][
                    (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                    & (jaws_npas_pre_post[i]["Medial"] == 1)
                ]
            )
            # npas precicted naive %
            jaws_npas_pre_post_predicts_medial[i][run, 3] = len(
                jaws_npas_predicts[i][
                    (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                    & (jaws_npas_predicts[i] == 1)
                    & (jaws_npas_pre_post[i]["Medial"] == 1)
                ]
            ) / len(
                jaws_npas_pre_post[i][
                    (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                    & (jaws_npas_pre_post[i]["Medial"] == 1)
                ]
            )

            ### FIND PREDICTED PROPORTIONS OF CLASSES BY NON MEDIAL
            jaws_npas_pre_post_predicts_non_medial[i][run, 0] = (
                len(
                    jaws_npas_predicts[i][
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_predicts[i] == 0)
                        & (jaws_npas_pre_post[i]["Medial"] == 0)
                    ]
                )
                / len(
                    jaws_npas_pre_post[i][
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_pre_post[i]["Medial"] == 0)
                    ]
                )
                if len(
                    jaws_npas_pre_post[i][
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_pre_post[i]["Medial"] == 0)
                    ]
                )
                > 0
                else 0
            )
            # jaws precicted naive %
            jaws_npas_pre_post_predicts_non_medial[i][run, 1] = (
                len(
                    jaws_npas_predicts[i][
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_predicts[i] == 1)
                        & (jaws_npas_pre_post[i]["Medial"] == 0)
                    ]
                )
                / len(
                    jaws_npas_pre_post[i][
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_pre_post[i]["Medial"] == 0)
                    ]
                )
                if len(
                    jaws_npas_pre_post[i][
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_pre_post[i]["Medial"] == 0)
                    ]
                )
                > 0
                else 0
            )
            # npas precicted DD %
            jaws_npas_pre_post_predicts_non_medial[i][run, 2] = len(
                jaws_npas_predicts[i][
                    (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                    & (jaws_npas_predicts[i] == 0)
                    & (jaws_npas_pre_post[i]["Medial"] == 0)
                ]
            ) / len(
                jaws_npas_pre_post[i][
                    (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                    & (jaws_npas_pre_post[i]["Medial"] == 0)
                ]
            )
            # npas precicted naive %
            jaws_npas_pre_post_predicts_non_medial[i][run, 3] = len(
                jaws_npas_predicts[i][
                    (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                    & (jaws_npas_predicts[i] == 1)
                    & (jaws_npas_pre_post[i]["Medial"] == 0)
                ]
            ) / len(
                jaws_npas_pre_post[i][
                    (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                    & (jaws_npas_pre_post[i]["Medial"] == 0)
                ]
            )

        run += 1

    # Plot results
    jaws_probs = [
        np.mean(jaws_npas_pre_post_probs[i][:, :, 0], axis=1)
        for i in range(len(jaws_npas_pre_post_probs))
    ]

    npas_probs = [
        jaws_npas_pre_post_probs[i][
            :,
            (jaws_npas_pre_post[i]["mouse"] == "NPAS"),
            0,
        ]
        for i in range(len(jaws_npas_pre_post_probs))
    ]

    fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    axis_count = 0
    for col in jaws_npas_pre_post[0].columns[0:12]:
        xvals = []
        yvals = []
        for i in range(4):
            xvals.extend(
                jaws_npas_pre_post[i + 1][jaws_npas_pre_post[i + 1]["mouse"] == "NPAS"][
                    col
                ]
            )
            yvals.extend(
                np.mean(
                    jaws_npas_pre_post_probs[i + 1][
                        :,
                        (jaws_npas_pre_post[i + 1]["mouse"] == "NPAS"),
                        0,
                    ],
                    axis=0,
                )
            )
        slope, intercept, r_value, pval_post, std_err = scipy.stats.linregress(
            xvals, yvals
        )
        slope, intercept, r_value, pval_pre, std_err = scipy.stats.linregress(
            jaws_npas_pre_post[0][jaws_npas_pre_post[0]["mouse"] == "NPAS"][col],
            np.mean(
                jaws_npas_pre_post_probs[0][
                    :,
                    (jaws_npas_pre_post[0]["mouse"] == "NPAS"),
                    0,
                ],
                axis=0,
            ),
        )
        sns.regplot(
            x=xvals, y=yvals, color="b", scatter_kws={"s": 8}, ax=axes[axis_count]
        )
        sns.regplot(
            x=jaws_npas_pre_post[0][jaws_npas_pre_post[0]["mouse"] == "NPAS"][col],
            y=np.mean(
                jaws_npas_pre_post_probs[0][
                    :,
                    (jaws_npas_pre_post[0]["mouse"] == "NPAS"),
                    0,
                ],
                axis=0,
            ),
            color="gray",
            scatter_kws={"s": 8},
            ax=axes[axis_count],
        )
        axes[axis_count].set_title(
            f"Pre pval: {pval_pre:.02f}\tPost pval: {pval_post:.02f}", fontsize=8
        )
        axes[axis_count].set_ylabel("DD Confidence")
        axes[axis_count].set_ylim([0, 1])
        axis_count += 1
    makeNice(axes)
    fig.savefig(f"../data/features_vs_prob_npas.pdf", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    axis_count = 0
    for col in jaws_npas_pre_post[0].columns[0:12]:
        xvals = []
        yvals = []
        for i in range(4):
            xvals.extend(
                jaws_npas_pre_post[i + 1][jaws_npas_pre_post[i + 1]["mouse"] == "JAWS"][
                    col
                ]
            )
            yvals.extend(
                np.mean(
                    jaws_npas_pre_post_probs[i + 1][
                        :,
                        (jaws_npas_pre_post[i + 1]["mouse"] == "JAWS"),
                        0,
                    ],
                    axis=0,
                )
            )
        slope, intercept, r_value, pval_post, std_err = scipy.stats.linregress(
            xvals, yvals
        )
        slope, intercept, r_value, pval_pre, std_err = scipy.stats.linregress(
            jaws_npas_pre_post[0][jaws_npas_pre_post[0]["mouse"] == "JAWS"][col],
            np.mean(
                jaws_npas_pre_post_probs[0][
                    :,
                    (jaws_npas_pre_post[0]["mouse"] == "JAWS"),
                    0,
                ],
                axis=0,
            ),
        )

        sns.regplot(
            x=xvals, y=yvals, color="b", scatter_kws={"s": 8}, ax=axes[axis_count]
        )
        sns.regplot(
            x=jaws_npas_pre_post[0][jaws_npas_pre_post[0]["mouse"] == "JAWS"][col],
            y=np.mean(
                jaws_npas_pre_post_probs[0][
                    :,
                    (jaws_npas_pre_post[0]["mouse"] == "JAWS"),
                    0,
                ],
                axis=0,
            ),
            color="gray",
            scatter_kws={"s": 8},
            ax=axes[axis_count],
        )
        axes[axis_count].set_title(
            f"Pre pval: {pval_pre:.02f}\tPost pval: {pval_post:.02f}", fontsize=8
        )
        axes[axis_count].set_ylabel("DD Confidence")
        axes[axis_count].set_ylim([0, 1])
        axis_count += 1
    makeNice(axes)
    fig.savefig(f"../data/features_vs_prob_jaws.pdf", bbox_inches="tight")
    plt.close()

    # Save data
    time_plot = ["pre", "30", "60", "90_120", "150_180_210"]
    for i in range(len(jaws_npas_pre_post_probs)):
        np.savetxt(
            f"../data/npas_probs_conf_all_{time_plot[i]}.csv",
            jaws_npas_pre_post_probs[i][
                :,
                (jaws_npas_pre_post[i]["mouse"] == "NPAS"),
                0,
            ],
            fmt="%f",
            delimiter=",",
            newline="\n",
        )
    time_plot = ["pre", "30", "60", "90_120", "150_180_210"]
    for i in range(len(jaws_npas_pre_post_probs)):
        np.savetxt(
            f"../data/jaws_probs_conf_all_{time_plot[i]}.csv",
            jaws_npas_pre_post_probs[i][
                :,
                (jaws_npas_pre_post[i]["mouse"] == "JAWS"),
                0,
            ],
            fmt="%f",
            delimiter=",",
            newline="\n",
        )

    for i in range(len(jaws_npas_pre_post)):
        jaws_npas_pre_post[i]["DD Probability"] = np.mean(
            jaws_npas_pre_post_probs[i][:, :, 0], axis=0
        )
        jaws_npas_pre_post[i]["Naive Probability"] = np.mean(
            jaws_npas_pre_post_probs[i][:, :, 1], axis=0
        )

    # Plot results
    fig, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(2) for j in range(2)]
    count = 0
    dd_prob_splits = [[0, 0.50], [0.50, 0.67], [0.67, 0.83], [0.83, 1]]
    dd_colors = ["gray", "lightblue", "b", "darkblue"]
    jaws_probs_all = np.zeros((len(jaws_npas_pre_post), 4))
    npas_probs_all = np.zeros((len(jaws_npas_pre_post), 4))
    for x in jaws_npas_pre_post:
        jaws_probs = np.zeros(4)  # 50-70, 70-90, 90+, Naive
        npas_probs = np.zeros(4)  # 50-70, 70-90, 90+, Naive
        jaws_x = x[x["mouse"] == "JAWS"]
        npas_x = x[x["mouse"] == "NPAS"]
        for i in range(4):
            jaws_probs[i] = len(
                jaws_x[
                    (jaws_x["DD Probability"] > dd_prob_splits[i][0])
                    & (jaws_x["DD Probability"] <= dd_prob_splits[i][1])
                ]
            )

            npas_probs[i] = len(
                npas_x[
                    (npas_x["DD Probability"] > dd_prob_splits[i][0])
                    & (npas_x["DD Probability"] <= dd_prob_splits[i][1])
                ]
            )

        print("jaws:\t", jaws_probs)
        print("npas:\t", npas_probs)

        jaws_probs = jaws_probs / np.sum(jaws_probs)
        npas_probs = npas_probs / np.sum(npas_probs)

        axes[0].scatter(
            np.random.rand(len(jaws_x)) * 0.25 + 0.25 + count,
            jaws_x["DD Probability"],
            color="b",
            s=2,
        )
        axes[1].scatter(
            np.random.rand(len(npas_x)) * 0.25 + 0.25 + count,
            npas_x["DD Probability"],
            color="r",
            s=2,
        )

        jaws_probs_all[count] = jaws_probs
        npas_probs_all[count] = npas_probs

        for i in range(4):
            axes[2].bar(
                count, jaws_probs[i], bottom=np.sum(jaws_probs[0:i]), color=dd_colors[i]
            )
            axes[3].bar(
                count, npas_probs[i], bottom=np.sum(npas_probs[0:i]), color=dd_colors[i]
            )
        count += 1
    for i in [2, 3]:
        axes[i].set_xticks(np.arange(5))
        axes[i].set_xticklabels(
            ["Pre", "30min", "60min", "90-120min", "150-210min"],
            fontsize=6,
            rotation=15,
        )
        axes[i].set_ylabel(f"{'JAWS' if i == 2 else 'NPAS'} DD Proportions")
    for i in range(2):
        axes[i].hlines(0.5, 0, 5, ls="dashed", lw=0.5, color="k")
        axes[i].set_xticks(np.arange(0.375, 5.25, 1))
        axes[i].set_xticklabels(
            ["Pre", "30min", "60min", "90-120min", "150-210min"],
            fontsize=6,
            rotation=15,
        )
        axes[i].set_ylabel(f"{'JAWS' if i == 0 else 'NPAS'} DD Probability")
    makeNice(axes)
    add_fig_labels(axes)
    fig.savefig("../data/neural_net/jaws_npas_summary.pdf", bbox_inches="tight")
    plt.close()

    # Save data
    np.savetxt(
        "../data/neural_net/jaws_pre_post_summary.csv",
        jaws_probs_all,
        newline="\n",
        delimiter=",",
        fmt="%f",
        header="naive, weak DD, DD, strong DD",
    )
    np.savetxt(
        "../data/neural_net/npas_pre_post_summary.csv",
        npas_probs_all,
        newline="\n",
        delimiter=",",
        fmt="%f",
        header="naive, weak DD, DD, strong DD",
    )

    all_pre_post = pd.concat(jaws_npas_pre_post)
    all_pre_post.to_csv("../data/neural_net/jaws_npas_pre_post.csv")
    print(
        "Train/Test Accuracies (avg +/- sem):\t",
        np.mean(accuracies, axis=0),
        scipy.stats.sem(accuracies, axis=0),
    )

    # Plot histograms
    fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    count = 0
    for col in feature_df.columns:
        bins = np.histogram_bin_edges(target_df_outliers_removed[col], bins=nbins)
        feat_mean_dd = np.mean(target_df[target_df["Type"] == 0][col])
        feat_std_dd = np.std(target_df[target_df["Type"] == 0][col])
        feat_mean_naive = np.mean(target_df[target_df["Type"] == 1][col])
        feat_std_naive = np.std(target_df[target_df["Type"] == 1][col])
        axes[count].bar(
            x=np.arange(nbins),
            height=histogram_counts_dd[count] / np.sum(histogram_counts_dd[count]),
            color="b",
            alpha=0.5,
            width=0.8,
            edgecolor="w",
        )
        axes[count].bar(
            x=np.arange(nbins),
            height=histogram_counts_naive[count]
            / np.sum(histogram_counts_naive[count]),
            color="r",
            alpha=0.5,
            width=0.8,
            edgecolor="w",
        )
        ylims = axes[count].get_ylim()
        axes[count].set_xticks(np.arange(0, nbins + 1, 5))
        axes[count].set_xticklabels(
            [
                f"{x:.02E}" if count == 2 or count == 3 else f"{x:.02f}"
                for x in bins[::5]
            ],
            rotation=25,
        )
        axes[count].vlines(
            nbins * feat_mean_dd / (bins[-1] - bins[0]),
            ylims[0],
            ylims[1],
            color="b",
            ls="dashed",
        )
        axes[count].vlines(
            nbins * (feat_mean_dd + 3 * feat_std_dd) / (bins[-1] - bins[0]),
            ylims[0],
            ylims[1],
            color="b",
            ls="dotted",
        )
        axes[count].vlines(
            nbins * feat_mean_naive / (bins[-1] - bins[0]),
            ylims[0],
            ylims[1],
            color="r",
            ls="dashed",
        )
        axes[count].vlines(
            nbins * (feat_mean_naive + 3 * feat_std_naive) / (bins[-1] - bins[0]),
            ylims[0],
            ylims[1],
            color="r",
            ls="dotted",
        )
        axes[count].set_xlabel(col)
        axes[count].set_ylim(ylims)
        count += 1
    makeNice(axes)
    fig.savefig("../data/conf_thres_features.pdf", bbox_inches="tight")
    plt.close()

    # Plot outliers
    outliers_correct_naive = target_df_outliers.loc[
        (outlier_probabilities[:, 1] > 0.5) & (y_outliers == 1), :
    ]

    outliers_correct_dd = target_df_outliers.loc[
        (outlier_probabilities[:, 0] > 0.5) & (y_outliers == 0), :
    ]

    outliers_incorrect_naive = target_df_outliers.loc[
        (outlier_probabilities[:, 1] < 0.5) & (y_outliers == 1), :
    ]

    outliers_incorrect_dd = target_df_outliers.loc[
        (outlier_probabilities[:, 0] < 0.5) & (y_outliers == 0), :
    ]

    fig, ax = plt.subplots(2, 2, figsize=(20, 12), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(2) for j in range(2)]
    count = 0
    titles = ["Correct DD", "Correct Naive", "Incorrect DD", "Incorrect Naive"]
    for df in [
        outliers_correct_dd,
        outliers_correct_naive,
        outliers_incorrect_dd,
        outliers_incorrect_naive,
    ]:

        target_df_tmp = target_df[target_df["Type"] == count % 2]

        norm_df = clean_data.normalize_data(
            target_df_tmp.drop(columns=["Type"]),
            df,
            min_max=min_max_scale,
        )[1]
        norm_df = norm_df.transpose()
        sns.heatmap(
            norm_df,
            cmap="rocket_r",
            annot=True,
            vmin=3,
            vmax=7,
            ax=axes[count],
            annot_kws={"fontsize": 6},
            mask=norm_df < 3,
            linecolor="gray",
            linewidths=0.05,
        )
        axes[count].set_title(titles[count], fontsize=6)
        count += 1
    add_fig_labels(axes)
    fig.savefig("../data/neural_net/outliers_heatmaps.pdf")
    plt.close()

    # Plot training results (confusion matrix)
    fig, ax = plt.subplots(3, 2, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(3) for j in range(2)]

    fig2, ax2 = plt.subplots(1, 1, figsize=(3, 2), dpi=300, tight_layout=True)
    fig3, ax3 = plt.subplots(1, 1, figsize=(3, 2), dpi=300, tight_layout=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=[0, 1])

    disp.plot(ax=axes[0], im_kw={"aspect": "auto"}, values_format=".0f")
    axes[0].set_title(
        f"Accuracy: {np.mean(accuracies[:,0],axis=0):.02f} +/- {np.std(accuracies[:,0],axis=0)}\nPrecision: {np.mean(precisions[:,1]):.02f}, {np.mean(precisions[:,0]):.02f}\nRecall: {np.mean(recalls[:,1]):.02f}, {np.mean(recalls[:,0]):.02f}",
        fontsize=10,
    )

    disp.plot(ax=ax2, im_kw={"aspect": "auto"}, values_format=".0f")
    ax2.set_title(
        f"Accuracy: {np.mean(accuracies[:,0],axis=0):.02f}",
        fontsize=8,
    )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Depleted", "Control"])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Depleted", "Control"])
    makeNice(ax2)
    for i in ["left", "right", "top", "bottom"]:
        ax2.spines[i].set_visible(True)
    fig2.savefig(
        f"../data/neural_net/conf_mat_train.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()

    # Plot testing results (confusion matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=[0, 1])
    disp.plot(ax=axes[1], im_kw={"aspect": "auto"}, values_format=".0f")
    axes[1].set_title(
        f"Accuracy: {np.mean(accuracies[:,1],axis=0):.02f}\nPrecision: {np.mean(precisions[:,3]):.02f}, {np.mean(precisions[:,2]):.02f}\nRecall: {np.mean(recalls[:,3]):.02f}, {np.mean(recalls[:,2]):.02f}",
    )

    disp.plot(ax=ax3, im_kw={"aspect": "auto"}, values_format=".0f")
    ax3.set_title(
        f"Accuracy: {np.mean(accuracies[:,1],axis=0):.02f}",
        fontsize=8,
    )
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Depleted", "Control"])
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Depleted", "Control"])
    makeNice(ax3)
    for i in ["left", "right", "top", "bottom"]:
        ax3.spines[i].set_visible(True)

    fig3.savefig(
        f"../data/neural_net/conf_mat_test.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()

    # Plot JAWS and NPAS results
    jaws_predicts = np.array(
        [jaws_npas_pre_post_predicts[i][:, 0] for i in range(len(jaws_npas_pre_post))]
    )
    npas_predicts = np.array(
        [jaws_npas_pre_post_predicts[i][:, 2] for i in range(len(jaws_npas_pre_post))]
    )
    np.savetxt(
        "../data/jaws_predicts_runs_data.csv",
        jaws_predicts,
        delimiter=",",
        fmt="%f",
        newline="\n",
    )
    np.savetxt(
        "../data/npas_predicts_runs_data.csv",
        npas_predicts,
        delimiter=",",
        fmt="%f",
        newline="\n",
    )

    # Plot predicted DD percentages
    axes[2].errorbar(
        np.arange(len(jaws_npas_pre_post)),
        [
            np.mean(jaws_npas_pre_post_predicts[i][:, 0])
            for i in range(len(jaws_npas_pre_post))
        ],
        yerr=[
            scipy.stats.sem(jaws_npas_pre_post_predicts[i][:, 0])
            for i in range(len(jaws_npas_pre_post))
        ],
        color="b",
        lw=0.5,
        markersize=4,
        capsize=4,
        marker="o",
    )

    axes[2].errorbar(
        np.arange(len(jaws_npas_pre_post)),
        [
            np.mean(jaws_npas_pre_post_predicts[i][:, 2])
            for i in range(len(jaws_npas_pre_post))
        ],
        yerr=[
            scipy.stats.sem(jaws_npas_pre_post_predicts[i][:, 2])
            for i in range(len(jaws_npas_pre_post))
        ],
        color="r",
        lw=0.5,
        markersize=4,
        capsize=4,
        marker="o",
    )

    # Plot predicted DD percentages for medial units
    axes[3].errorbar(
        np.arange(len(jaws_npas_pre_post)),
        [
            np.mean(jaws_npas_pre_post_predicts_medial[i][:, 0])
            for i in range(len(jaws_npas_pre_post))
        ],
        yerr=[
            scipy.stats.sem(jaws_npas_pre_post_predicts_medial[i][:, 0])
            for i in range(len(jaws_npas_pre_post))
        ],
        color="b",
        lw=0.5,
        markersize=4,
        capsize=4,
        marker="o",
        ls="dashed",
    )

    axes[3].errorbar(
        np.arange(len(jaws_npas_pre_post), 2 * len(jaws_npas_pre_post)),
        [
            np.mean(jaws_npas_pre_post_predicts_medial[i][:, 2])
            for i in range(len(jaws_npas_pre_post))
        ],
        yerr=[
            scipy.stats.sem(jaws_npas_pre_post_predicts_medial[i][:, 2])
            for i in range(len(jaws_npas_pre_post))
        ],
        color="r",
        lw=0.5,
        markersize=4,
        capsize=4,
        marker="o",
        ls="dashed",
    )

    # Plot predicted DD percentages for non medial units
    axes[3].errorbar(
        np.arange(len(jaws_npas_pre_post)),
        [
            np.mean(jaws_npas_pre_post_predicts_non_medial[i][:, 0])
            for i in range(len(jaws_npas_pre_post))
        ],
        yerr=[
            scipy.stats.sem(jaws_npas_pre_post_predicts_non_medial[i][:, 0])
            for i in range(len(jaws_npas_pre_post))
        ],
        color="b",
        lw=0.5,
        markersize=4,
        capsize=4,
        marker="o",
        ls="dotted",
    )

    axes[3].errorbar(
        np.arange(len(jaws_npas_pre_post), 2 * len(jaws_npas_pre_post)),
        [
            np.mean(jaws_npas_pre_post_predicts_non_medial[i][:, 2])
            for i in range(len(jaws_npas_pre_post))
        ],
        yerr=[
            scipy.stats.sem(jaws_npas_pre_post_predicts_non_medial[i][:, 2])
            for i in range(len(jaws_npas_pre_post))
        ],
        color="r",
        lw=0.5,
        markersize=4,
        capsize=4,
        marker="o",
        ls="dotted",
    )

    # Plot DD confidence for JAWS and NPAS
    axes[4].errorbar(
        np.arange(len(jaws_npas_pre_post)),
        np.mean(
            [
                np.mean(
                    jaws_npas_pre_post_probs[i][
                        :, jaws_npas_pre_post[i]["mouse"] == "JAWS", 0
                    ],
                    axis=1,
                )
                for i in range(len(jaws_npas_pre_post))
            ],
            axis=1,
        ),
        yerr=scipy.stats.sem(
            [
                np.mean(
                    jaws_npas_pre_post_probs[i][
                        :, jaws_npas_pre_post[i]["mouse"] == "JAWS", 0
                    ],
                    axis=1,
                )
                for i in range(len(jaws_npas_pre_post))
            ],
            axis=1,
        ),
        color="b",
        lw=0.5,
        markersize=4,
        capsize=4,
        marker="o",
    )

    axes[4].errorbar(
        np.arange(len(jaws_npas_pre_post)),
        np.mean(
            [
                np.mean(
                    jaws_npas_pre_post_probs[i][
                        :, jaws_npas_pre_post[i]["mouse"] == "NPAS", 0
                    ],
                    axis=1,
                )
                for i in range(len(jaws_npas_pre_post))
            ],
            axis=1,
        ),
        yerr=scipy.stats.sem(
            [
                np.mean(
                    jaws_npas_pre_post_probs[i][
                        :, jaws_npas_pre_post[i]["mouse"] == "NPAS", 0
                    ],
                    axis=1,
                )
                for i in range(len(jaws_npas_pre_post))
            ],
            axis=1,
        ),
        color="r",
        lw=0.5,
        markersize=4,
        capsize=4,
        marker="o",
    )

    # Plot DD confidence for medial units
    axes[5].errorbar(
        np.arange(len(jaws_npas_pre_post)),
        np.mean(
            [
                np.mean(
                    jaws_npas_pre_post_probs[i][
                        :,
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_pre_post[i]["Medial"] == 1),
                        0,
                    ],
                    axis=1,
                )
                for i in range(len(jaws_npas_pre_post))
            ],
            axis=1,
        ),
        yerr=scipy.stats.sem(
            [
                np.mean(
                    jaws_npas_pre_post_probs[i][
                        :,
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_pre_post[i]["Medial"] == 1),
                        0,
                    ],
                    axis=1,
                )
                for i in range(len(jaws_npas_pre_post))
            ],
            axis=1,
        ),
        color="b",
        lw=0.5,
        markersize=4,
        capsize=4,
        marker="o",
        ls="dashed",
    )

    axes[5].errorbar(
        np.arange(len(jaws_npas_pre_post), 2 * len(jaws_npas_pre_post)),
        np.mean(
            [
                np.mean(
                    jaws_npas_pre_post_probs[i][
                        :,
                        (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                        & (jaws_npas_pre_post[i]["Medial"] == 1),
                        0,
                    ],
                    axis=1,
                )
                for i in range(len(jaws_npas_pre_post))
            ],
            axis=1,
        ),
        yerr=scipy.stats.sem(
            [
                np.mean(
                    jaws_npas_pre_post_probs[i][
                        :,
                        (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                        & (jaws_npas_pre_post[i]["Medial"] == 1),
                        0,
                    ],
                    axis=1,
                )
                for i in range(len(jaws_npas_pre_post))
            ],
            axis=1,
        ),
        color="r",
        lw=0.5,
        markersize=4,
        capsize=4,
        marker="o",
        ls="dashed",
    )

    # Plot DD confidence for non medial units
    axes[5].errorbar(
        np.arange(len(jaws_npas_pre_post)),
        np.mean(
            [
                np.mean(
                    jaws_npas_pre_post_probs[i][
                        :,
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_pre_post[i]["Medial"] == 0),
                        0,
                    ],
                    axis=1,
                )
                for i in range(len(jaws_npas_pre_post))
            ],
            axis=1,
        ),
        yerr=scipy.stats.sem(
            [
                np.mean(
                    jaws_npas_pre_post_probs[i][
                        :,
                        (jaws_npas_pre_post[i]["mouse"] == "JAWS")
                        & (jaws_npas_pre_post[i]["Medial"] == 0),
                        0,
                    ],
                    axis=1,
                )
                for i in range(len(jaws_npas_pre_post))
            ],
            axis=1,
        ),
        color="b",
        lw=0.5,
        markersize=4,
        capsize=4,
        marker="o",
        ls="dotted",
    )

    axes[5].errorbar(
        np.arange(len(jaws_npas_pre_post), 2 * len(jaws_npas_pre_post)),
        np.mean(
            [
                np.mean(
                    jaws_npas_pre_post_probs[i][
                        :,
                        (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                        & (jaws_npas_pre_post[i]["Medial"] == 0),
                        0,
                    ],
                    axis=1,
                )
                for i in range(len(jaws_npas_pre_post))
            ],
            axis=1,
        ),
        yerr=scipy.stats.sem(
            [
                np.mean(
                    jaws_npas_pre_post_probs[i][
                        :,
                        (jaws_npas_pre_post[i]["mouse"] == "NPAS")
                        & (jaws_npas_pre_post[i]["Medial"] == 0),
                        0,
                    ],
                    axis=1,
                )
                for i in range(len(jaws_npas_pre_post))
            ],
            axis=1,
        ),
        color="r",
        lw=0.5,
        markersize=4,
        capsize=4,
        marker="o",
        ls="dotted",
    )

    # Adjust legends
    axes[2].legend(fancybox=False, frameon=False, fontsize=6)
    axes[3].legend(fancybox=False, frameon=False, fontsize=6, ncols=2)

    # Set axis labels
    for i in range(2):
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Depleted", "Control"])
        axes[i].set_yticks([0, 1])
        axes[i].set_yticklabels(["Depleted", "Control"])

    # Set axis labels
    for i in [2, 3, 4, 5]:
        axes[i].set_ylabel("Predicted DD %" if i < 4 else "DD Confidence")
        axes[i].set_xticks([0, 1, 2, 3])
        xlim = axes[i].get_xlim()

        if i == 3 or i == 5:
            axes[i].set_xticks(np.arange(len(jaws_npas_pre_post) * 2))
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
            axes[i].set_xticks(np.arange(len(jaws_npas_pre_post)))
            axes[i].set_xticklabels(
                [
                    f"Pre",
                    30,
                    60,
                    "90-120",
                    "150-210",
                ]
            )
    [axes[i].grid(lw=0.1, alpha=0.5) for i in range(2, len(axes))]

    makeNice(axes[2:], labelsize=6)
    add_fig_labels(axes)
    fig.savefig(
        f"../data/neural_net/motor_rescue_conf_mat.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()

    # Show plots
    if show:
        run_cmd("open ../data/neural_net/motor_rescue_conf_mat.pdf")
        run_cmd("open ../data/neural_net/conf_mat_train.pdf")
        run_cmd("open ../data/neural_net/conf_mat_test.pdf")
        run_cmd("open ../data/neural_net/outliers_heatmaps.pdf")
        run_cmd("open ../data/conf_thres_features.pdf")
        run_cmd("open ../data/neural_net/jaws_npas_summary.pdf")


def feature_importance_clustered(
    target_df,
    feature_array,
    outlier_dict,
    train_amount,
    seeds=np.arange(15),
    show=False,
):
    """
    Determine important features for MLP networks

    Parameters
    ----------
    \t target_df (dataframe): dataframe containing the target data

    \t feature_array (list): list of columns to use as features

    \t outlier_dict (dict): dictionary containing the outlier thresholds for features

    \t train_amount=0.8 (float): proportion of data to use for training

    \t seeds=np.arange(15) (list): list of random seeds to use for training

    \t show=False (bool): whether to show the plots
    """

    # Remove outliers and create target data frame
    target_df = clean_data.remove_outliers_by_group_zscore_independent(
        target_df[target_df["Type"] == 1],
        target_df[target_df["Type"] == 0],
        outlier_dict,
    )

    # Create feature dataframe
    feature_df = target_df.iloc[:, feature_array]

    # Create labels for plotting
    xlabels_cluster = [
        "None",
        "Cluster I",
        "Cluster II",
        "Cluster III",
        "Cluster IV",
        "Cluster V",
        "Cluster VI",
        "Cluster VII",
    ]

    # values to track for cluster performance
    cluster_acc_scores = np.zeros((len(xlabels_cluster) + 2, len(seeds)))
    cluster_probs_dd = np.zeros((len(xlabels_cluster) + 2, len(seeds)))
    cluster_probs_naive = np.zeros((len(xlabels_cluster) + 2, len(seeds)))
    train_probs_dd = np.zeros((len(xlabels_cluster) + 2, len(seeds)))
    train_probs_naive = np.zeros((len(xlabels_cluster) + 2, len(seeds)))
    test_probs_dd = np.zeros((len(xlabels_cluster) + 2, len(seeds)))
    test_probs_naive = np.zeros((len(xlabels_cluster) + 2, len(seeds)))
    train_acc_scores = np.zeros((len(xlabels_cluster) + 2, len(seeds)))
    test_acc_scores = np.zeros((len(xlabels_cluster) + 2, len(seeds)))

    # Define clusters for correlation matrix
    cluster_I = ["Delta Power"]
    cluster_II = [
        "Percent of Spikes in Bursts",
        "Percent Time Bursting",
        "CV",
        "Burst Firing Rate Increase",
    ]
    cluster_III = ["Num Bursts"]
    cluster_IV = ["FR", "Avg Burst Firing Rate", "Non Bursting Firing Rate"]
    cluster_V = ["Beta Power"]
    cluster_VI = ["Avg Interburst Interval"]
    cluster_VII = ["Avg Burst Duration"]
    cluster_I_II = [
        "Delta Power",
        "Percent of Spikes in Bursts",
        "Percent Time Bursting",
        "CV",
        "Burst Firing Rate Increase",
    ]

    except_I_II = [
        "FR",
        "Avg Burst Firing Rate",
        "Non Bursting Firing Rate",
        "Num Bursts",
        "Beta Power",
        "Avg Interburst Interval",
        "Avg Burst Duration",
    ]

    # Create list of clusters to evaluate
    cluster_runs = [
        cluster_I,
        cluster_II,
        cluster_III,
        cluster_IV,
        cluster_V,
        cluster_VI,
        cluster_VII,
        cluster_I_II,
        except_I_II,
    ]

    # Run through all MLP seeds and all data
    run = 0
    for i in seeds:

        # Set Seed
        np.random.seed(i)

        # Split data
        X_train, X_test, y_train, y_test = clean_data.split_data(
            feature_df, target_df, train_amount, seed=i
        )

        # Normalize training and testing data
        X_train_norm, X_test_norm = clean_data.normalize_data(
            X_train, X_test, min_max=False
        )

        # Normalize entire feature dataframe
        _, all_norm = clean_data.normalize_data(X_train, feature_df, min_max=False)

        # Create MLP based on seed, same hyperparameters as predict_motor_rescue MLPs
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

        # Fit MLP to normalized training data
        clf.fit(X_train_norm, y_train)

        # Update metrics based on seed
        cluster_acc_scores[run, i] = accuracy_score(
            target_df["Type"], clf.predict(all_norm)
        )

        # Update metrics based on seed
        train_acc_scores[run, i] = accuracy_score(y_train, clf.predict(X_train_norm))

        test_acc_scores[run, i] = accuracy_score(y_test, clf.predict(X_test_norm))

        train_probs = clf.predict_proba(X_train_norm)

        train_probs_dd[run, i] = np.mean(train_probs[y_train == 0, 0])
        train_probs_naive[run, i] = np.mean(train_probs[y_train == 1, 0])

        test_probs = clf.predict_proba(X_test_norm)

        test_probs_dd[run, i] = np.mean(test_probs[y_test == 0, 0])
        test_probs_naive[run, i] = np.mean(test_probs[y_test == 1, 0])

        all_probs = clf.predict_proba(all_norm)

        cluster_probs_dd[run, i] = np.mean(all_probs[target_df["Type"] == 0, 0])
        cluster_probs_naive[run, i] = np.mean(all_probs[target_df["Type"] == 1, 0])

    # Run through all MLP seeds and each cluster
    run += 1
    features = feature_df.columns
    for cr in cluster_runs:
        cols = []

        # Get columns to remove
        for i in range(len(features)):
            if features[i] not in cr:
                cols.append(i)

        # Create temporary dataframes
        tmp_feature_df = pd.concat(
            [
                target_df.iloc[:, cols],  # target_df.iloc[:, [entry]],  #
            ],
        )

        tmp_target_df = pd.concat(
            [target_df],
        )

        # Run through all MLP seeds
        for i in seeds:
            np.random.seed(i)

            # Split data
            X_train, X_test, y_train, y_test = clean_data.split_data(
                tmp_feature_df, tmp_target_df, train_amount, seed=i
            )

            # Normalize training and testing data
            X_train_norm, X_test_norm = clean_data.normalize_data(
                X_train, X_test, min_max=False
            )

            # Normalize entire feature dataframe
            _, all_norm = clean_data.normalize_data(
                X_train, tmp_feature_df, min_max=False
            )

            # Create MLP based on seed, same hyperparameters as predict_motor_rescue MLPs
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

            # Fit MLP to normalized training data
            clf.fit(X_train_norm, y_train)

            # Update metrics based on seed
            cluster_acc_scores[run, i] = accuracy_score(
                tmp_target_df["Type"], clf.predict(all_norm)
            )

            train_acc_scores[run, i] = accuracy_score(
                y_train, clf.predict(X_train_norm)
            )

            test_acc_scores[run, i] = accuracy_score(y_test, clf.predict(X_test_norm))

            train_probs = clf.predict_proba(X_train_norm)

            train_probs_dd[run, i] = np.mean(train_probs[y_train == 0, 0])
            train_probs_naive[run, i] = np.mean(train_probs[y_train == 1, 0])

            test_probs = clf.predict_proba(X_test_norm)

            test_probs_dd[run, i] = np.mean(test_probs[y_test == 0, 0])
            test_probs_naive[run, i] = np.mean(test_probs[y_test == 1, 0])

            all_probs = clf.predict_proba(all_norm)

            cluster_probs_dd[run, i] = np.mean(all_probs[target_df["Type"] == 0, 0])
            cluster_probs_naive[run, i] = np.mean(all_probs[target_df["Type"] == 1, 0])
        run += 1

    # Plot accuracy scores of clusters
    fig3, ax3 = plt.subplots(1, 1, figsize=(3, 3), dpi=300, tight_layout=True)
    ax3.errorbar(
        np.arange(len(xlabels_cluster)),
        np.mean(cluster_acc_scores[:-2, :], axis=1),
        yerr=scipy.stats.sem(cluster_acc_scores[:-2, :], axis=1),
        color="k",
        capsize=4,
        marker="o",
        lw=0.5,
        markersize=4,
        label="clusters",
    )

    print(np.mean(cluster_acc_scores[:-2, :], axis=1))
    print(scipy.stats.sem(cluster_acc_scores[:-2, :]))

    ax3.hlines(
        np.mean(cluster_acc_scores[-2, :]), 0, len(xlabels_cluster), lw=0.5, color="k"
    )
    ax3.hlines(
        np.mean(cluster_acc_scores[-1, :]),
        0,
        len(xlabels_cluster),
        lw=0.5,
        color="gray",
    )
    print(np.mean(cluster_acc_scores[-2, :]), np.mean(cluster_acc_scores[-1, :]))
    ax3.set_xticks(np.arange(len(xlabels_cluster)))
    ax3.set_xticklabels(xlabels_cluster, rotation=90, fontsize=6)
    ax3.grid(visible=False, which="both")
    ax3.set_ylabel("Accuracy")
    ax3.set_xlabel("Features Removed")
    makeNice(ax3)
    fig3.savefig(
        f"../data/neural_net/acc_train_test_removal_clustered.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # Plot DD probabilitie scores of clusters
    fig3, ax3 = plt.subplots(1, 1, figsize=(3, 3), dpi=300, tight_layout=True)
    ax3.errorbar(
        np.arange(len(xlabels_cluster)),
        np.mean(cluster_probs_dd[:-2, :], axis=1),
        yerr=scipy.stats.sem(cluster_probs_dd[:-2, :], axis=1),
        color="k",
        capsize=4,
        marker="o",
        lw=0.5,
        markersize=4,
        label="clusters",
    )

    ax3.hlines(
        np.mean(cluster_probs_dd[-2, :]), 0, len(xlabels_cluster), lw=0.5, color="k"
    )
    ax3.hlines(
        np.mean(cluster_probs_dd[-1, :]), 0, len(xlabels_cluster), lw=0.5, color="gray"
    )
    print(np.mean(cluster_probs_dd[-2, :]), np.mean(cluster_probs_dd[-1, :]))
    ax3.set_xticks(np.arange(len(xlabels_cluster)))
    ax3.set_xticklabels(xlabels_cluster, rotation=90, fontsize=6)
    ax3.grid(visible=False, which="both")
    ax3.set_ylabel("Confidence")
    ax3.set_xlabel("Features Removed")
    makeNice(ax3)
    fig3.savefig(
        f"../data/neural_net/probs_train_test_removal_clustered.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # Rearrange features for correlation matrix
    feature_df = feature_df[
        [
            "Delta Power",
            "Percent of Spikes in Bursts",
            "Percent Time Bursting",
            "CV",
            "Burst Firing Rate Increase",
            "Num Bursts",
            "FR",
            "Avg Burst Firing Rate",
            "Non Bursting Firing Rate",
            "Beta Power",
            "Avg Interburst Interval",
            "Avg Burst Duration",
        ]
    ]

    # Plot correlation matrix
    corr_mat = feature_df.corr()
    fig2, ax2 = plt.subplots(1, 1, figsize=(3, 3), dpi=300, tight_layout=True)
    sns.heatmap(
        corr_mat,  # feature_df.corr(),
        cmap="coolwarm",
        ax=ax2,
        vmin=-1,
        vmax=1,
        fmt=".2f",
        annot=False,
        annot_kws={"fontsize": 4},
        # mask=np.triu(feature_df.corr()),
        # cbar_kws={"shrink": 0.5},
        cbar=True,
        cbar_kws={"orientation": "vertical"},
    )
    makeNice(ax2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig2.savefig(
        f"../data/neural_net/corr_mat_single_plot2.pdf",
        bbox_inches="tight",
    )
    plt.close()
