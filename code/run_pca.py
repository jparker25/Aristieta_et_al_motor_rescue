"""
run_pca.py

Performs PCA analysis on training data and evaluates different classification methods in PCA space.

Author: John E. Parker (2024)
"""

# python modules
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import pickle
import sys

# user modules
from helpers import *
import clean_data


def run_train_test_pca(
    target_df, feature_array, outlier_dict, seeds=np.arange(15), train_amount=0.8
):
    """
    Runs PCA and evaluates classification methods in PCA space for figures. Generates plots of results.

    Parameters
    ----------
    \t target_df (dataframe) : dataframe of all data

    \t feature_array (list) : list of feature indices to use in PCA

    \t outlier_dict (dict) : dictionary of outlier thresholds for each feature

    \t seeds=np.arange(15) (list) : list of random seeds to use for MLPs

    \t train_amount=0.8 (float) : fraction of data to use for training

    """

    # Remove outliers and return new dataframe
    target_df = clean_data.remove_outliers_by_group_zscore_independent(
        target_df[target_df["Type"] == 1],
        target_df[target_df["Type"] == 0],
        outlier_dict,
    )

    # Create feature dataframe
    feature_df = target_df.iloc[:, feature_array]

    acc_scores = np.zeros((len(seeds), 2))
    for seedi in seeds:
        np.random.seed(seedi)

        # Split data
        X_train, X_test, y_train, y_test = clean_data.split_data(
            feature_df, target_df, train_amount, seed=seedi
        )

        # Normalize training and testing data
        X_train_norm, X_test_norm = clean_data.normalize_data(
            X_train, X_test, min_max=False
        )

        # Set up and perform PCA on feature dataframe
        pca = PCA(n_components=len(feature_array))
        X_pca = pca.fit_transform(X_train_norm)
        test_X_pca = pca.transform(X_test_norm)

        # Create DD and Naive PCA variables
        X_pca_dd = X_pca[y_train == 0, :]
        X_pca_naive = X_pca[y_train == 1, :]

        # Determine centroids from PCA
        dd_centroid = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(X_pca_dd)
        naive_centroid = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(
            X_pca_naive
        )

        # Detemine nearest centroid by L2 norm
        centroid_match_train = np.zeros(len(X_pca))
        for i in range(len(X_pca)):
            dd_dist = np.linalg.norm(X_pca[i, :] - dd_centroid.cluster_centers_)
            naive_dist = np.linalg.norm(X_pca[i, :] - naive_centroid.cluster_centers_)
            if naive_dist < dd_dist:
                centroid_match_train[i] = 1

        # Detemine nearest centroid by L2 norm
        centroid_match_test = np.zeros(len(test_X_pca))
        for i in range(len(test_X_pca)):
            dd_dist = np.linalg.norm(test_X_pca[i, :] - dd_centroid.cluster_centers_)
            naive_dist = np.linalg.norm(
                test_X_pca[i, :] - naive_centroid.cluster_centers_
            )
            if naive_dist < dd_dist:
                centroid_match_test[i] = 1

        # Print accuracy by K-means centroid clustering
        acc_scores[seedi, 0] = accuracy_score(y_train, centroid_match_train)
        acc_scores[seedi, 1] = accuracy_score(y_test, centroid_match_test)
        print("PCA K-Means Cluster Acc:\t", acc_scores[seedi, 0], acc_scores[seedi, 1])
    print(
        f"PCA K-means Cluster Acc:\t{np.mean(acc_scores[:,0]):.4f}+/-{np.std(acc_scores[:,0]):.4f}\t{np.mean(acc_scores[:,1]):.4f}+/-{np.std(acc_scores[:,1]):.4f}"
    )
    # sys.exit()


def run_pca_for_figures(
    target_df, feature_array, outlier_dict, seeds=np.arange(15), train_amount=0.8
):
    """
    Runs PCA and evaluates classification methods in PCA space for figures. Generates plots of results.

    Parameters
    ----------
    \t target_df (dataframe) : dataframe of all data

    \t feature_array (list) : list of feature indices to use in PCA

    \t outlier_dict (dict) : dictionary of outlier thresholds for each feature

    \t seeds=np.arange(15) (list) : list of random seeds to use for MLPs

    \t train_amount=0.8 (float) : fraction of data to use for training

    """

    run_train_test_pca(
        target_df, feature_array, outlier_dict, seeds=seeds, train_amount=train_amount
    )
    # sys.exit()

    # Remove outliers and return new dataframe
    target_df_outliers_removed = clean_data.remove_outliers_by_group_zscore_independent(
        target_df[target_df["Type"] == 1],
        target_df[target_df["Type"] == 0],
        outlier_dict,
    )

    # Create feature dataframe and grab target classifications
    feature_df = target_df_outliers_removed.iloc[:, feature_array]
    types = target_df_outliers_removed["Type"].values

    # Normalize feature dataframe
    feature_df_norm, _ = clean_data.normalize_data(
        feature_df, feature_df, min_max=False
    )

    # Set up and perform PCA on feature dataframe
    pca = PCA(n_components=len(feature_array))
    X_pca = pca.fit_transform(feature_df_norm)

    # Color code based on classes (red DD, gray Naive)
    c = ["r" if types[i] == 0 else "gray" for i in range(len(types))]

    # Create DD and Naive PCA variables
    X_pca_dd = X_pca[types == 0, :]
    X_pca_naive = X_pca[types == 1, :]

    # Determine centroids from PCA
    dd_centroid = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(X_pca_dd)
    naive_centroid = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(
        X_pca_naive
    )

    # Plot PCA color coded by true values, include centroids
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300, tight_layout=True)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=c, s=4)
    ax.scatter(
        dd_centroid.cluster_centers_[0, 0],
        dd_centroid.cluster_centers_[0, 1],
        marker="X",
        color="darkred",
        edgecolors="w",
    )
    ax.scatter(
        naive_centroid.cluster_centers_[0, 0],
        naive_centroid.cluster_centers_[0, 1],
        marker="X",
        color="black",
        edgecolors="w",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    makeNice(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig("../data/pca/all_pre_pca.pdf", bbox_inches="tight")
    plt.close()
    run_cmd("open ../data/pca/all_pre_pca.pdf")

    # Detemine nearest centroid by L2 norm
    centroid_match_train = np.zeros(len(X_pca))
    for i in range(len(X_pca)):
        dd_dist = np.linalg.norm(X_pca[i, :] - dd_centroid.cluster_centers_)
        naive_dist = np.linalg.norm(X_pca[i, :] - naive_centroid.cluster_centers_)
        if naive_dist < dd_dist:
            centroid_match_train[i] = 1

    # Print accuracy by K-means centroid clustering
    acc = accuracy_score(types, centroid_match_train)
    print(len(centroid_match_train))
    print("PCA K-Means Cluster Acc:\t", acc)

    # Plot PCA color coded by nearest centroid
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300, tight_layout=True)
    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=["r" if x == 0 else "gray" for x in centroid_match_train],
        s=4,
    )
    ax.scatter(
        dd_centroid.cluster_centers_[0, 0],
        dd_centroid.cluster_centers_[0, 1],
        marker="X",
        color="darkred",
        edgecolors="w",
    )
    ax.scatter(
        naive_centroid.cluster_centers_[0, 0],
        naive_centroid.cluster_centers_[0, 1],
        marker="X",
        color="black",
        edgecolors="w",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    makeNice(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig("../data/pca/all_pca_clustered.pdf", bbox_inches="tight")
    plt.close()
    run_cmd("open ../data/pca/all_pca_clustered.pdf")

    # Read in 15 MLPs and determine class probabilities of all data points
    predicts_prob = np.zeros((len(feature_df_norm), 2))
    predicts_all = np.zeros(len(feature_df_norm))
    for seed in seeds:
        np.random.seed(seed)
        with open(f"../data/neural_net/MLP_seed_{seed:02d}.pkl", "rb") as f:
            clf = pickle.load(f)
        predicts_all += clf.predict(feature_df_norm)
        predicts_prob += clf.predict_proba(feature_df_norm) / len(seeds)

    # Determine and print accuracy of averaged MLP probablities and majority rule
    acc = accuracy_score(
        types,
        [0 if predicts_prob[i, 0] >= 0.5 else 1 for i in range(predicts_prob.shape[0])],
    )
    print(len(types))
    print("PCA MLP Accuracy:\t", acc)

    acc = accuracy_score(
        types,
        [1 if predicts_all[i] >= 8 else 0 for i in range(predicts_prob.shape[0])],
    )
    print("PCA MLP Accuracy by majority:\t", acc)

    """dd_centroid = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(
        X_pca[predicts_prob[:, 0] >= 0.5]
    )
    naive_centroid = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(
        X_pca[predicts_prob[:, 0] < 0.5]
    )"""

    # Plot PCA color coded by MLP average probability
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300, tight_layout=True)
    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=[
            "r" if x[0] >= 0.5 else "gray" for x in predicts_prob
        ],  # ["r" if x < 8 else "gray" for x in predicts_all],  # uncomment to switch to majority rule,
        s=4,
    )
    ax.scatter(
        dd_centroid.cluster_centers_[0, 0],
        dd_centroid.cluster_centers_[0, 1],
        marker="X",
        color="darkred",
        edgecolors="w",
    )
    ax.scatter(
        naive_centroid.cluster_centers_[0, 0],
        naive_centroid.cluster_centers_[0, 1],
        marker="X",
        color="black",
        edgecolors="w",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    makeNice(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig("../data/pca/all_mlp_predictions.pdf", bbox_inches="tight")
    plt.close()
    run_cmd("open ../data/pca/all_mlp_predictions.pdf")
