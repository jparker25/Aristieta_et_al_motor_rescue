import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from scipy import stats
from scipy.stats import ks_2samp

import os, sys

# user modules
from helpers import *
import clean_data
import plot_data


def apply_pca(norm_df, pca):
    pca_applied = np.zeros((norm_df.shape[0], pca.n_components_))
    for i in range(pca_applied.shape[0]):
        pca_applied[i, :] = [
            np.dot(norm_df.iloc[i, :], pca.components_[k, :])
            for k in range(pca.n_components_)
        ]
    return pca_applied


def pca_on_df(
    target_df,
    feature_array,
    motor_rescue_dfs,
    post_jaws_dfs,
    post_npas_dfs,
    xlabels,
    chunked,
    n_components_to_display=5,
    show=False,
):
    feature_df = target_df.iloc[:, feature_array]

    X_train, X_test, y_train, y_test = clean_data.split_data(
        feature_df, target_df, 0.8, seed=1
    )

    motor_feature_dfs = [x.iloc[:, feature_array] for x in motor_rescue_dfs]
    norm_motor_dfs = [
        clean_data.normalize_data(X_train, x, min_max=False)[1]
        for x in motor_feature_dfs
    ]

    X_train_norm, X_test_norm = clean_data.normalize_data(
        X_train, X_test, min_max=False
    )

    dd_feature_df = X_train_norm.loc[y_train == 0, :]
    naive_feature_df = X_train_norm.loc[y_train == 1, :]

    pca = PCA(n_components=len(feature_array))
    X_pca = pca.fit_transform(X_train_norm)
    c = ["b" if y_train.values[i] == 0 else "r" for i in range(len(y_train))]
    c_test = ["b" if y_test.values[i] == 0 else "r" for i in range(len(y_test))]

    x_test_pca = apply_pca(X_test_norm, pca)
    X_pca_dd = X_pca[y_train == 0, :]
    X_pca_naive = X_pca[y_train == 1, :]

    dd_centroid = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(X_pca_dd)
    naive_centroid = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(
        X_pca_naive
    )
    centroid_match_train = np.zeros(len(X_pca))
    for i in range(len(X_pca)):
        dd_dist = np.linalg.norm(X_pca[i, :] - dd_centroid.cluster_centers_)
        naive_dist = np.linalg.norm(X_pca[i, :] - naive_centroid.cluster_centers_)
        if naive_dist < dd_dist:
            centroid_match_train[i] = 1
    train_acc = accuracy_score(y_train, centroid_match_train)

    centroid_match_test = np.zeros(len(x_test_pca))
    for i in range(len(x_test_pca)):
        dd_dist = np.linalg.norm(x_test_pca[i, :] - dd_centroid.cluster_centers_)
        naive_dist = np.linalg.norm(x_test_pca[i, :] - naive_centroid.cluster_centers_)
        if naive_dist < dd_dist:
            centroid_match_test[i] = 1
    test_acc = accuracy_score(y_test, centroid_match_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(200, 100),
        activation="relu",
        solver="adam",
        random_state=0,
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

    clf_pca = MLPClassifier(
        hidden_layer_sizes=(200, 100),
        activation="relu",
        solver="adam",
        random_state=0,
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
    clf_pca.fit(X_pca, y_train)

    y_mlp_train = clf.predict(X_train_norm)
    y_mlp_test = clf.predict(X_test_norm)

    y_mlp_pca_train = clf.predict(X_pca)
    y_mlp_pca_test = clf.predict(x_test_pca)

    mlp_train_acc = accuracy_score(y_train, y_mlp_train)
    mlp_test_acc = accuracy_score(y_test, y_mlp_test)

    mlp_pca_train_acc = accuracy_score(y_train, y_mlp_pca_train)
    mlp_pca_test_acc = accuracy_score(y_test, y_mlp_pca_test)

    print(
        "MLP:\t",
        mlp_train_acc,
        mlp_test_acc,
    )

    print(
        "MLP PCA:\t",
        mlp_pca_train_acc,
        mlp_pca_test_acc,
    )

    print("PCA Clusters:\t", train_acc, test_acc)

    classifier_PCA = RandomForestClassifier()
    classifier_PCA.fit(X_pca, y_train)
    y_pred_test_rforest_pca = classifier_PCA.predict(x_test_pca)
    y_pred_train_rforest_pca = classifier_PCA.predict(X_pca)

    forest_pca_train_acc = accuracy_score(y_train, y_pred_train_rforest_pca)
    forest_pca_test_acc = accuracy_score(y_test, y_pred_test_rforest_pca)

    print(
        "RandomForest PCA:\t",
        forest_pca_train_acc,
        forest_pca_test_acc,
    )

    classifier = RandomForestClassifier()
    classifier.fit(X_train_norm, y_train)
    y_pred_test_rforest = classifier.predict(X_test_norm)
    y_pred_train_rforest = classifier.predict(X_train_norm)

    forest_train_acc = accuracy_score(y_train, y_pred_train_rforest)
    forest_test_acc = accuracy_score(y_test, y_pred_test_rforest)

    print(
        "RandomForest:\t",
        forest_train_acc,
        forest_test_acc,
    )

    classifier = RandomForestClassifier()

    fig, ax = plt.subplots(4, 3, figsize=(12, 10), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    fig2, ax2 = plt.subplots(4, 3, figsize=(12, 8), dpi=300, tight_layout=True)
    axes2 = [ax2[i, j] for i in range(4) for j in range(3)]
    prob_thres_dd = 0.75
    prob_thres_naive = 0.75
    probabilities = clf.predict_proba(X_train_norm)
    probabilities_dd = probabilities[:, 0]
    probabilities_naive = probabilities[:, 1]
    feats_dd = X_train[(probabilities_dd > prob_thres_dd) & (y_train == 0)]
    feats_naive = X_train[(probabilities_naive > prob_thres_naive) & (y_train == 1)]
    i = 0
    for col in X_train.columns:
        bins = np.histogram_bin_edges(X_train[col], bins=20)
        sns.ecdfplot(
            feats_dd,
            x=col,
            ax=axes2[i],
            legend=False,
            stat="proportion",
            color="b",
        )
        sns.ecdfplot(
            feats_naive,
            x=col,
            ax=axes2[i],
            legend=False,
            stat="proportion",
            color="r",
        )
        sns.histplot(
            data=feats_dd,
            x=col,
            edgecolor="w",
            bins=bins,
            kde=True,
            stat="probability",
            color="b",
            ax=axes[i],
            legend=False if i != 0 else True,
            label=f"DD (n={len(feats_dd)}/{len(X_train_norm[y_train == 0])}, {len(feats_dd)/len(X_train_norm[y_train == 0]):.02f})",
        )
        sns.histplot(
            data=feats_naive,
            x=col,
            edgecolor="w",
            bins=bins,
            kde=True,
            stat="probability",
            legend=False if i != 0 else True,
            color="r",
            ax=axes[i],
            label=f"Naive (n={len(feats_naive)}/{len(X_train_norm[y_train == 1])}, {len(feats_naive)/len(X_train_norm[y_train == 1]):.02f})",
        )
        if i == 0:
            axes[i].legend()
        dd_col = feats_dd[col]
        naive_col = feats_naive[col]
        ylims = axes[i].get_ylim()
        xlims = axes[i].get_xlim()
        axes[i].vlines(
            np.mean(dd_col),
            ylims[0],
            ylims[1],
            color="b",
            linestyle="dashed",
        )
        axes[i].vlines(
            np.mean(naive_col),
            ylims[0],
            ylims[1],
            color="r",
            linestyle="dashed",
        )
        axes2[i].vlines(
            np.mean(dd_col),
            0,
            1,
            color="b",
            linestyle="dashed",
        )
        axes2[i].vlines(
            np.mean(naive_col),
            0,
            1,
            color="r",
            linestyle="dashed",
        )
        axes2[i].vlines(
            np.median(dd_col),
            0,
            1,
            color="b",
            linestyle="dotted",
        )
        axes2[i].vlines(
            np.median(naive_col),
            0,
            1,
            color="r",
            linestyle="dotted",
        )
        ks_pval = ks_2samp(dd_col, naive_col).pvalue
        plot_data.plot_bracket(
            axes[i],
            xlims[0] + (xlims[1] - xlims[0]) * 0.25,
            xlims[1] - (xlims[1] - xlims[0]) * 0.25,
            ylims[0] + 1.15 * (ylims[1] - ylims[0]),
            ylims[0] + 1.18 * (ylims[1] - ylims[0]),
            f"KS $p{'*' if ks_pval < 0.05 else ''}=${ks_pval:.3f}",
        )

        ttest_pval = stats.ttest_ind(dd_col, naive_col).pvalue
        plot_data.plot_bracket(
            axes[i],
            np.mean(naive_col),
            np.mean(dd_col),
            ylims[1],
            ylims[0] + 1.03 * (ylims[1] - ylims[0]),
            f"T-test $p{'*' if ttest_pval < 0.05 else ''}=${ttest_pval:.3f}",
        )
        i += 1
    makeNice(axes)
    add_fig_labels(axes)
    fig.savefig("../data/pca/prob_features_split.pdf", bbox_inches="tight")

    makeNice(axes2)
    add_fig_labels(axes2)
    fig2.savefig("../data/pca/prob_features_split_cdf.pdf", bbox_inches="tight")
    plt.close()
    run_cmd("open ../data/pca/prob_features_split.pdf")
    run_cmd("open ../data/pca/prob_features_split_cdf.pdf")

    sys.exit()

    fig, ax = plt.subplots(3, 4, figsize=(12, 10), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(3) for j in range(4)]
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=c, alpha=0.1)

    axes[1].scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=c_test, alpha=0.1)

    axes[2].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=["b" if x == 0 else "r" for x in centroid_match_train],
        alpha=0.1,
    )
    axes[3].scatter(
        x_test_pca[:, 0],
        x_test_pca[:, 1],
        c=["b" if x == 0 else "r" for x in centroid_match_test],
        alpha=0.1,
    )

    axes[4].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=["b" if x == 0 else "r" for x in y_mlp_train],
        alpha=0.1,
    )
    axes[5].scatter(
        x_test_pca[:, 0],
        x_test_pca[:, 1],
        c=["b" if x == 0 else "r" for x in y_mlp_test],
        alpha=0.1,
    )

    axes[6].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=["b" if x == 0 else "r" for x in y_mlp_pca_train],
        alpha=0.1,
    )
    axes[7].scatter(
        x_test_pca[:, 0],
        x_test_pca[:, 1],
        c=["b" if x == 0 else "r" for x in y_mlp_pca_test],
        alpha=0.1,
    )

    axes[8].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=["b" if x == 0 else "r" for x in y_pred_train_rforest],
        alpha=0.1,
    )
    axes[9].scatter(
        x_test_pca[:, 0],
        x_test_pca[:, 1],
        c=["b" if x == 0 else "r" for x in y_pred_test_rforest],
        alpha=0.1,
    )

    axes[10].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=["b" if x == 0 else "r" for x in y_pred_train_rforest_pca],
        alpha=0.1,
    )
    axes[11].scatter(
        x_test_pca[:, 0],
        x_test_pca[:, 1],
        c=["b" if x == 0 else "r" for x in y_pred_test_rforest_pca],
        alpha=0.1,
    )

    for i in range(len(axes)):
        axes[i].scatter(
            dd_centroid.cluster_centers_[0, 0],
            dd_centroid.cluster_centers_[0, 1],
            marker="x",
            color="darkblue",
        )
        axes[i].scatter(
            naive_centroid.cluster_centers_[0, 0],
            naive_centroid.cluster_centers_[0, 1],
            marker="x",
            color="darkred",
        )
        axes[i].set_xlabel("PC1")
        axes[i].set_ylabel("PC2")

    axes[0].set_title("PCA Training", fontsize=10)
    axes[1].set_title("PCA Testing", fontsize=10)
    axes[2].set_title(f"PCA Cluster Train Acc:\t{train_acc:.02f}", fontsize=10)
    axes[3].set_title(f"PCA Cluster Test Acc:\t{test_acc:.02f}", fontsize=10)
    axes[4].set_title(f"MLP Train Acc:\t{mlp_train_acc:.02f}", fontsize=10)
    axes[5].set_title(f"MLP Test Acc:\t{mlp_test_acc:.02f}", fontsize=10)
    axes[6].set_title(f"MLP PCA Train Acc:\t{mlp_pca_train_acc:.02f}", fontsize=10)
    axes[7].set_title(f"MLP PCA Test Acc:\t{mlp_pca_test_acc:.02f}", fontsize=10)
    axes[8].set_title(f"RForest Train Acc:\t{forest_train_acc:.02f}", fontsize=10)
    axes[9].set_title(f"RForest Test Acc:\t{forest_test_acc:.02f}", fontsize=10)
    axes[10].set_title(
        f"RForest PCA Train Acc:\t{forest_pca_train_acc:.02f}", fontsize=10
    )
    axes[11].set_title(
        f"RForest PCA Test Acc:\t{forest_pca_test_acc:.02f}", fontsize=10
    )

    makeNice(axes, labelsize=6)
    match_axis(axes)
    add_fig_labels(axes)
    fig.savefig(f"data/pca/cluster_accuracy.pdf", bbox_inches="tight")
    plt.close()
    run_cmd(f"open data/pca/cluster_accuracy.pdf")

    # Analyze the principal components
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)

    # Get the loadings (or weights) of the first few principal components
    loadings = pca.components_[:n_components_to_display, :]

    # Create a DataFrame for better visualization
    loading_matrix = pd.DataFrame(
        loadings,
        columns=[f"{feature_df.columns[i]}" for i in range(len(feature_array))],
        index=[f"PC {i+1}" for i in range(n_components_to_display)],
    )

    dd_pca = apply_pca(dd_feature_df, pca)
    naive_pca = apply_pca(naive_feature_df, pca)

    model = KMeans(n_clusters=2, random_state=0, max_iter=3000, n_init=5, tol=1e-8).fit(
        X_train
    )
    ckmeans = ["b" if model.labels_[i] == 0 else "r" for i in range(len(y_train))]

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i] for i in range(2)]

    # Plot the explained variance
    axes[0].bar(
        range(1, len(feature_array) + 1),
        explained_variance,
        alpha=0.5,
        label="Individual explained variance",
        color="b",
    )
    axes[0].step(
        range(1, len(feature_array) + 1),
        cumulative_explained_variance,
        where="mid",
        label="Cumulative explained variance",
        color="b",
    )
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_xlabel("Principal components")
    axes[0].legend(loc="best", frameon=False, fontsize=6)
    axes[0].set_title("PCA Explained Variance")

    sns.heatmap(
        loading_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        annot_kws={"fontsize": 8},
    )
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("PC#")
    makeNice(axes, labelsize=6)

    fig.savefig(f"data/pca/pca.pdf")
    plt.close()

    jaws_pre_pca = apply_pca(norm_motor_dfs[0], pca)
    jaws_post_pca = apply_pca(norm_motor_dfs[1], pca)
    npas_pre_pca = apply_pca(norm_motor_dfs[2], pca)
    npas_post_pca = apply_pca(norm_motor_dfs[3], pca)

    ### PLOT ALL DATA IN PCA SPACE ###
    plot_post_data = False
    fig = plt.figure(figsize=(8, 6), dpi=300, tight_layout=True)
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    c = ["b" if y_train.values[i] == 0 else "r" for i in range(len(y_train))]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], color=c)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], color=c, s=4, alpha=0.25)

    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(X_pca[:, 0], X_pca[:, 2], color=c, s=4, alpha=0.25)
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC3")
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(X_pca[:, 1], X_pca[:, 2], color=c, s=4, alpha=0.25)
    if plot_post_data:
        ax.scatter(
            npas_post_pca[:, 0], npas_post_pca[:, 1], npas_post_pca[:, 2], color="g"
        )
        ax.scatter(
            jaws_post_pca[:, 0],
            jaws_post_pca[:, 1],
            jaws_post_pca[:, 2],
            color="purple",
        )
        ax2.scatter(npas_post_pca[:, 0], npas_post_pca[:, 1], color="g", s=4)
        ax2.scatter(jaws_post_pca[:, 0], jaws_post_pca[:, 1], color="purple", s=4)
        ax3.scatter(npas_post_pca[:, 0], npas_post_pca[:, 2], color="g", s=4)
        ax3.scatter(jaws_post_pca[:, 0], jaws_post_pca[:, 2], color="purple", s=4)
        ax4.scatter(npas_post_pca[:, 1], npas_post_pca[:, 2], color="g", s=4)
        ax4.scatter(jaws_post_pca[:, 1], jaws_post_pca[:, 2], color="purple", s=4)
    ax4.set_xlabel("PC2")
    ax4.set_ylabel("PC3")
    makeNice([ax2, ax3, ax4])

    fig.savefig("data/pca/pc_comp_3d.pdf", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300, tight_layout=True)
    ax.plot(
        np.arange(len(pca.explained_variance_)),
        pca.explained_variance_,
        marker="o",
        color="k",
        markersize=4,
        lw=0.5,
    )
    ax.set_xlabel("PC#")
    ax.set_ylabel("Explained Variance")
    makeNice(ax, labelsize=8)
    fig.savefig("data/pca/explained_variance.pdf", bbox_inches="tight")
    plt.close()

    if show:
        run_cmd("open data/pca/pca.pdf")
        run_cmd("open data/pca/pc_comp_3d.pdf")
        run_cmd("open data/pca/explained_variance.pdf")
    sys.exit()

    dd_pca = apply_pca(dd_feature_df, pca)
    naive_pca = apply_pca(naive_feature_df, pca)

    jaws_pre_pca = apply_pca(norm_motor_dfs[0], pca)
    jaws_post_pca = apply_pca(norm_motor_dfs[1], pca)
    npas_pre_pca = apply_pca(norm_motor_dfs[2], pca)
    npas_post_pca = apply_pca(norm_motor_dfs[3], pca)

    post_jaws_dfs = [
        x.drop(columns=["Post-Time", "Medial", "mouse", "folder", "name"])
        for x in post_jaws_dfs
    ]
    post_npas_dfs = [
        x.drop(columns=["Post-Time", "Medial", "mouse", "folder", "name"])
        for x in post_npas_dfs
    ]

    jaws_post_dfs_pcas = [
        clean_data.normalize_data(X_train, x, min_max=False)[1] for x in post_jaws_dfs
    ]

    jaws_post_dfs_pcas = [x.values for x in jaws_post_dfs_pcas]

    npas_post_dfs_pcas = [
        clean_data.normalize_data(X_train, x, min_max=False)[1] for x in post_npas_dfs
    ]

    npas_post_dfs_pcas = [x.values for x in npas_post_dfs_pcas]

    ### FIND CENTROIDS BASED ON DIMENISONS
    centroid_dimensions = 12
    dd_centroid = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(
        dd_pca[:, 0:centroid_dimensions]
    )
    naive_centroid = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(
        naive_pca[:, 0:centroid_dimensions]
    )

    ### CREATE FIGURE FOR  HEALTHY METRICS ###
    fig, ax = plt.subplots(2, 3, figsize=(10, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(2) for j in range(3)]

    ### PLOT CENTROID SCORES ###
    jaws_naive_score, jaws_dd_score, jaws_healthy_score = determine_centroid_score(
        jaws_pre_pca, naive_centroid, dd_centroid, n=centroid_dimensions
    )

    npas_naive_score, npas_dd_score, npas_healthy_score = determine_centroid_score(
        npas_pre_pca, naive_centroid, dd_centroid, n=centroid_dimensions
    )

    np.savetxt(
        f"data/pca/jaws_pre_centroid_healthy_score.txt",
        jaws_healthy_score,
        newline="\n",
        fmt="%f",
    )

    np.savetxt(
        f"data/pca/npas_pre_centroid_healthy_score.txt",
        npas_healthy_score,
        newline="\n",
        fmt="%f",
    )

    jaws_scores = np.zeros((len(post_jaws_dfs) + 1, 2))
    jaws_scores[0] = [np.mean(jaws_healthy_score), scipy.stats.sem(jaws_healthy_score)]
    jaws_ttest_scores = [jaws_healthy_score]
    time_chunk_labels = [30, 60, "90_120", "150_180_210"]
    for i in range(1, jaws_scores.shape[0]):
        _, _, post_jaws_healthy = determine_centroid_score(
            jaws_post_dfs_pcas[i - 1],
            naive_centroid,
            dd_centroid,
            n=centroid_dimensions,
        )
        np.savetxt(
            f"data/pca/jaws_post_{time_chunk_labels[i-1]}_centroid_healthy_score.txt",
            post_jaws_healthy,
            newline="\n",
            fmt="%f",
        )
        jaws_ttest_scores.append(post_jaws_healthy)
        jaws_scores[i] = [
            np.mean(post_jaws_healthy),
            scipy.stats.sem(post_jaws_healthy),
        ]

    npas_scores = np.zeros((len(post_npas_dfs) + 1, 2))
    npas_scores[0] = [np.mean(npas_healthy_score), scipy.stats.sem(npas_healthy_score)]
    npas_ttest_scores = [npas_healthy_score]
    for i in range(1, npas_scores.shape[0]):
        _, _, post_npas_healthy = determine_centroid_score(
            npas_post_dfs_pcas[i - 1],
            naive_centroid,
            dd_centroid,
            n=centroid_dimensions,
        )
        np.savetxt(
            f"data/pca/npas_post_{time_chunk_labels[i-1]}_centroid_healthy_score.txt",
            post_npas_healthy,
            newline="\n",
            fmt="%f",
        )
        npas_ttest_scores.append(post_npas_healthy)
        npas_scores[i] = [
            np.mean(post_npas_healthy),
            scipy.stats.sem(post_npas_healthy),
        ]

    ttest_results = np.zeros((3, npas_scores.shape[0]))
    ttest_results[0] = [
        scipy.stats.ttest_ind(
            jaws_ttest_scores[i], npas_ttest_scores[i], equal_var=False
        ).pvalue
        for i in range(len(jaws_ttest_scores))
    ]
    ttest_results[1, 1:] = [
        scipy.stats.ttest_ind(
            jaws_ttest_scores[0], jaws_ttest_scores[i], equal_var=False
        ).pvalue
        for i in range(1, len(jaws_ttest_scores))
    ]
    ttest_results[2, 1:] = [
        scipy.stats.ttest_ind(
            npas_ttest_scores[0], npas_ttest_scores[i], equal_var=False
        ).pvalue
        for i in range(1, len(jaws_ttest_scores))
    ]

    sns.heatmap(
        ttest_results,
        annot=True,
        ax=axes[3],
        fmt=".3f",
        annot_kws={"fontsize": 6},
        vmin=0,
        vmax=0.1,
        cmap="RdYlGn",
        mask=ttest_results == 0,
    )

    axes[0].plot(
        np.arange(jaws_scores.shape[0]),
        jaws_scores[:, 0],
        color="b",
        marker="o",
        label="JAWS",
    )
    axes[0].errorbar(
        np.arange(jaws_scores.shape[0]),
        jaws_scores[:, 0],
        yerr=jaws_scores[:, 1],
        color="b",
        capsize=4,
    )
    axes[0].plot(
        np.arange(jaws_scores.shape[0]),
        npas_scores[:, 0],
        color="r",
        marker="o",
        label="Npas",
    )
    axes[0].errorbar(
        np.arange(jaws_scores.shape[0]),
        npas_scores[:, 0],
        yerr=npas_scores[:, 1],
        color="r",
        capsize=4,
    )
    axes[0].legend(fontsize=8)

    ### PLOT DISTANCE SCORES ###
    npas_naive_score, npas_dd_score, npas_healthy_score = determine_distance_score(
        npas_pre_pca, naive_pca, dd_pca, n=centroid_dimensions
    )

    jaws_naive_score, jaws_dd_score, jaws_healthy_score = determine_distance_score(
        jaws_pre_pca, naive_pca, dd_pca, n=centroid_dimensions
    )

    np.savetxt(
        f"data/pca/jaws_pre_distance_healthy_score.txt",
        jaws_healthy_score,
        newline="\n",
        fmt="%f",
    )

    np.savetxt(
        f"data/pca/npas_pre_distance_healthy_score.txt",
        npas_healthy_score,
        newline="\n",
        fmt="%f",
    )

    jaws_scores = np.zeros((len(post_jaws_dfs) + 1, 2))
    jaws_scores[0] = [np.mean(jaws_healthy_score), scipy.stats.sem(jaws_healthy_score)]
    jaws_ttest_scores = [jaws_healthy_score]
    for i in range(1, jaws_scores.shape[0]):
        _, _, post_jaws_healthy = determine_distance_score(
            jaws_post_dfs_pcas[i - 1], naive_pca, dd_pca, n=centroid_dimensions
        )
        np.savetxt(
            f"data/pca/jaws_post_{time_chunk_labels[i-1]}_distance_healthy_score.txt",
            post_jaws_healthy,
            newline="\n",
            fmt="%f",
        )
        jaws_ttest_scores.append(post_jaws_healthy)
        jaws_scores[i] = [
            np.mean(post_jaws_healthy),
            scipy.stats.sem(post_jaws_healthy),
        ]

    npas_scores = np.zeros((len(post_npas_dfs) + 1, 2))
    npas_scores[0] = [np.mean(npas_healthy_score), scipy.stats.sem(npas_healthy_score)]
    npas_ttest_scores = [npas_healthy_score]
    for i in range(1, npas_scores.shape[0]):
        _, _, post_npas_healthy = determine_distance_score(
            npas_post_dfs_pcas[i - 1], naive_pca, dd_pca, n=centroid_dimensions
        )
        np.savetxt(
            f"data/pca/npas_post_{time_chunk_labels[i-1]}_distance_healthy_score.txt",
            post_npas_healthy,
            newline="\n",
            fmt="%f",
        )
        npas_ttest_scores.append(post_npas_healthy)
        npas_scores[i] = [
            np.mean(post_npas_healthy),
            scipy.stats.sem(post_npas_healthy),
        ]

    ttest_results = np.zeros((3, npas_scores.shape[0]))
    ttest_results[0] = [
        scipy.stats.ttest_ind(
            jaws_ttest_scores[i], npas_ttest_scores[i], equal_var=False
        ).pvalue
        for i in range(len(jaws_ttest_scores))
    ]
    ttest_results[1, 1:] = [
        scipy.stats.ttest_ind(
            jaws_ttest_scores[0], jaws_ttest_scores[i], equal_var=False
        ).pvalue
        for i in range(1, len(jaws_ttest_scores))
    ]
    ttest_results[2, 1:] = [
        scipy.stats.ttest_ind(
            npas_ttest_scores[0], npas_ttest_scores[i], equal_var=False
        ).pvalue
        for i in range(1, len(jaws_ttest_scores))
    ]

    sns.heatmap(
        ttest_results,
        annot=True,
        ax=axes[4],
        fmt=".3f",
        annot_kws={"fontsize": 6},
        vmin=0,
        vmax=0.1,
        cmap="RdYlGn",
        mask=ttest_results == 0,
    )

    axes[1].plot(
        np.arange(jaws_scores.shape[0]),
        jaws_scores[:, 0],
        color="b",
        marker="o",
    )
    axes[1].errorbar(
        np.arange(jaws_scores.shape[0]),
        jaws_scores[:, 0],
        yerr=jaws_scores[:, 1],
        color="b",
        capsize=4,
    )
    axes[1].plot(
        np.arange(jaws_scores.shape[0]),
        npas_scores[:, 0],
        color="r",
        marker="o",
    )
    axes[1].errorbar(
        np.arange(jaws_scores.shape[0]),
        npas_scores[:, 0],
        yerr=npas_scores[:, 1],
        color="r",
        capsize=4,
    )

    ### PLOT PERCENTILE SCORES ###
    npas_naive_score, npas_dd_score, npas_healthy_score = determine_percentile_score(
        npas_pre_pca,
        naive_pca,
        dd_pca,
        naive_centroid,
        dd_centroid,
        n=centroid_dimensions,
    )

    jaws_naive_score, jaws_dd_score, jaws_healthy_score = determine_percentile_score(
        jaws_pre_pca,
        naive_pca,
        dd_pca,
        naive_centroid,
        dd_centroid,
        n=centroid_dimensions,
    )

    np.savetxt(
        f"data/pca/jaws_pre_percentile_healthy_score.txt",
        jaws_healthy_score,
        newline="\n",
        fmt="%f",
    )

    np.savetxt(
        f"data/pca/npas_pre_percentile_healthy_score.txt",
        npas_healthy_score,
        newline="\n",
        fmt="%f",
    )

    jaws_scores = np.zeros((len(post_jaws_dfs) + 1, 2))
    jaws_scores[0] = [np.mean(jaws_healthy_score), scipy.stats.sem(jaws_healthy_score)]
    jaws_ttest_scores = [jaws_healthy_score]
    for i in range(1, jaws_scores.shape[0]):
        _, _, post_jaws_healthy = determine_percentile_score(
            jaws_post_dfs_pcas[i - 1],
            naive_pca,
            dd_pca,
            naive_centroid,
            dd_centroid,
            n=centroid_dimensions,
        )
        np.savetxt(
            f"data/pca/jaws_post_{time_chunk_labels[i-1]}_percentile_healthy_score.txt",
            post_jaws_healthy,
            newline="\n",
            fmt="%f",
        )
        jaws_ttest_scores.append(post_jaws_healthy)
        jaws_scores[i] = [
            np.mean(post_jaws_healthy),
            scipy.stats.sem(post_jaws_healthy),
        ]

    npas_scores = np.zeros((len(post_npas_dfs) + 1, 2))
    npas_scores[0] = [np.mean(npas_healthy_score), scipy.stats.sem(npas_healthy_score)]
    npas_ttest_scores = [npas_healthy_score]
    for i in range(1, npas_scores.shape[0]):
        _, _, post_npas_healthy = determine_percentile_score(
            npas_post_dfs_pcas[i - 1],
            naive_pca,
            dd_pca,
            naive_centroid,
            dd_centroid,
            n=centroid_dimensions,
        )
        np.savetxt(
            f"data/pca/npas_post_{time_chunk_labels[i-1]}_percentile_healthy_score.txt",
            post_npas_healthy,
            newline="\n",
            fmt="%f",
        )
        npas_ttest_scores.append(post_npas_healthy)
        npas_scores[i] = [
            np.mean(post_npas_healthy),
            scipy.stats.sem(post_npas_healthy),
        ]

    ttest_results = np.zeros((3, npas_scores.shape[0]))
    ttest_results[0] = [
        scipy.stats.ttest_ind(
            jaws_ttest_scores[i], npas_ttest_scores[i], equal_var=False
        ).pvalue
        for i in range(len(jaws_ttest_scores))
    ]
    ttest_results[1, 1:] = [
        scipy.stats.ttest_ind(
            jaws_ttest_scores[0], jaws_ttest_scores[i], equal_var=False
        ).pvalue
        for i in range(1, len(jaws_ttest_scores))
    ]
    ttest_results[2, 1:] = [
        scipy.stats.ttest_ind(
            npas_ttest_scores[0], npas_ttest_scores[i], equal_var=False
        ).pvalue
        for i in range(1, len(jaws_ttest_scores))
    ]

    sns.heatmap(
        ttest_results,
        annot=True,
        ax=axes[5],
        fmt=".3f",
        annot_kws={"fontsize": 6},
        vmin=0,
        vmax=0.1,
        cmap="RdYlGn",
        mask=ttest_results == 0,
    )

    axes[2].plot(
        np.arange(jaws_scores.shape[0]),
        jaws_scores[:, 0],
        color="b",
        marker="o",
    )
    axes[2].errorbar(
        np.arange(jaws_scores.shape[0]),
        jaws_scores[:, 0],
        yerr=jaws_scores[:, 1],
        color="b",
        capsize=4,
    )
    axes[2].plot(
        np.arange(jaws_scores.shape[0]),
        npas_scores[:, 0],
        color="r",
        marker="o",
    )
    axes[2].errorbar(
        np.arange(jaws_scores.shape[0]),
        npas_scores[:, 0],
        yerr=npas_scores[:, 1],
        color="r",
        capsize=4,
    )
    for i in range(len(axes[:3])):
        axes[i].set_xticks(np.arange(jaws_scores.shape[0]))
        axes[i].set_xticklabels(xlabels)
        axes[i].set_xlabel("Post-Stim Time (mins)")
        axes[i].grid(visible=True)
    for i in range(3, len(axes)):
        axes[i].set_xticks(np.arange(0.5, jaws_scores.shape[0] + 0.5))
        axes[i].set_xticklabels(xlabels, fontsize=6)
        axes[i].set_yticks(np.arange(0.5, 3.5))
        axes[i].set_yticklabels(
            ["JAWS vs\nNpas", "Jaws\nPre vs Post", "Npas\nPre vs Post"],
            rotation=0,
            fontsize=6,
        )
    axes[0].set_ylabel("Centroid Metric")
    axes[1].set_ylabel("Distance Metric")
    axes[2].set_ylabel("Percentile Metric")
    plt.suptitle(f"PCA Healthy Score, n = {centroid_dimensions}")

    match_axis(axes[:3], type="y")
    makeNice(axes[:3])
    fig.savefig(
        f"data/pca/healthy_scores_{centroid_dimensions}_chunked_{chunked}.pdf",
        bbox_inches="tight",
    )
    plt.close()
    if show:
        run_cmd(
            f"open data/pca/healthy_scores_{centroid_dimensions}_chunked_{chunked}.pdf"
        )


def determine_centroid_score(post_data, naive_centroid, dd_centroid, n=3):
    dd_score = np.zeros(post_data.shape[0])
    naive_score = np.zeros(post_data.shape[0])
    healthy_score = np.zeros(post_data.shape[0])
    for i in range(dd_score.shape[0]):
        naive_score[i] = np.linalg.norm(
            post_data[i, :n] - naive_centroid.cluster_centers_[0]
        )
        dd_score[i] = np.linalg.norm(post_data[i, :n] - dd_centroid.cluster_centers_[0])
        healthy_score[i] = 1 - naive_score[i] / (naive_score[i] + dd_score[i])
    return naive_score, dd_score, healthy_score


def determine_distance_score(post_data, naive_data, dd_data, n=3):
    dd_score = np.zeros(post_data.shape[0])
    naive_score = np.zeros(post_data.shape[0])
    healthy_score = np.zeros(post_data.shape[0])
    for i in range(dd_score.shape[0]):
        for k in range(naive_data.shape[0]):
            naive_score[i] += (
                np.linalg.norm(post_data[i, :n] - naive_data[k, :n])
                / naive_data.shape[0]
            )
        for k in range(dd_data.shape[0]):
            dd_score[i] += (
                np.linalg.norm(post_data[i, :n] - dd_data[k, :n]) / dd_data.shape[0]
            )
        healthy_score[i] = 1 - naive_score[i] / (naive_score[i] + dd_score[i])
    return naive_score, dd_score, healthy_score


def determine_percentile_score(
    post_data, naive_data, dd_data, naive_centroid, dd_centroid, n=3
):
    dd_score = np.zeros(post_data.shape[0])
    naive_score = np.zeros(post_data.shape[0])
    healthy_score = np.zeros(post_data.shape[0])
    naive_dist = np.zeros(naive_data.shape[0])
    dd_dist = np.zeros(dd_data.shape[0])
    for i in range(naive_data.shape[0]):
        naive_dist[i] = np.linalg.norm(
            naive_data[i, :n] - naive_centroid.cluster_centers_[0]
        )
    for i in range(dd_data.shape[0]):
        dd_dist[i] = np.linalg.norm(dd_data[i, :n] - dd_centroid.cluster_centers_[0])
    dd_dist = sorted(dd_dist)
    naive_dist = sorted(naive_dist)
    for i in range(dd_score.shape[0]):
        dd_score[i] = scipy.stats.percentileofscore(
            dd_dist, np.linalg.norm(post_data[i, :n] - dd_centroid.cluster_centers_[0])
        )
        naive_score[i] = scipy.stats.percentileofscore(
            naive_dist,
            np.linalg.norm(post_data[i, :n] - naive_centroid.cluster_centers_[0]),
        )
        healthy_score[i] = 1 - naive_score[i] / (naive_score[i] + dd_score[i])
    return naive_score, dd_score, healthy_score
