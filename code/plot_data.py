import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os, sys
from scipy.stats import ks_2samp
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec

from helpers import *
import scipy.stats
import matplotlib as mpl
import clean_data

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def plot_pre_post_histograms(
    motor_rescue_dfs, post_jaws_dfs, post_npas_dfs, show=False
):
    fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    axis_plot = 0
    chunk = 0
    for col in post_npas_dfs[0].columns[0:12]:
        bins = np.histogram_bin_edges(motor_rescue_dfs[0][col], bins="auto")
        jaws_data = motor_rescue_dfs[0][col]
        npas_data = motor_rescue_dfs[2][col]
        sns.histplot(
            data=motor_rescue_dfs[0],
            x=col,
            bins=bins,
            color="blue",
            ax=axes[axis_plot],
            legend=False,
            kde=True,
            stat="probability",
            edgecolor="w",
        )
        sns.histplot(
            data=motor_rescue_dfs[2],
            x=col,
            bins=bins,
            color="red",
            ax=axes[axis_plot],
            legend=False,
            kde=True,
            stat="probability",
            edgecolor="w",
        )
        ylims = axes[axis_plot].get_ylim()
        xlims = axes[axis_plot].get_xlim()
        axes[axis_plot].vlines(
            np.mean(jaws_data),
            ylims[0],
            ylims[1],
            color="b",
            linestyle="dashed",
        )
        axes[axis_plot].vlines(
            np.mean(npas_data),
            ylims[0],
            ylims[1],
            color="r",
            linestyle="dashed",
        )
        ks_pval = ks_2samp(jaws_data, npas_data).pvalue
        plot_bracket(
            axes[axis_plot],
            xlims[0] + (xlims[1] - xlims[0]) * 0.25,
            xlims[1] - (xlims[1] - xlims[0]) * 0.25,
            ylims[0] + 1.15 * (ylims[1] - ylims[0]),
            ylims[0] + 1.18 * (ylims[1] - ylims[0]),
            f"KS $p{'*' if ks_pval < 0.05 else ''}=${ks_pval:.3f}",
        )

        ttest_pval = stats.ttest_ind(jaws_data, npas_data).pvalue
        plot_bracket(
            axes[axis_plot],
            np.mean(npas_data),
            np.mean(jaws_data),
            ylims[1],
            ylims[0] + 1.03 * (ylims[1] - ylims[0]),
            f"T-test $p{'*' if ttest_pval < 0.05 else ''}=${ttest_pval:.3f}",
        )
        axes[axis_plot].set_xlabel(col, fontsize=8)
        axes[axis_plot].set_ylabel("Probability", fontsize=8)
        axis_plot += 1
    makeNice(axes, labelsize=6)
    fig.savefig("../data/histogram_pre_stim.pdf", bbox_inches="tight")
    plt.close()
    if show:
        run_cmd("open ../data/histogram_pre_stim.pdf")

    labels = [30, 60, "90_120", "150_180_210"]
    for chunk in range(len(post_npas_dfs)):
        fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
        axes = [ax[i, j] for i in range(4) for j in range(3)]
        axis_plot = 0
        for col in post_npas_dfs[0].columns[0:12]:
            bins = np.histogram_bin_edges(post_npas_dfs[chunk][col], bins="auto")
            jaws_data = post_jaws_dfs[chunk][col]
            npas_data = post_npas_dfs[chunk][col]
            sns.histplot(
                data=post_jaws_dfs[chunk],
                x=col,
                bins=bins,
                color="blue",
                ax=axes[axis_plot],
                legend=False,
                kde=True,
                stat="probability",
                edgecolor="w",
            )
            sns.histplot(
                data=post_npas_dfs[chunk],
                x=col,
                bins=bins,
                color="red",
                ax=axes[axis_plot],
                legend=False,
                kde=True,
                stat="probability",
                edgecolor="w",
            )
            ylims = axes[axis_plot].get_ylim()
            xlims = axes[axis_plot].get_xlim()
            axes[axis_plot].vlines(
                np.mean(jaws_data),
                ylims[0],
                ylims[1],
                color="b",
                linestyle="dashed",
            )
            axes[axis_plot].vlines(
                np.mean(npas_data),
                ylims[0],
                ylims[1],
                color="r",
                linestyle="dashed",
            )
            ks_pval = ks_2samp(jaws_data, npas_data).pvalue
            plot_bracket(
                axes[axis_plot],
                xlims[0] + (xlims[1] - xlims[0]) * 0.25,
                xlims[1] - (xlims[1] - xlims[0]) * 0.25,
                ylims[0] + 1.15 * (ylims[1] - ylims[0]),
                ylims[0] + 1.18 * (ylims[1] - ylims[0]),
                f"KS $p{'*' if ks_pval < 0.05 else ''}=${ks_pval:.3f}",
            )

            ttest_pval = stats.ttest_ind(jaws_data, npas_data).pvalue
            plot_bracket(
                axes[axis_plot],
                np.mean(npas_data),
                np.mean(jaws_data),
                ylims[1],
                ylims[0] + 1.03 * (ylims[1] - ylims[0]),
                f"T-test $p{'*' if ttest_pval < 0.05 else ''}=${ttest_pval:.3f}",
            )
            axes[axis_plot].set_xlabel(col, fontsize=8)
            axes[axis_plot].set_ylabel("Probability", fontsize=8)
            axis_plot += 1
        makeNice(axes, labelsize=6)
        fig.savefig(f"../data/histogram_{labels[chunk]}.pdf", bbox_inches="tight")
        plt.close()
        if show:
            run_cmd(f"open ../data/histogram_{labels[chunk]}.pdf")


def plot_feature_percent_correct(naive_df, dd_df, bins=10):
    dfs = []
    for seed in np.arange(5):
        dfs.append(pd.read_csv(f"../data/neural_net/x_train_values_{seed}.csv"))
        dfs[-1]["DD Probability"] = np.loadtxt(
            f"../data/neural_net/x_train_probs_seed_{seed}.txt", usecols=0
        )
        dfs[-1]["Naive Probability"] = np.loadtxt(
            f"../data/neural_net/x_train_probs_seed_{seed}.txt", usecols=1
        )

    super_df = pd.concat(dfs)
    naive_super = super_df[super_df["Type"] == 1]
    dd_super = super_df[super_df["Type"] == 0]

    cols_to_plot = [
        "FR",
        "CV",
        "Delta Power",
        "Beta Power",
        "Num Bursts",
        "Avg Burst Firing Rate",
        "Percent Time Bursting",
        "Percent of Spikes in Bursts",
        "Avg Burst Duration",
        "Avg Interburst Interval",
        "Non Bursting Firing Rate",
        "Burst Firing Rate Increase",
    ]
    fig, ax = plt.subplots(4, 3, figsize=(12, 8), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    cols_to_plot = [
        "FR",
        "CV",
        "Delta Power",
        "Beta Power",
        "Num Bursts",
        "Avg Burst Firing Rate",
        "Percent Time Bursting",
        "Percent of Spikes in Bursts",
        "Avg Burst Duration",
        "Avg Interburst Interval",
        "Non Bursting Firing Rate",
        "Burst Firing Rate Increase",
    ]
    count = 0
    for col in cols_to_plot:
        col_bins = np.linspace(0, np.max(super_df[col]), bins + 1)
        for i in range(bins):
            dd_bins = dd_super[
                (dd_super[col] >= col_bins[i]) & (dd_super[col] < col_bins[i + 1])
            ]
            dd_percent_correct = (
                len(dd_bins[dd_bins["DD Probability"] >= 0.5]) / len(dd_bins)
                if len(dd_bins) > 0
                else 0
            )
            naive_bins = naive_super[
                (naive_super[col] >= col_bins[i]) & (naive_super[col] < col_bins[i + 1])
            ]
            naive_percent_correct = (
                len(naive_bins[naive_bins["Naive Probability"] >= 0.5])
                / len(naive_bins)
                if len(naive_bins) > 0
                else 0
            )
            axes[count].bar(
                i, dd_percent_correct, color="b", edgecolor="w", alpha=0.5, width=1
            )
            axes[count].bar(
                i,
                naive_percent_correct,
                color="r",
                edgecolor="w",
                alpha=0.5,
                width=1,
            )
            axes[count].set_xticks(np.arange(0, len(col_bins), 4))
            axes[count].set_xticklabels(
                (
                    [f"{x:.02f}" for x in col_bins[::4]]
                    if "Power" not in col
                    else [
                        np.format_float_scientific(x, precision=2)
                        for x in col_bins[::4]
                    ]
                ),
                fontsize=6,
            )
            axes[count].set_ylabel("Percent Correct")
            axes[count].set_xlabel(col)
        count += 1
    makeNice(axes, labelsize=8)
    fig.savefig(f"../data/feature_percent_correct.pdf", bbox_inches="tight")
    plt.close()
    run_cmd(f"open ../data/feature_percent_correct.pdf")


def plot_features_probabilities(naive_df, dd_df):
    dfs = []
    for seed in np.arange(5):
        dfs.append(pd.read_csv(f"../data/neural_net/x_train_values_{seed}.csv"))
        dfs[-1]["DD Probability"] = np.loadtxt(
            f"../data/neural_net/x_train_probs_seed_{seed}.txt", usecols=0
        )
        dfs[-1]["Naive Probability"] = np.loadtxt(
            f"../data/neural_net/x_train_probs_seed_{seed}.txt", usecols=1
        )

    super_df = pd.concat(dfs)
    naive_super = super_df[super_df["Type"] == 1]
    dd_super = super_df[super_df["Type"] == 0]

    cols_to_plot = [
        "FR",
        "CV",
        "Delta Power",
        "Beta Power",
        "Num Bursts",
        "Avg Burst Firing Rate",
        "Percent Time Bursting",
        "Percent of Spikes in Bursts",
        "Avg Burst Duration",
        "Avg Interburst Interval",
        "Non Bursting Firing Rate",
        "Burst Firing Rate Increase",
    ]

    # fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
    # axes = [ax[i, j] for i in range(4) for j in range(3)]

    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35)
    gs_axes = [gs[i, j] for i in range(4) for j in range(3)]

    count = 0
    for col in cols_to_plot:
        gs0 = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=gs_axes[count],
            height_ratios=[2, 1],
            hspace=0.25,
        )
        axes = [fig.add_subplot(gs0[0]), fig.add_subplot(gs0[1])]
        sns.regplot(
            data=naive_super,
            y="Naive Probability",
            x=col,
            color="r",
            scatter=True,
            scatter_kws={"s": 2, "alpha": 0.2, "color": "salmon"},
            line_kws={"lw": 1},
            ax=axes[0],
        )
        sns.regplot(
            data=dd_super,
            y="DD Probability",
            x=col,
            color="b",
            scatter=True,
            scatter_kws={"s": 2, "alpha": 0.2, "color": "lightblue"},
            line_kws={"lw": 1},
            ax=axes[0],
        )
        # axes[0].set_ylim([0.5, 1])
        axes[0].set_ylabel("Probability", fontsize=8)
        axes[0].set_xlabel(col, fontsize=8)

        axes[1].errorbar(
            np.mean(naive_df[col]),
            0,
            xerr=3 * np.std(naive_df[col]),
            color="r",
            marker="o",
            capsize=4,
        )
        axes[1].errorbar(
            np.mean(dd_df[col]),
            1,
            xerr=3 * np.std(dd_df[col]),
            color="b",
            marker="o",
            capsize=4,
        )
        axes[1].set_xlim(
            [
                0 if axes[1].get_xlim()[0] < 0 else axes[1].get_xlim()[0],
                axes[1].get_xlim()[1],
            ]
        )
        axes[1].set_xlabel(col, fontsize=8)
        axes[1].set_ylim([-1, 2])
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["Naive", "DD"])
        count += 1
        match_axis(axes, type="x")
        makeNice(axes, labelsize=8)
    fig.savefig(f"../data/feature_proabilities_training.pdf", bbox_inches="tight")
    plt.close()
    run_cmd(f"open ../data/feature_proabilities_training.pdf")


def plot_column_comparison_over_time(
    axes,
    column,
    jaws_pre_df,
    npas_pre_df,
    post_jaws_dfs,
    post_npas_dfs,
    time_chunk_labels,
):

    count = 1
    jaws_points = [jaws_pre_df[column]]
    jaws_delta_means = np.zeros(len(post_jaws_dfs) + 1)
    jaws_errors = np.zeros(len(post_jaws_dfs) + 1)
    for post_jaws in post_jaws_dfs:
        jaws_points.append(post_jaws[column])
        jaws_delta_means[count] = np.mean(post_jaws[column])
        jaws_errors[count] = scipy.stats.sem(post_jaws[column])
        count += 1
    jaws_delta_means[0] = np.mean(jaws_pre_df[column])
    jaws_errors[0] = scipy.stats.sem(jaws_pre_df[column])

    count = 1
    npas_points = [npas_pre_df[column]]
    npas_delta_means = np.zeros(len(post_npas_dfs) + 1)
    npas_errors = np.zeros(len(post_npas_dfs) + 1)
    for post_npas in post_npas_dfs:
        npas_points.append(post_npas[column])
        npas_delta_means[count] = np.mean(post_npas[column])
        npas_errors[count] = scipy.stats.sem(post_npas[column])
        count += 1
    npas_delta_means[0] = np.mean(npas_pre_df[column])
    npas_errors[0] = scipy.stats.sem(npas_pre_df[column])

    axes.plot(
        np.arange(jaws_delta_means.shape[0]),
        jaws_delta_means,
        marker="o",
        ls="dashed",
        color="b",
        markersize=4,
        lw=0.5,
    )
    axes.errorbar(
        np.arange(jaws_delta_means.shape[0]),
        jaws_delta_means,
        yerr=jaws_errors,
        color="b",
        capsize=2,
        lw=0.5,
    )

    axes.plot(
        np.arange(npas_delta_means.shape[0]),
        npas_delta_means,
        marker="o",
        ls="dashed",
        color="r",
        markersize=4,
        lw=0.5,
    )

    axes.errorbar(
        np.arange(npas_delta_means.shape[0]),
        npas_delta_means,
        yerr=npas_errors,
        color="r",
        capsize=2,
        lw=0.5,
    )

    axes.set_ylabel(column)
    axes.set_xticks(np.arange(npas_delta_means.shape[0]))
    axes.set_xticklabels(time_chunk_labels)
    axes.set_xlabel("Post-Stim Time (min)")


def plot_tracked_pre_post(motor_rescue_dfs):
    jaws_pre = motor_rescue_dfs[0]
    jaws_post = motor_rescue_dfs[1]
    npas_pre = motor_rescue_dfs[2]
    npas_post = motor_rescue_dfs[3]

    jaws_pre_post = np.loadtxt("../data/jaws_neurons/pre_post_units.txt", dtype=int)
    npas_pre_post = np.loadtxt("../data/npas_neurons/pre_post_units.txt", dtype=int)

    fig, ax = plt.subplots(4, 3, figsize=(12, 10), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]

    axis_count = 0
    for col in motor_rescue_dfs[0].columns:
        jaws_pre_post_values = np.zeros(jaws_pre_post.shape)
        for i in range(jaws_pre_post.shape[0]):
            jaws_pre_post_values[i, 0] = motor_rescue_dfs[0].iloc[
                jaws_pre_post[i, 0] - 1, motor_rescue_dfs[0].columns.get_loc(col)
            ]
            jaws_pre_post_values[i, 1] = motor_rescue_dfs[1].iloc[
                jaws_pre_post[i, 1] - 1, motor_rescue_dfs[1].columns.get_loc(col)
            ]

            axes[axis_count].plot(
                [0, 1],
                jaws_pre_post_values[i],
                color="gray",
                marker="o",
                alpha=0.5,
                lw=0.5,
            )

        axes[axis_count].plot(
            [0, 1], np.mean(jaws_pre_post_values, axis=0), color="b", marker="o", lw=0.5
        )
        axes[axis_count].errorbar(
            [0, 1],
            np.mean(jaws_pre_post_values, axis=0),
            yerr=scipy.stats.sem(jaws_pre_post_values, axis=0),
            color="b",
            capsize=4,
        )

        npas_pre_post_values = np.zeros(npas_pre_post.shape)
        for i in range(npas_pre_post.shape[0]):
            npas_pre_post_values[i, 0] = motor_rescue_dfs[2].iloc[
                npas_pre_post[i, 0] - 1, motor_rescue_dfs[2].columns.get_loc(col)
            ]
            npas_pre_post_values[i, 1] = motor_rescue_dfs[3].iloc[
                npas_pre_post[i, 1] - 1, motor_rescue_dfs[3].columns.get_loc(col)
            ]

            axes[axis_count].plot(
                [2, 3],
                npas_pre_post_values[i],
                color="gray",
                marker="o",
                alpha=0.5,
                lw=0.5,
            )
        axes[axis_count].plot(
            [2, 3], np.mean(npas_pre_post_values, axis=0), color="r", marker="o", lw=0.5
        )
        axes[axis_count].errorbar(
            [2, 3],
            np.mean(npas_pre_post_values, axis=0),
            yerr=scipy.stats.sem(npas_pre_post_values, axis=0),
            color="r",
            capsize=4,
        )
        axes[axis_count].set_ylabel(col)
        axes[axis_count].set_xticks(np.arange(4))
        axes[axis_count].set_xticklabels(
            ["Pre-Stim\nJAWS", "Post-Stim\nJAWS", "Pre-Stim\nNpas", "Post-Stim\nNpas"]
        )

        ylims = axes[axis_count].get_ylim()
        pval = scipy.stats.ttest_rel(
            jaws_pre_post_values[:, 0], jaws_pre_post_values[:, 1]
        ).pvalue
        plot_bracket(
            axes[axis_count],
            0,
            1,
            ylims[1] * 0.9,
            ylims[1] * 0.95,
            f"T-test: {pval:.3f}",
        )
        pval = scipy.stats.ttest_rel(
            npas_pre_post_values[:, 0], npas_pre_post_values[:, 1]
        ).pvalue
        plot_bracket(
            axes[axis_count],
            2,
            3,
            ylims[1] * 0.9,
            ylims[1] * 0.95,
            f"T-test: {pval:.3f}",
        )
        axes[axis_count].set_ylim(ylims)

        axis_count += 1

    makeNice(axes)
    fig.savefig("../data/pre_post_tracked.pdf", bbox_inches="tight")
    plt.close()
    run_cmd("open ../data/pre_post_tracked.pdf")


def feature_fail(
    jaws_pre_df,
    npas_pre_df,
    post_jaws_dfs,
    post_npas_dfs,
    ylabels,
    show=True,
):
    cols = jaws_pre_df.columns
    jaws_pvalues = np.zeros((len(post_jaws_dfs), cols.shape[0]))
    npas_pvalues = np.zeros(
        (
            len(post_npas_dfs),
            cols.shape[0],
        )
    )
    for i in range(jaws_pvalues.shape[0]):
        for j in range(jaws_pvalues.shape[1]):
            jaws_pvalues[i, j] = scipy.stats.ttest_ind(
                jaws_pre_df[cols[j]], post_jaws_dfs[i][cols[j]], equal_var=False
            ).pvalue
            npas_pvalues[i, j] = scipy.stats.ttest_ind(
                npas_pre_df[cols[j]], post_npas_dfs[i][cols[j]], equal_var=False
            ).pvalue
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=300, tight_layout=True)
    axes = [ax[i] for i in range(2)]
    sns.heatmap(
        jaws_pvalues,
        annot=True,
        ax=ax[0],
        fmt=".3f",
        annot_kws={"fontsize": 6},
        vmin=0,
        vmax=0.1,
        cmap="RdYlGn",
    )
    sns.heatmap(
        npas_pvalues,
        annot=True,
        ax=ax[1],
        fmt=".3f",
        annot_kws={"fontsize": 6},
        vmin=0,
        vmax=0.1,
        cmap="RdYlGn",
    )
    for i in range(2):
        axes[i].invert_yaxis()
        axes[i].set_xticks(np.arange(0.5, cols.shape[0] + 0.5))
        axes[i].set_xticklabels(cols, rotation=90)
        axes[i].set_yticks(np.arange(0.5, len(post_jaws_dfs) + 0.5))
        axes[i].set_yticklabels(ylabels, rotation=0)
        axes[i].set_ylabel("Post-Stim Time (mins)")
    fig.savefig(f"../data/feature_fail.pdf", bbox_inches="tight")
    plt.close()
    if show:
        run_cmd(f"open ../data/feature_fail.pdf")


def plot_bracket(ax, x1, x2, y1, y2, text, color="k", lw=0.5):
    ax.hlines(y2, x1, x2, color=color, lw=lw, zorder=15)
    ax.vlines([x1, x2], y1, y2, color=color, lw=lw, zorder=15)
    ax.annotate(
        text,
        xycoords="data",
        xy=((x1 + x2) / 2, y2 + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01),
        fontsize=6,
        zorder=15,
        ha="center",
    )


def plot_vertical_bracket(ax, x1, x2, y1, y2, text, color="k", lw=0.5):
    ax.hlines([y1, y2], x1, x2, color=color, lw=lw, zorder=15)
    ax.vlines(x2, y1, y2, color=color, lw=lw, zorder=15)
    ax.annotate(
        text,
        xycoords="data",
        xy=(x2 + 0.02, (y1 + y2) / 2),
        fontsize=6,
        zorder=15,
        ha="right",
    )


def plot_corr_matrix(df):

    naive_df = df[df["Type"] == 1]
    dd_df = df[df["Type"] == 0]
    df = df.drop(columns=["Type", "mouse", "folder", "name"])
    dd_df = dd_df.drop(columns=["Type", "mouse", "folder", "name"])
    naive_df = naive_df.drop(columns=["Type", "mouse", "folder", "name"])
    fig, ax = plt.subplots(1, 3, figsize=(12, 6), tight_layout=True, dpi=300)
    axes = [ax[i] for i in range(3)]
    sns.heatmap(
        df.corr(),
        cmap="coolwarm",
        ax=axes[0],
        vmin=-1,
        vmax=1,
        fmt=".2f",
        annot=True,
        annot_kws={"fontsize": 4},
        mask=np.triu(df.corr()),
        cbar_kws={"shrink": 0.5},
        cbar=False,
    )

    sns.heatmap(
        naive_df.corr(),
        cmap="coolwarm",
        ax=axes[1],
        vmin=-1,
        vmax=1,
        fmt=".2f",
        annot=True,
        annot_kws={"fontsize": 4},
        mask=np.triu(naive_df.corr()),
        cbar_kws={"shrink": 0.5},
        cbar=False,
    )
    sns.heatmap(
        dd_df.corr(),
        cmap="coolwarm",
        ax=axes[2],
        vmin=-1,
        vmax=1,
        fmt=".2f",
        annot=True,
        annot_kws={"fontsize": 4},
        mask=np.triu(dd_df.corr()),
        cbar_kws={"shrink": 0.5},
        cbar=False,
    )
    axes[0].set_title("All")
    axes[1].set_title("Naive")
    axes[2].set_title("DD")
    makeNice(axes, labelsize=6)
    fig.savefig("../data/corr_matrix.pdf")
    plt.close()


def feature_time(
    motor_rescue_dfs, post_jaws_dfs, post_npas_dfs, time_chunks_labels, show=False
):
    fig, ax = plt.subplots(4, 3, figsize=(12, 8), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    axis_count = 0
    for col in motor_rescue_dfs[0].columns[0:12]:
        if col != "Type":
            plot_column_comparison_over_time(
                axes[axis_count],
                col,
                motor_rescue_dfs[0],
                motor_rescue_dfs[2],
                post_jaws_dfs,
                post_npas_dfs,
                time_chunks_labels,
            )
            axis_count += 1
    makeNice(axes, labelsize=8)
    fig.savefig(f"../data/motor_rescue_features_over_time.pdf", bbox_inches="tight")
    plt.close()
    if show:
        run_cmd("open ../data/motor_rescue_features_over_time.pdf")


def bursts_delta(df):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300, tight_layout=True)
    axes = [ax[i] for i in range(2)]
    naive_df = df[df["Type"] == 1]
    dd_df = df[df["Type"] == 0]
    df = df.drop(columns=["Type"])
    dd_df = dd_df.drop(columns=["Type"])
    naive_df = naive_df.drop(columns=["Type"])

    """df = df[(np.abs(stats.zscore(df["Delta Power"])) < 3)]
    dd_df = dd_df[(np.abs(stats.zscore(dd_df["Delta Power"])) < 3)]
    naive_df = naive_df[(np.abs(stats.zscore(naive_df["Delta Power"])) < 3)]"""

    sns.regplot(
        df, x="Delta Power", y="Num Bursts", ax=axes[0], color="gray", scatter=False
    )
    sns.regplot(
        naive_df,
        x="Delta Power",
        y="Num Bursts",
        ax=axes[0],
        color="r",
        scatter_kws={"s": 2, "alpha": 0.5},
    )
    sns.regplot(
        dd_df,
        x="Delta Power",
        y="Num Bursts",
        ax=axes[0],
        color="b",
        scatter_kws={"s": 2, "alpha": 0.5},
    )
    sns.regplot(
        df, x="Beta Power", y="Num Bursts", ax=axes[1], color="gray", scatter=False
    )
    sns.regplot(
        naive_df,
        x="Beta Power",
        y="Num Bursts",
        ax=axes[1],
        color="r",
        scatter_kws={"s": 2, "alpha": 0.5},
    )
    sns.regplot(
        dd_df,
        x="Beta Power",
        y="Num Bursts",
        ax=axes[1],
        color="b",
        scatter_kws={"s": 2, "alpha": 0.5},
    )

    makeNice(axes, labelsize=8)
    fig.savefig("../data/bursts_delta.pdf")
    plt.close()


def plot_feature_histogram_comparison(df, motor_rescue_dfs):
    fig, ax = plt.subplots(4, 3, figsize=(12, 8), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    i = 0
    motor_labels = [
        f"Jaws-Pre",
        # f"Jaws-Post",
        f"Npas-Pre",
        # f"Npas-Post",
    ]
    motor_rescue_dfs = [motor_rescue_dfs[0], motor_rescue_dfs[2]]
    for col in df.columns:
        if col != "Type" and col != "T" and "Osc" not in col and "_z" not in col:
            bins = np.histogram_bin_edges(df[col], bins="auto")
            sns.histplot(
                df[df["Type"] == 0],
                x=col,
                ax=axes[i],
                legend=True,
                kde=True,
                stat="probability",
                edgecolor="w",
                color="b",
                bins=bins,
                label="DD",
            )

            """sns.histplot(
                df,
                x=col,
                ax=axes[i],
                hue="Type",
                legend=False,
                kde=True,
                stat="probability",
                edgecolor="w",
                palette=["b", "r"],
                bins=bins,
            )"""
            dd_col = df[df["Type"] == 0][[col]]

            ylims = axes[i].get_ylim()
            xlims = axes[i].get_xlim()
            axes[i].vlines(
                np.mean(dd_col),
                ylims[0],
                ylims[1],
                color="b",
                linestyle="dashed",
            )
            """naive_col = df[df["Type"] == 1][[col]]
            axes[i].vlines(
                np.mean(naive_col),
                ylims[0],
                ylims[1],
                color="r",
                linestyle="dashed",
            )
            axes[i].vlines(
                np.mean(df[col]),
                ylims[0],
                ylims[1],
                color="k",
                linestyle="dashed",
            )"""
            motor_count = 0
            for motor_df in motor_rescue_dfs:
                sns.histplot(
                    motor_df,
                    x=col,
                    ax=axes[i],
                    kde=True,
                    stat="probability",
                    edgecolor="w",
                    label=motor_labels[motor_count],
                    bins=bins,
                    legend=True,
                )
                motor_count += 1
            axes[i].legend()

            i += 1
    makeNice(axes)
    fig.savefig("../data/feature_comparison.pdf")
    plt.close()


def plot_ephys_pre_post(pre, post, name):
    fig, ax = plt.subplots(4, 3, figsize=(12, 8), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    i = 0

    for col in pre.columns:
        if col != "Type" and col != "T" and "Osc" not in col and "_z" not in col:
            pre_col = pre[col]
            post_col = post[col]
            bins = np.histogram_bin_edges(pre_col, bins="auto")
            sns.histplot(
                x=pre_col,
                ax=axes[i],
                kde=True,
                stat="probability",
                edgecolor="w",
                bins=bins,
                legend=True,
                color="r",
                label=f"Pre {name}",
            )
            sns.histplot(
                x=post_col,
                ax=axes[i],
                kde=True,
                stat="probability",
                edgecolor="w",
                bins=bins,
                legend=True,
                color="b",
                label=f"Post {name}",
            )
            axes[i].legend()
            xlims = axes[i].get_xlim()
            ylims = axes[i].get_ylim()

            ks_pval = ks_2samp(pre_col, post_col).pvalue
            plot_bracket(
                axes[i],
                xlims[0] + (xlims[1] - xlims[0]) * 0.25,
                xlims[1] - (xlims[1] - xlims[0]) * 0.25,
                ylims[0] + 1.15 * (ylims[1] - ylims[0]),
                ylims[0] + 1.18 * (ylims[1] - ylims[0]),
                f"KS $p{'*' if ks_pval < 0.05 else ''}=${ks_pval:.3f}",
            )

            ttest_pval = stats.ttest_ind(pre_col, post_col).pvalue
            plot_bracket(
                axes[i],
                np.mean(pre_col),
                np.mean(post_col),
                ylims[1],
                ylims[0] + 1.03 * (ylims[1] - ylims[0]),
                f"T-test $p{'*' if ttest_pval < 0.05 else ''}=${ttest_pval:.3f}",
            )
            i += 1
    makeNice(axes)
    fig.savefig(f"../data/ephys_pre_post_{name}.pdf")
    plt.close()


def plot_feature_histograms_separate(df, outlier_dict):
    dd_color = "r"
    naive_color = "gray"
    df = clean_data.remove_outliers_by_group_zscore_independent(
        df[df["Type"] == 1],
        df[df["Type"] == 0],
        outlier_dict,
    )
    i = 0
    for col in df.columns[0:12]:
        if col != "Type" and col != "T" and "Osc" not in col and "_z" not in col:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300, tight_layout=True)
            fig2, ax2 = plt.subplots(1, 1, figsize=(3, 2), dpi=300, tight_layout=True)
            bins = np.histogram_bin_edges(df[col], bins=20)
            sns.ecdfplot(
                df[df["Type"] == 0],
                x=col,
                ax=ax2,
                legend=False,
                stat="proportion",
                color=dd_color,
            )
            sns.ecdfplot(
                df[df["Type"] == 1],
                x=col,
                ax=ax2,
                legend=False,
                stat="proportion",
                color=naive_color,
            )
            sns.histplot(
                df[df["Type"] == 1],
                x=col,
                ax=ax,
                legend=False if i != 0 else True,
                kde=True,
                stat="probability",
                edgecolor="w",
                linewidth=0.25,
                color=naive_color,
                bins=bins,
                label="Naive",
                line_kws={"lw": 0.5},
            )
            sns.histplot(
                df[df["Type"] == 0],
                x=col,
                ax=ax,
                legend=False if i != 0 else True,
                kde=True,
                stat="probability",
                edgecolor="w",
                linewidth=0.25,
                color=dd_color,
                bins=bins,
                label="DD",
                line_kws={"lw": 0.5},
            )
            dd_col = df[df["Type"] == 0][[col]]
            naive_col = df[df["Type"] == 1][[col]]
            ylims = ax.get_ylim()
            xlims = ax.get_xlim()
            ax.vlines(
                np.mean(dd_col),
                ylims[0],
                ylims[1],
                color=dd_color,
                linestyle="dashed",
                lw=0.5,
            )
            ax.vlines(
                np.mean(naive_col),
                ylims[0],
                ylims[1],
                color=naive_color,
                linestyle="dashed",
                lw=0.5,
            )
            ax2.vlines(
                np.mean(dd_col), 0, 1, color=dd_color, linestyle="dashed", lw=0.5
            )
            ax2.vlines(
                np.mean(naive_col), 0, 1, color=naive_color, linestyle="dashed", lw=0.5
            )
            ax2.vlines(
                np.median(dd_col), 0, 1, color=dd_color, linestyle="dotted", lw=0.5
            )
            ax2.vlines(
                np.median(naive_col),
                0,
                1,
                color=naive_color,
                linestyle="dotted",
                lw=0.5,
            )
            ax.vlines(
                np.mean(df[col]),
                ylims[0],
                ylims[1],
                color="k",
                linestyle="dashed",
                lw=0.5,
            )
            ks_pval = ks_2samp(dd_col, naive_col).pvalue
            plot_bracket(
                ax,
                xlims[0] + (xlims[1] - xlims[0]) * 0.25,
                xlims[1] - (xlims[1] - xlims[0]) * 0.25,
                ylims[0] + 1.15 * (ylims[1] - ylims[0]),
                ylims[0] + 1.18 * (ylims[1] - ylims[0]),
                f"KS $p{'*' if ks_pval[0] < 0.05 else ''}=${ks_pval[0]:.3f}",
            )

            ttest_pval = stats.ttest_ind(dd_col, naive_col).pvalue
            plot_bracket(
                ax,
                np.mean(naive_col),
                np.mean(dd_col),
                ylims[1],
                ylims[0] + 1.03 * (ylims[1] - ylims[0]),
                f"T-test $p{'*' if ttest_pval[0] < 0.05 else ''}=${ttest_pval[0]:.3f}",
            )
            i += 1
            makeNice(ax)
            fig.savefig(f"../data/features_{col}.pdf")
            plt.close()

            makeNice(ax2)
            fig2.savefig(f"../data/features_cdf_{col}.pdf")
            plt.close()


def plot_feature_histograms(df):
    fig, ax = plt.subplots(4, 3, figsize=(12, 8), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    fig2, ax2 = plt.subplots(4, 3, figsize=(12, 8), dpi=300, tight_layout=True)
    axes2 = [ax2[i, j] for i in range(4) for j in range(3)]
    i = 0
    for col in df.columns[0:12]:
        if col != "Type" and col != "T" and "Osc" not in col and "_z" not in col:
            bins = np.histogram_bin_edges(df[col], bins=20)
            sns.ecdfplot(
                df[df["Type"] == 0],
                x=col,
                ax=axes2[i],
                legend=False,
                stat="proportion",
                color="b",
            )
            sns.ecdfplot(
                df[df["Type"] == 1],
                x=col,
                ax=axes2[i],
                legend=False,
                stat="proportion",
                color="r",
            )
            sns.histplot(
                df[df["Type"] == 0],
                x=col,
                ax=axes[i],
                legend=False if i != 0 else True,
                kde=True,
                stat="probability",
                edgecolor="w",
                color="b",
                bins=bins,
                label="DD",
            )
            sns.histplot(
                df[df["Type"] == 1],
                x=col,
                ax=axes[i],
                legend=False if i != 0 else True,
                kde=True,
                stat="probability",
                edgecolor="w",
                color="r",
                bins=bins,
                label="Naive",
            )
            if i == 0:
                axes[i].legend()
            dd_col = df[df["Type"] == 0][[col]]
            naive_col = df[df["Type"] == 1][[col]]
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
            axes[i].vlines(
                np.mean(df[col]),
                ylims[0],
                ylims[1],
                color="k",
                linestyle="dashed",
            )
            ks_pval = ks_2samp(dd_col, naive_col).pvalue
            plot_bracket(
                axes[i],
                xlims[0] + (xlims[1] - xlims[0]) * 0.25,
                xlims[1] - (xlims[1] - xlims[0]) * 0.25,
                ylims[0] + 1.15 * (ylims[1] - ylims[0]),
                ylims[0] + 1.18 * (ylims[1] - ylims[0]),
                f"KS $p{'*' if ks_pval[0] < 0.05 else ''}=${ks_pval[0]:.3f}",
            )

            ttest_pval = stats.ttest_ind(dd_col, naive_col).pvalue
            plot_bracket(
                axes[i],
                np.mean(naive_col),
                np.mean(dd_col),
                ylims[1],
                ylims[0] + 1.03 * (ylims[1] - ylims[0]),
                f"T-test $p{'*' if ttest_pval[0] < 0.05 else ''}=${ttest_pval[0]:.3f}",
            )
            i += 1
    makeNice(axes)
    fig.savefig("../data/features.pdf")

    makeNice(axes2)
    fig2.savefig("../data/features_cdf.pdf")
    plt.close()


def plot_feature_histograms_normed_together(df):
    fig, ax = plt.subplots(4, 3, figsize=(12, 8), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    i = 0
    norm_df = df.apply(stats.zscore)
    # df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    for col in norm_df.columns:
        if col != "Type" and col != "T" and "Osc" not in col and "_z" not in col:
            sns.histplot(
                norm_df,
                x=col,
                ax=axes[i],
                hue="Type",
                legend=False,
                kde=True,
                stat="probability",
                edgecolor="w",
                palette=["b", "r"],
            )

            dd_col = norm_df[df["Type"] == 0][[col]]
            naive_col = norm_df[df["Type"] == 1][[col]]
            axes[i].set_xlabel(f"{col}")
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
            axes[i].vlines(
                np.mean(norm_df[col]),
                ylims[0],
                ylims[1],
                color="k",
                linestyle="dashed",
            )
            ks_pval = ks_2samp(dd_col, naive_col).pvalue
            plot_bracket(
                axes[i],
                xlims[0] + (xlims[1] - xlims[0]) * 0.25,
                xlims[1] - (xlims[1] - xlims[0]) * 0.25,
                ylims[0] + 1.15 * (ylims[1] - ylims[0]),
                ylims[0] + 1.18 * (ylims[1] - ylims[0]),
                f"KS $p{'*' if ks_pval[0] < 0.05 else ''}=${ks_pval[0]:.3f}",
            )

            ttest_pval = stats.ttest_ind(dd_col, naive_col).pvalue
            plot_bracket(
                axes[i],
                np.mean(naive_col),
                np.mean(dd_col),
                ylims[1],
                ylims[0] + 1.03 * (ylims[1] - ylims[0]),
                f"T-test $p{'*' if ttest_pval[0] < 0.05 else ''}=${ttest_pval[0]:.3f}",
            )
            i += 1
    makeNice(axes)
    fig.savefig("../data/norm_features.pdf")
    plt.close()


def plot_feature_histograms_normed_by_group(df):
    fig, ax = plt.subplots(4, 3, figsize=(12, 8), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    i = 0
    # df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    for col in df.columns:
        if col != "Type" and col != "T" and "Osc" not in col and "_z" not in col:
            dd_col = df[df["Type"] == 0][[col]]
            naive_col = df[df["Type"] == 1][[col]]
            norm_dd_col = stats.zscore(dd_col)
            norm_naive_col = stats.zscore(naive_col)
            norm_dd_col["Type"] = 0
            norm_naive_col["Type"] = 1
            plot_pd = pd.concat([norm_dd_col, norm_naive_col])
            sns.histplot(
                plot_pd,
                x=col,
                ax=axes[i],
                hue="Type",
                legend=False,
                kde=True,
                stat="probability",
                edgecolor="w",
                palette=["b", "r"],
            )
            dd_col = plot_pd[plot_pd["Type"] == 0][[col]]
            naive_col = plot_pd[plot_pd["Type"] == 1][[col]]
            axes[i].set_xlabel(f"{col}")
            i += 1
    makeNice(axes)
    fig.savefig("../data/group_norm_features.pdf")
    plt.close()


def plot_outliers_removed_cv_fr_by_group_zscore(
    naive_df, dd_df, threshold=3, show=False
):
    fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    ylabels = []
    count = 0
    naive_copy_df = naive_df
    naive_copy_df["FR_z"] = np.abs(stats.zscore(naive_df["FR"]))
    naive_copy_df["CV_z"] = np.abs(stats.zscore(naive_df["CV"]))
    naive_outliers = naive_df[
        (naive_copy_df["CV_z"] >= threshold) | (naive_copy_df["FR_z"] >= threshold)
    ]
    dd_copy_df = dd_df
    dd_copy_df["FR_z"] = np.abs(stats.zscore(dd_df["FR"]))
    dd_copy_df["CV_z"] = np.abs(stats.zscore(dd_df["CV"]))
    dd_outliers = dd_df[
        (dd_copy_df["CV_z"] >= threshold) | (dd_copy_df["FR_z"] >= threshold)
    ]

    for col in naive_df.columns:
        if col != "Type" and col != "T" and "Osc" not in col and "_z" not in col:
            ylabels.append(col)
            col_mean = np.mean(naive_df[col])
            col_std = np.std(naive_df[col])
            axes[count].plot(
                [col_mean - col_std * threshold, col_mean + col_std * threshold],
                [1, 1],
                color="r",
                marker="|",
                lw=0.5,
            )
            axes[count].scatter(col_mean, 1, color="r", s=8)
            if col == "FR" or col == "CV":
                plot_outliers = naive_outliers[
                    (naive_outliers[col] >= col_mean + col_std * threshold)
                    | (naive_outliers[col] <= col_mean - col_std * threshold)
                ][col]
                axes[count].scatter(
                    plot_outliers,
                    np.ones(len(plot_outliers)),
                    color="r",
                    s=6,
                    marker="x",
                    zorder=5,
                )
            axes[count].scatter(
                naive_df[col],
                np.ones(len(naive_df[col])),
                color="gray",
                s=6,
                marker=".",
                alpha=0.5,
            )
            col_mean = np.mean(dd_df[col])
            col_std = np.std(dd_df[col])
            axes[count].plot(
                [col_mean - col_std * threshold, col_mean + col_std * threshold],
                [0, 0],
                color="b",
                marker="|",
                lw=0.5,
            )
            axes[count].scatter(col_mean, 0, color="b", s=8)
            if col == "FR" or col == "CV":
                plot_outliers = dd_outliers[
                    (dd_outliers[col] >= col_mean + col_std * threshold)
                    | (dd_outliers[col] <= col_mean - col_std * threshold)
                ][col]
                axes[count].scatter(
                    plot_outliers,
                    np.zeros(len(plot_outliers)),
                    color="b",
                    s=6,
                    marker="x",
                    zorder=5,
                )

            axes[count].scatter(
                dd_df[col],
                np.zeros(len(dd_df[col])),
                color="gray",
                s=6,
                marker=".",
                alpha=0.5,
            )
            axes[count].set_yticks([0, 1])
            axes[count].set_yticklabels([f"DD", "Naive"])
            axes[count].set_ylim([-1, 2])
            axes[count].set_xlabel(col, fontsize=8)

            count += 1
    makeNice(axes, labelsize=6)
    fig.savefig(f"../data/outliers_removed_cv_fr_by_group_zscore_{threshold}.pdf")
    plt.close()
    if show:
        run_cmd(f"open ../data/outliers_removed_cv_fr_by_group_zscore_{threshold}.pdf")


def plot_outliers_removed_collectively_zscore(df, threshold=3, show=False):
    fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    ylabels = []
    count = 0
    naive_df = df[df["Type"] == 1]
    dd_df = df[df["Type"] == 0]
    outlier_df = df[(np.abs(stats.zscore(df)) >= threshold).any(axis=1)]
    naive_outliers = outlier_df[outlier_df["Type"] == 1]
    dd_outliers = outlier_df[outlier_df["Type"] == 0]

    for col in df.columns:
        if col != "Type" and col != "T" and "Osc" not in col and "_z" not in col:
            ylabels.append(col)
            col_mean = np.mean(df[col])
            col_std = np.std(df[col])
            axes[count].plot(
                [col_mean - col_std * threshold, col_mean + col_std * threshold],
                [1, 1],
                color="r",
                marker="|",
                lw=0.5,
            )
            axes[count].scatter(col_mean, 1, color="r", s=4)
            plot_outliers = naive_outliers[
                (naive_outliers[col] >= col_mean + col_std * threshold)
                | (naive_outliers[col] <= col_mean - col_std * threshold)
            ][col]
            axes[count].scatter(
                plot_outliers,
                np.ones(len(plot_outliers)),
                color="r",
                s=6,
                marker="x",
                zorder=5,
            )
            axes[count].scatter(
                naive_df[col],
                np.ones(len(naive_df[col])),
                color="gray",
                s=6,
                marker=".",
                alpha=0.5,
            )

            col_mean = np.mean(df[col])
            col_std = np.std(df[col])
            axes[count].plot(
                [col_mean - col_std * threshold, col_mean + col_std * threshold],
                [0, 0],
                color="b",
                marker="|",
                lw=0.5,
            )
            axes[count].scatter(col_mean, 0, color="b", s=4)
            """axes[count].scatter(
                dd_outliers[col],
                np.zeros(len(dd_outliers[col])),
                color="b",
                s=6,
                marker="x",
            )"""
            plot_outliers = dd_outliers[
                (dd_outliers[col] >= col_mean + col_std * threshold)
                | (dd_outliers[col] <= col_mean - col_std * threshold)
            ][col]
            axes[count].scatter(
                plot_outliers,
                np.zeros(len(plot_outliers)),
                color="b",
                s=6,
                marker="x",
                zorder=5,
            )

            axes[count].scatter(
                dd_df[col],
                np.zeros(len(dd_df[col])),
                color="gray",
                s=6,
                marker=".",
                alpha=0.5,
            )
            axes[count].set_yticks([0, 1])
            axes[count].set_yticklabels(["DD", "Naive"])
            axes[count].set_ylim([-1, 2])
            axes[count].set_xlabel(col, fontsize=8)

            count += 1
    makeNice(axes, labelsize=6)
    fig.savefig(f"../data/outliers_removed_collectively_zscore_{threshold}.pdf")
    plt.close()
    if show:
        run_cmd(f"open ../data/outliers_removed_collectively_zscore_{threshold}.pdf")


def plot_outliers_removed_by_group_zscore(naive_df, dd_df, threshold=3, show=False):
    fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]
    ylabels = []
    count = 0
    df = pd.concat([naive_df, dd_df])

    naive_df = df[df["Type"] == 1]
    dd_df = df[df["Type"] == 0]

    naive_outliers = naive_df.iloc[:, 1:]
    naive_outliers = naive_outliers[
        (np.abs(stats.zscore(naive_outliers)) >= threshold).any(axis=1)
    ]

    dd_outliers = dd_df.iloc[:, 1:]
    dd_outliers = dd_outliers[
        (np.abs(stats.zscore(dd_outliers)) >= threshold).any(axis=1)
    ]

    for col in df.columns:
        if col != "Type" and col != "T" and "Osc" not in col and "_z" not in col:
            ylabels.append(col)
            col_mean = np.mean(naive_df[col])
            col_std = np.std(naive_df[col])
            axes[count].plot(
                [col_mean - col_std * threshold, col_mean + col_std * threshold],
                [1, 1],
                color="r",
                marker="|",
                lw=0.5,
            )
            axes[count].scatter(col_mean, 1, color="r", s=4)
            plot_outliers = naive_outliers[
                (naive_outliers[col] >= col_mean + col_std * threshold)
                | (naive_outliers[col] <= col_mean - col_std * threshold)
            ][col]
            axes[count].scatter(
                plot_outliers,
                np.ones(len(plot_outliers)),
                color="r",
                s=6,
                marker="x",
                zorder=5,
            )
            axes[count].scatter(
                naive_df[col],
                np.ones(len(naive_df[col])),
                color="gray",
                s=6,
                marker=".",
                alpha=0.5,
            )

            col_mean = np.mean(dd_df[col])
            col_std = np.std(dd_df[col])
            axes[count].plot(
                [col_mean - col_std * threshold, col_mean + col_std * threshold],
                [0, 0],
                color="b",
                marker="|",
                lw=0.5,
            )
            axes[count].scatter(col_mean, 0, color="b", s=4)

            plot_outliers = dd_outliers[
                (dd_outliers[col] >= col_mean + col_std * threshold)
                | (dd_outliers[col] <= col_mean - col_std * threshold)
            ][col]
            axes[count].scatter(
                plot_outliers,
                np.zeros(len(plot_outliers)),
                color="b",
                s=6,
                marker="x",
                zorder=5,
            )

            axes[count].scatter(
                dd_df[col],
                np.zeros(len(dd_df[col])),
                color="gray",
                s=6,
                marker=".",
                alpha=0.5,
            )
            axes[count].set_yticks([0, 1])
            axes[count].set_yticklabels(["DD", "Naive"])
            axes[count].set_ylim([-1, 2])
            axes[count].set_xlabel(col, fontsize=8)

            count += 1
    makeNice(axes, labelsize=6)
    fig.savefig(f"../data/outliers_removed_by_group_zscore_{threshold}.pdf")
    plt.close()
    if show:
        run_cmd(f"open ../data/outliers_removed_by_group_zscore_{threshold}.pdf")
