import numpy as np
import json
import poisson_surprise
from helpers import *
import os
import sys
import pandas as pd

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
import sys

import parse_data

# Reduce all font sizes globally
plt.rcParams.update(
    {
        "font.size": 8,  # Base font size
        "axes.titlesize": 6,  # Title font size
        "axes.labelsize": 8,  # Axis label font size
        "xtick.labelsize": 6,  # X-axis tick labels
        "ytick.labelsize": 6,  # Y-axis tick labels
        "legend.fontsize": 6,  # Legend font size
    }
)

# user modules
import clean_data

# set plotting parameters
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def get_naive_dd_training_data_set(renewal_power=False):
    """
    Generates naive and DD and a combined dataframe of all spike train features.

    Returns
    -------
    \t naive_df (dataframe) : feature dataframe of all naive data

    \t dd_df (dataframe) : feature dataframe of all DD data

    \t combined_df (dataframe) : feature dataframe of all naive and DD data
    """

    # Load oscillation data
    naive_osc_data = np.loadtxt(
        "../data/training/naive_neurons/osc_data.txt", delimiter=","
    )
    # Load naive spike train statistics
    naive_frs = np.loadtxt("../data/training/naive_neurons/cell_baseline_frs.txt")
    naive_cvs = np.loadtxt("../data/training/naive_neurons/cell_baseline_cvs.txt")
    naive_lengths = np.loadtxt(
        "../data/training/naive_neurons/cell_baseline_lengths.txt"
    )

    naive_burst_statistics = np.loadtxt(
        "../data/training/naive_neurons/cell_baseline_burst_statistics.txt"
    )
    # Construct naive dataframe
    naive_df = pd.DataFrame(
        {
            "FR": naive_frs,
            "CV": naive_cvs,
            # "T": naive_lengths, # feature not used
            # "Delta Osc": naive_osc_data[:, 0], # feature not used
            "Delta Power": naive_osc_data[:, 1 if renewal_power else 2],
            # "Beta Osc": naive_osc_data[:, 4], # feature not used
            "Beta Power": naive_osc_data[:, 5 if renewal_power else 6],
            "Num Bursts": naive_burst_statistics[:, 0] / naive_lengths,
            "Avg Burst Firing Rate": naive_burst_statistics[:, 1],
            "Percent Time Bursting": naive_burst_statistics[:, 2],
            "Percent of Spikes in Bursts": naive_burst_statistics[:, 3],
            "Avg Burst Duration": naive_burst_statistics[:, 4],
            "Avg Interburst Interval": naive_burst_statistics[:, 5],
            # "CV Interburst Interval": naive_burst_statistics[:, 6], # feature not used
            # "Avg Surprise": naive_burst_statistics[:, 7], # feature not used
            "Non Bursting Firing Rate": naive_burst_statistics[:, 8],
            "Burst Firing Rate Increase": naive_burst_statistics[:, 9],
        }
    )

    # Replace NaNs with 0s
    naive_df.fillna(0, inplace=True)

    return naive_df


def gather_unilateral_data(path):
    spikes = []
    for dir in os.listdir(path):
        if dir != ".DS_Store":
            for csv in os.listdir(f"{path}/{dir}"):
                if csv != ".DS_Store":
                    spike_csv = pd.read_csv(f"{path}/{dir}/{csv}")
                    for col in spike_csv.columns:
                        spikes.append(spike_csv[col].dropna().values)
    return spikes


def process_data(spikes, fixed_length, spike_path):
    spikes = [st[st < fixed_length] for st in spikes]
    for i in range(len(spikes)):
        np.savetxt(
            f"{spike_path}/Neuron_{int(i):04d}.txt",
            spikes[i],
            newline="\n",
            delimiter="\t",
            fmt="%f",
        )
    frs = parse_data.calc_firing_rates(
        spikes,
        np.ones(len(spikes)) * fixed_length,
    )
    cvs = parse_data.calc_cv(spikes)
    burst_statistics = parse_data.calc_burst_statistics(
        spikes,
        frs,
        np.ones(len(spikes)) * fixed_length,
        min_spikes=3,
        surprise_threshold=3,
        window=2,
    )

    parse_data.save_stat(frs, f"{save_path}/cell_frs.txt")
    parse_data.save_stat(cvs, f"{save_path}/cell_cvs.txt")
    parse_data.save_stat(
        burst_statistics,
        f"{save_path}/cell_burst_statistics.txt",
    )
    parse_data.save_stat(
        np.ones(len(spikes)) * fixed_length,
        f"{save_path}/cell_lengths.txt",
    )

    run_cmd(
        f"/Applications/MATLAB_R2021b.app/bin/matlab -nojvm -nodesktop -batch 'get_osc_data_unilateral'"
    )


def get_unilateral_data(renewal_power=False, mlp_seeds=15):
    uni_osc_data = np.loadtxt("../data/unilateral/osc_data.txt", delimiter=",")
    uni_frs = np.loadtxt("../data/unilateral/cell_frs.txt")
    uni_cvs = np.loadtxt("../data/unilateral/cell_cvs.txt")
    uni_lengths = np.loadtxt("../data/unilateral/cell_lengths.txt")
    uni_burst_statistics = np.loadtxt("../data/unilateral/cell_burst_statistics.txt")

    uni_df = pd.DataFrame(
        {
            "FR": uni_frs,
            "CV": uni_cvs,
            # "T": uni_lengths, # feature not used
            # "Delta Osc": uni_osc_data[:, 0], # feature not used
            "Delta Power": uni_osc_data[:, 1 if renewal_power else 2],
            # "Beta Osc": uni_osc_data[:, 4], # feature not used
            "Beta Power": uni_osc_data[:, 5 if renewal_power else 6],
            "Num Bursts": uni_burst_statistics[:, 0] / uni_lengths,
            "Avg Burst Firing Rate": uni_burst_statistics[:, 1],
            "Percent Time Bursting": uni_burst_statistics[:, 2],
            "Percent of Spikes in Bursts": uni_burst_statistics[:, 3],
            "Avg Burst Duration": uni_burst_statistics[:, 4],
            "Avg Interburst Interval": uni_burst_statistics[:, 5],
            # "CV Interburst Interval": uni_burst_statistics[:, 6], # feature not used
            # "Avg Surprise": uni_burst_statistics[:, 7], # feature not used
            "Non Bursting Firing Rate": uni_burst_statistics[:, 8],
            "Burst Firing Rate Increase": uni_burst_statistics[:, 9],
        }
    )

    bi_df = pd.read_csv("../data/neural_net/jaws_npas_pre_post.csv")
    bi_df = bi_df[bi_df["Post-Time"] == 0]
    bi_df = bi_df.iloc[:, 1:13]
    naive_df = get_naive_dd_training_data_set()

    predicts = np.zeros(mlp_seeds)
    naive_predicts = np.zeros(mlp_seeds)
    bi_predicts = np.zeros(mlp_seeds)

    dd_probs = np.zeros((mlp_seeds, len(uni_df)))
    bi_probs = np.zeros((mlp_seeds, len(bi_df)))
    naive_probs = np.zeros((mlp_seeds, len(naive_df)))

    for i in range(mlp_seeds):
        seed_train_data = pd.read_csv(
            f"../data/neural_net/X_train_seed_{int(i):02d}.csv"
        )
        seed_train_data = seed_train_data.drop(
            ["Unnamed: 0", "DD Probability", "Type"], axis=1
        )
        X_train_norm, X_test_norm = clean_data.normalize_data(
            seed_train_data, uni_df, min_max=False
        )

        _, bi_test_norm = clean_data.normalize_data(
            seed_train_data, bi_df, min_max=False
        )

        _, naive_test_norm = clean_data.normalize_data(
            seed_train_data, naive_df, min_max=False
        )

        with open(f"../data/neural_net/MLP_seed_{int(i):02d}.pkl", "rb") as file:
            clf = pickle.load(file)
            predict_test = clf.predict(X_test_norm)
            predicts[i] = 1 - np.sum(predict_test) / len(X_test_norm)
            dd_probs[i, :] = clf.predict_proba(X_test_norm)[:, 0]

            bi_predict_test = clf.predict(bi_test_norm)
            bi_predicts[i] = 1 - np.sum(bi_predict_test) / len(bi_test_norm)
            bi_probs[i, :] = clf.predict_proba(bi_test_norm)[:, 0]

            naive_predict_test = clf.predict(naive_test_norm)
            naive_predicts[i] = 1 - np.sum(naive_predict_test) / len(naive_test_norm)
            naive_probs[i, :] = clf.predict_proba(naive_test_norm)[:, 0]

    fig, ax = plt.subplots(1, 2, figsize=(4, 3), dpi=300, tight_layout=True)
    axes = [ax[i] for i in range(2)]
    axes[0].errorbar(
        x=0,
        y=np.mean(naive_predicts),
        yerr=scipy.stats.sem(naive_predicts),
        marker="o",
        color="k",
        markersize=2,
    )

    axes[0].errorbar(
        x=1,
        y=np.mean(predicts),
        yerr=scipy.stats.sem(predicts),
        marker="o",
        color="k",
        markersize=2,
    )
    axes[0].errorbar(
        x=2,
        y=np.mean(bi_predicts),
        yerr=scipy.stats.sem(bi_predicts),
        marker="o",
        color="k",
        markersize=2,
    )
    average_probs = np.mean(dd_probs, axis=0)
    bi_average_probs = np.mean(bi_probs, axis=0)
    naive_average_probs = np.mean(naive_probs, axis=0)

    naive_df["DD Confidence"] = naive_average_probs
    bi_df["DD Confidence"] = bi_average_probs
    uni_df["DD Confidence"] = average_probs
    df1_labeled = naive_df.copy()
    df1_labeled["Dataset"] = "Naive"

    df2_labeled = uni_df.copy()
    df2_labeled["Dataset"] = "Uni"

    df3_labeled = bi_df.copy()
    df3_labeled["Dataset"] = "Bi"

    df_all = pd.concat([df1_labeled, df2_labeled, df3_labeled], ignore_index=True)
    df_all.to_csv("../data/unilateral/naive_uni_bi_data.csv")
    print(df_all.columns)

    sys.exit()

    dd_prob_splits = [[0, 0.50], [0.50, 0.67], [0.67, 0.83], [0.83, 1]]
    dd_colors = ["gray", "lightblue", "b", "darkblue"]
    prob_splits = np.zeros(4)
    bi_prob_splits = np.zeros(4)
    naive_prob_splits = np.zeros(4)
    for i in range(4):
        prob_splits[i] = len(
            average_probs[
                (average_probs >= dd_prob_splits[i][0])
                & (average_probs < dd_prob_splits[i][1])
            ]
        ) / len(average_probs)

        bi_prob_splits[i] = len(
            bi_average_probs[
                (bi_average_probs >= dd_prob_splits[i][0])
                & (bi_average_probs < dd_prob_splits[i][1])
            ]
        ) / len(bi_average_probs)

        naive_prob_splits[i] = len(
            naive_average_probs[
                (naive_average_probs >= dd_prob_splits[i][0])
                & (naive_average_probs < dd_prob_splits[i][1])
            ]
        ) / len(naive_average_probs)
    for i in range(4):
        axes[1].bar(
            0,
            naive_prob_splits[i],
            bottom=np.sum(naive_prob_splits[0:i]),
            color=dd_colors[i],
        )
        axes[1].bar(
            1, prob_splits[i], bottom=np.sum(prob_splits[0:i]), color=dd_colors[i]
        )
        axes[1].bar(
            2, bi_prob_splits[i], bottom=np.sum(bi_prob_splits[0:i]), color=dd_colors[i]
        )

    axes[0].set_ylim([0.0, 1])
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["Naive", "Uni", "Bi"])
    axes[0].set_ylabel("Percentage DD")

    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["Naive", "Uni", "Bi"])
    axes[1].set_ylabel("DD Confidence")

    makeNice(axes)
    add_fig_labels(axes)
    fig.savefig("../data/unilateral/summary.pdf", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(4, 3, figsize=(8, 6), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]

    # Label each DataFrame
    df1_labeled = naive_df.copy()
    df1_labeled["Dataset"] = "Naive"

    df2_labeled = uni_df.copy()
    df2_labeled["Dataset"] = "Uni"

    df3_labeled = bi_df.copy()
    df3_labeled["Dataset"] = "Bi"

    cols_to_plot = bi_df.columns.to_list()

    # Combine and melt
    df_combined = pd.concat([df1_labeled, df2_labeled, df3_labeled], ignore_index=True)
    df_melted = pd.melt(
        df_combined,
        id_vars="Dataset",
        value_vars=cols_to_plot,
        var_name="Feature",
        value_name="Value",
    )

    print(df_combined.columns)

    for i in range(len(axes)):
        sns.boxplot(
            data=df_melted[df_melted["Feature"] == cols_to_plot[i]],
            x="Dataset",
            y="Value",
            ax=axes[i],
            showfliers=False,
        )
        axes[i].set_xlabel("")
        axes[i].set_ylabel(f"{cols_to_plot[i]}")
    makeNice(axes)
    add_fig_labels(axes)
    fig.savefig("../data/unilateral/summary_features.pdf", bbox_inches="tight")
    plt.close()

    uni_df["DD Confidence"] = average_probs

    uni_df.to_csv("../data/unilateral/unilateral_df.csv")
    run_cmd(f"open ../data/unilateral/summary.pdf")
    run_cmd(f"open ../data/unilateral/summary_features.pdf")


save_path = "/Users/johnparker/paper_repos/Aristieta_Parker_Rubin_Gittis_2024_motor_rescue/data/unilateral"
raw_path = "/Users/johnparker/paper_repos/Aristieta_Parker_Rubin_Gittis_2024_motor_rescue/data/unilateral/Uni_DD_SNr_recordings"
spike_path = "/Users/johnparker/paper_repos/Aristieta_Parker_Rubin_Gittis_2024_motor_rescue/data/unilateral/spike_trains"
spikes = gather_unilateral_data(raw_path)
print(len(spikes))
sys.exit()


# process_data(spikes, 30, spike_path)

# get_unilateral_data()
