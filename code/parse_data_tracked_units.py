"""
parse_data_tracked_units.py

This script gathers and grabs statistics of trakced units from the ephys data. 
It will save the data in a format that can be used for further analysis.

Author: John E. Parker (2024)
"""

# python modules
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os, sys
import seaborn as sns
import json
from scipy import stats
import matplotlib.patches as patches
import pickle

# set up plotting parameters
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.labelsize"] = 8


# user modules
from helpers import *
import parse_data
import clean_data

# global variables
save_dir = "../data/tracked_units"
jaws_cell_save_dir = f"{save_dir}/jaws_neurons/cell_data"
jaws_features_save_dir = f"{save_dir}/jaws_neurons/features"
npas_cell_save_dir = f"{save_dir}/npas_neurons/cell_data"
npas_features_save_dir = f"{save_dir}/npas_neurons/features"
jaws_feature_plots = f"{save_dir}/jaws_neurons/plots"
npas_feature_plots = f"{save_dir}/npas_neurons/plots"
jaws_path = "/Users/johnparker/UPitt_Data/SNr_motor_rescue_project/jaws_pre_processed"
npas_path = "/Users/johnparker/UPitt_Data/SNr_motor_rescue_project/npas_pre_processed"
jaws_cells = 23
npas_cells = 10
seeds = 15
neural_net = "../data/neural_net"


def get_ephys_tracked_data(source_path, save_path, length):
    """
    This function will gather the ephys data from the tracked units and save it in a format that can be used for further analysis.

    Parameters
    ----------
    \t source_path (str) : path to the source data

    \t save_path (str) : path to save the data

    \t length (int) : length of the time window to analyze

    Returns
    -------
    \t all_spikes (list) : list of spikes for each cell

    \t light_on_off (list) : list of light on and off times for each cell

    \t is_medial (list) : list of whether the cell is medial or not

    \t mouse (list) : list of mouse identifiers

    \t folder (list) : list of folders

    \t cell_name (list) : list of cell names
    """

    # Create path to save data
    run_cmd(f"mkdir -p {save_path}")

    # Initialize lists to store data
    all_spikes = []
    light_on_off = []
    is_medial = []
    mouse = []
    cell_name = []
    folder = []

    # Iterate through files and gather data
    for cell in sorted(os.listdir(f"{source_path}/pre_post_opto")):
        if cell != ".DS_Store":
            light_on = np.loadtxt(f"{source_path}/pre_post_opto/{cell}/light_on.txt")
            spikes = np.loadtxt(f"{source_path}/pre_post_opto/{cell}/spikes.txt")
            meta_data = json.load(
                open(f"{source_path}/pre_post_opto/{cell}/meta_data.txt", "r")
            )
            is_medial.append(meta_data["medial"])
            mouse.append(meta_data["mouse"])
            cell_name.append(meta_data["cell_name"])
            folder.append(meta_data["folder"])
            shift = light_on[0, 0] - length
            light_on -= shift
            spikes -= shift

            # Collect spikes
            collect_spikes = []
            for i in range(6):
                collect_spikes.append(
                    spikes[
                        (light_on[0, 0] - (5 - i) * 30 > spikes)
                        & (spikes >= light_on[0, 0] - (6 - i) * length)
                    ]
                )

            # Collect trials in spikes
            for k in range(light_on.shape[0] - 1):
                collect_spikes.append(
                    spikes[(spikes >= light_on[k, 0]) & (spikes < light_on[k, 1])]
                )
                for j in range(
                    int(np.ceil(light_on[k + 1, 0] - light_on[k, 1]) / length)
                ):
                    collect_spikes.append(
                        spikes[
                            (spikes >= light_on[k, 1] + length * j)
                            & (spikes < light_on[k, 1] + length * (j + 1))
                        ]
                    )
            collect_spikes.append(
                spikes[(spikes >= light_on[-1, 0]) & (spikes < light_on[-1, 1])]
            )
            for j in range(int(np.ceil(light_on[-1, 0] - light_on[-2, 1]) / length)):
                collect_spikes.append(
                    spikes[
                        (spikes >= light_on[-1, 1] + length * j)
                        & (spikes < light_on[-1, 1] + length * (j + 1))
                    ]
                )

            all_spikes.append(collect_spikes)
            light_on_off.append(light_on)

    return all_spikes, light_on_off, is_medial, mouse, folder, cell_name


def save_spikes_light_ephys_tracked_data(spikes, light_on, save_path):
    """
    Saves tracked unit data in a format that can be used for further analysis.

    Parameters
    ----------
    \t spikes (list) : list of spikes for each cell

    \t light_on (list) : list of light on and off times for each cell

    \t save_path (str) : path to save the data
    """

    # Create path to save data
    cell_count = 1
    for cell_spikes in spikes:
        save_dir = f"{save_path}/cell_{cell_count:04d}"
        run_cmd(f"mkdir -p {save_dir}", print_out=False)
        run_cmd(f"mkdir -p {save_dir}/spikes", print_out=False)
        np.savetxt(
            f"{save_dir}/light_on.txt",
            light_on[cell_count - 1],
            newline="\n",
            delimiter="\t",
            fmt="%f",
        )
        segment = 0

        # save individual stim segments
        for stim_segment in cell_spikes:
            np.savetxt(
                f"{save_dir}/spikes/segment_{segment:04d}.txt",
                stim_segment,
                newline="\n",
                delimiter="\t",
                fmt="%f",
            )
            segment += 1
        cell_count += 1


def calc_firing_rates_tracked(spikes, length):
    """
    Calculate firing rates for tracked units.

    Parameters
    ----------
    \t spikes (list) : list of spikes for each cell

    \t length (int) : length of the time window to analyze

    Returns
    -------
    \t rates (np.ndarray) : firing rates for each cell
    """
    rates = []
    for cell_spikes in spikes:
        cell_rates = parse_data.calc_firing_rates(
            cell_spikes, np.ones(len(cell_spikes)) * length
        )
        rates.append(cell_rates)
    return np.asarray(rates)


def calc_cv_tracked(spikes):
    """
    Calculate CV from tracked cells

    Parameters
    ----------
    \t spikes (list) : list of spikes for each cell

    Returns
    -------
    \t cv (np.ndarray) : CV for each cell
    """
    cv = []
    for cell_spikes in spikes:
        cell_cvs = parse_data.calc_cv(cell_spikes)
        cv.append(cell_cvs)
    return np.asarray(cv)


def calc_burst_statistics_tracked(
    spikes, rates, length, min_spikes=3, surprise_threshold=3, window=0.25
):
    """
    Calculate burst statistics for tracked units.

    Parameters
    ----------
    \t spikes (list) : list of spikes for each cell

    \t rates (np.ndarray) : firing rates for each cell

    \t length (int) : length of the time window to analyze

    \t min_spikes=3 (int) : minimum number of spikes to be considered a burst

    \t surprise_threshold=3 (int) : threshold for surprise

    \t window=0.25 (float) : window for surprise

    Returns
    -------
    \t burst_stats (np.ndarray) : burst statistics for each cell
    """
    burst_stats = []
    count = 0
    for cell_spikes in spikes:
        burst_stats.append(
            parse_data.calc_burst_statistics(
                cell_spikes,
                rates[count],
                np.ones(len(cell_spikes)) * length,
                min_spikes=min_spikes,
                surprise_threshold=surprise_threshold,
                window=window,
            ),
        )
        count += 1
    return np.asarray(burst_stats)


def save_rates_ephys_tracked_data(rates, save_path):
    """
    Save firing rates for tracked data.

    Parameters
    ----------
    \t rates (np.ndarray) : firing rates for each cell

    \t save_path (str) : path to save the data
    """
    cell_count = 0
    for rate in rates:
        np.savetxt(
            f"{save_path}/cell_{cell_count+1:04d}/rates.txt",
            rate,
            fmt="%f",
            newline="\n",
            delimiter="\t",
        )
        cell_count += 1


def plot_tracked_healthy_dd(jaws_path, npas_path, jaws_units=23, npas_units=10):
    """
    Plot tracked units for healthy and DD cells.

    Parameters
    ----------
    \t jaws_path (str) : path to JAWS data

    \t npas_path (str) : path to NPAS data

    \t jaws_units=23 (int) : number of JAWS units

    \t npas_units=10 (int) : number of NPAS units
    """

    # Gather jaws data
    jaws_data = []
    for i in range(jaws_units):
        jaws_data.append(pd.read_csv(f"{jaws_path}/jaws_treatment_cell_{i+1}.csv"))

    # Set up jaws data for all spike trains
    jaws_dd = np.zeros(76)
    jaws_dd_prob = np.zeros(76)
    for i in range(len(jaws_dd)):
        for data in jaws_data:
            jaws_dd_prob[i] += (
                data.iloc[i, data.columns.get_loc("DD Probability")] / jaws_units
            )
            if data.iloc[i, data.columns.get_loc("DD Probability")] > 0.5:
                jaws_dd[i] += 1 / jaws_units

    # Gather npas data
    npas_data = []
    for i in range(npas_units):
        npas_data.append(pd.read_csv(f"{npas_path}/npas_treatment_cell_{i+1}.csv"))

    # Set up npas data for all spike trains
    npas_dd = np.zeros(76)
    npas_dd_prob = np.zeros(76)
    for i in range(len(npas_dd)):
        for data in npas_data:
            npas_dd_prob[i] += (
                data.iloc[i, data.columns.get_loc("DD Probability")] / npas_units
            )
            if data.iloc[i, data.columns.get_loc("DD Probability")] > 0.5:
                npas_dd[i] += 1 / npas_units

    # Plot DD confidence over time
    fig, ax = plt.subplots(2, 1, figsize=(8, 4), dpi=300, tight_layout=True)
    axes = [ax[i] for i in range(2)]
    axes[0].plot(np.arange(76), jaws_dd, color="blue", label="JAWS DD % (23 Units)")
    axes[0].plot(np.arange(76), npas_dd, color="red", label="Npas DD % (10 Units)")
    axes[0].vlines(np.arange(6, 76, 7), 0, 1, color="k", linestyle="dashed")
    axes[0].legend()
    axes[1].plot(
        np.arange(76), jaws_dd_prob, color="blue", label="JAWS Avg DD Probability"
    )
    axes[1].plot(
        np.arange(76), npas_dd_prob, color="red", label="Npas Avg DD Probability"
    )
    axes[1].vlines(np.arange(6, 76, 7), 0, 1, color="k", linestyle="dashed")
    axes[1].legend()
    axes[0].set_xticks(np.arange(0, 76))
    axes[0].set_xticklabels(np.arange(-180, 70 * 30, 30), rotation=90)
    axes[1].set_xticks(np.arange(0, 76))
    axes[1].set_xticklabels(np.arange(-180, 70 * 30, 30), rotation=90)
    axes[0].set_ylabel("Percent DD")
    axes[1].set_ylabel("Average DD Probability")
    makeNice(axes, labelsize=6)
    fig.savefig(f"../data/tracked_units/dd_probabilities.pdf", bbox_inches="tight")
    plt.close()
    run_cmd("open data/tracked_units/dd_probabilities.pdf")


def create_feature_results():
    """
    Create all feature results and save to file.
    """

    # Create save directories
    run_cmd(f"mkdir -p {jaws_feature_plots}")
    run_cmd(f"mkdir -p {npas_feature_plots}")

    # Get JAWS data
    jaws_feature_tables = [
        pd.read_csv(
            f"{jaws_features_save_dir}/jaws_treatment_cell_{i+1:04d}.csv",
            index_col="Unnamed: 0",
        )
        for i in range(jaws_cells)
    ]

    # replace NaN values with 0
    [jaws_feature_tables[i].fillna(0, inplace=True) for i in range(jaws_cells)]

    # Get NPAS data
    npas_feature_tables = [
        pd.read_csv(
            f"{npas_features_save_dir}/npas_treatment_cell_{i+1:04d}.csv",
            index_col="Unnamed: 0",
        )
        for i in range(npas_cells)
    ]

    # replace NaN values with 0
    [npas_feature_tables[i].fillna(0, inplace=True) for i in range(npas_cells)]

    # Set up plotting times
    times = np.arange(-180, 30 * len(jaws_feature_tables[0]) - 180, 30)
    stim_times = np.arange(0, (10 * 210), 210)
    pre_times = [f"{x}" for x in np.arange(-30, -30 + (10 * 210), 210)]
    post_times = [f"{x}" for x in np.arange(30, 30 + (10 * 210), 210)]

    # Initialize arrays
    jaws_prob_dd = np.zeros((len(jaws_feature_tables[0]), jaws_cells))
    jaws_prob_naive = np.zeros((len(jaws_feature_tables[0]), jaws_cells))

    # Load MLPs and determine DD probabilities for each segment
    for jc in range(jaws_cells):
        for i in range(seeds):

            # Load training data for noramlization of jaws data
            X_train = pd.read_csv(f"{neural_net}/X_train_seed_{i:02d}.csv").iloc[:, 1:]
            X_train_norm, jaws_norm = clean_data.normalize_data(
                X_train,
                jaws_feature_tables[jc].iloc[:, 0:12],
                min_max=False,
            )

            # Load neural network and determine DD probability
            with open(f"{neural_net}/MLP_seed_{i:02d}.pkl", "rb") as mlp_file:
                mlp_clf = pickle.load(mlp_file)
                probs = mlp_clf.predict_proba(jaws_norm) / seeds
                jaws_prob_dd[:, jc] += probs[:, 0]
                jaws_prob_naive[:, jc] += probs[:, 1]

    # Initialize arrays
    npas_prob_dd = np.zeros((len(npas_feature_tables[0]), npas_cells))
    npas_prob_naive = np.zeros((len(npas_feature_tables[0]), npas_cells))

    # Load MLPs and determine DD probabilities for each segment
    for jc in range(npas_cells):
        for i in range(seeds):

            # Load training data for noramlization of npas data
            X_train = pd.read_csv(f"{neural_net}/X_train_seed_{i:02d}.csv").iloc[:, 1:]
            X_train_norm, npas_norm = clean_data.normalize_data(
                X_train,
                npas_feature_tables[jc].iloc[:, 0:12],
                min_max=False,
            )

            # Load neural network and determine DD probability
            with open(f"{neural_net}/MLP_seed_{i:02d}.pkl", "rb") as mlp_file:
                mlp_clf = pickle.load(mlp_file)
                probs = mlp_clf.predict_proba(npas_norm) / seeds
                npas_prob_dd[:, jc] += probs[:, 0]
                npas_prob_naive[:, jc] += probs[:, 1]

    # Generate average probabilities for each segment
    jaws_prob_dd_pre = np.mean(
        jaws_prob_dd[jaws_feature_tables[0]["Treatment Times"].isin(pre_times)], axis=0
    )
    jaws_prob_dd_stim = np.mean(
        jaws_prob_dd[
            jaws_feature_tables[0]["Treatment Times"].str.contains("Treatment")
        ],
        axis=0,
    )
    jaws_prob_dd_post = np.mean(
        jaws_prob_dd[jaws_feature_tables[0]["Treatment Times"].isin(post_times)], axis=0
    )

    npas_prob_dd_pre = np.mean(
        npas_prob_dd[npas_feature_tables[0]["Treatment Times"].isin(pre_times)], axis=0
    )
    npas_prob_dd_stim = np.mean(
        npas_prob_dd[
            npas_feature_tables[0]["Treatment Times"].str.contains("Treatment")
        ],
        axis=0,
    )
    npas_prob_dd_post = np.mean(
        npas_prob_dd[npas_feature_tables[0]["Treatment Times"].isin(post_times)], axis=0
    )

    jaws_prob_naive_pre = np.mean(
        jaws_prob_naive[jaws_feature_tables[0]["Treatment Times"].isin(pre_times)],
        axis=0,
    )
    jaws_prob_naive_stim = np.mean(
        jaws_prob_naive[
            jaws_feature_tables[0]["Treatment Times"].str.contains("Treatment")
        ],
        axis=0,
    )
    jaws_prob_naive_post = np.mean(
        jaws_prob_naive[jaws_feature_tables[0]["Treatment Times"].isin(post_times)],
        axis=0,
    )

    npas_prob_naive_pre = np.mean(
        npas_prob_naive[npas_feature_tables[0]["Treatment Times"].isin(pre_times)],
        axis=0,
    )
    npas_prob_naive_stim = np.mean(
        npas_prob_naive[
            npas_feature_tables[0]["Treatment Times"].str.contains("Treatment")
        ],
        axis=0,
    )
    npas_prob_naive_post = np.mean(
        npas_prob_naive[npas_feature_tables[0]["Treatment Times"].isin(post_times)],
        axis=0,
    )

    ### Saving JAWS probabilities
    np.savetxt(
        f"{jaws_features_save_dir}/jaws_prob_dd.csv",
        jaws_prob_dd,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{jaws_features_save_dir}/jaws_prob_naive.csv",
        jaws_prob_naive,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{jaws_features_save_dir}/jaws_prob_dd_pre.csv",
        jaws_prob_dd_pre,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{jaws_features_save_dir}/jaws_prob_dd_stim.csv",
        jaws_prob_dd_stim,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{jaws_features_save_dir}/jaws_prob_dd_post.csv",
        jaws_prob_dd_post,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{jaws_features_save_dir}/jaws_prob_naive_pre.csv",
        jaws_prob_naive_pre,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{jaws_features_save_dir}/jaws_prob_naive_stim.csv",
        jaws_prob_naive_stim,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{jaws_features_save_dir}/jaws_prob_naive_post.csv",
        jaws_prob_naive_post,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    ### Saving NPAS probabilities
    np.savetxt(
        f"{npas_features_save_dir}/npas_prob_dd.csv",
        npas_prob_dd,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{npas_features_save_dir}/npas_prob_naive.csv",
        npas_prob_naive,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{npas_features_save_dir}/npas_prob_dd_pre.csv",
        npas_prob_dd_pre,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{npas_features_save_dir}/npas_prob_dd_stim.csv",
        npas_prob_dd_stim,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{npas_features_save_dir}/npas_prob_dd_post.csv",
        npas_prob_dd_post,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{npas_features_save_dir}/npas_prob_naive_pre.csv",
        npas_prob_naive_pre,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{npas_features_save_dir}/npas_prob_naive_stim.csv",
        npas_prob_naive_stim,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    np.savetxt(
        f"{npas_features_save_dir}/npas_prob_naive_post.csv",
        npas_prob_naive_post,
        fmt="%f",
        delimiter=",",
        newline="\n",
    )

    ### Plotting DD Confidence over time
    fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=300, tight_layout=True)
    ax.plot(times, np.mean(jaws_prob_dd, axis=1), lw=0.5, label="JAWS")
    ax.plot(times, np.mean(npas_prob_dd, axis=1), lw=0.5, label="NPAS")
    ylims = ax.get_ylim()
    ax.vlines(stim_times, ylims[0], ylims[1], color="k", lw=0.25, ls="dashed")
    ax.set_ylim(ylims)
    ax.set_xticks(stim_times)
    ax.set_xticklabels(stim_times)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("DD Confidence")
    ax.legend(fancybox=False, frameon=False)
    makeNice(ax, labelsize=6)
    fig.savefig(f"{save_dir}/dd_confidence_over_time.pdf", bbox_inches="tight")
    plt.close()

    # Save confidence over time
    pd.DataFrame(
        data=[
            np.mean(npas_prob_dd, axis=1),
            np.mean(jaws_prob_dd, axis=1),
            stats.sem(npas_prob_dd, axis=1),
            stats.sem(jaws_prob_dd, axis=1),
            np.std(npas_prob_dd, axis=1),
            np.std(jaws_prob_dd, axis=1),
        ],
        columns=jaws_feature_tables[0]["Treatment Times"],
        index=["NPAS Avg", "JAWS Avg", "NPAS SEM", "JAWS SEM", "NPAS STD", "JAWS STD"],
    ).to_csv(f"{save_dir}/dd_confidence_over_time.csv")

    ### JAWS Feature Pre Post
    jaws_pre_avg_treatment = np.zeros((jaws_cells, 12))  # 12 features
    jaws_avg_treatment = np.zeros((jaws_cells, 12))  # 12 features
    jaws_post_treatment = np.zeros((jaws_cells, 12))  # 12 features

    # Calculate average features for each cell
    for jc in range(jaws_cells):
        df = jaws_feature_tables[jc]
        df_pre = df[df["Treatment Times"].isin(pre_times)]
        df_post = df[df["Treatment Times"].isin(post_times)]
        df_stim = df[df["Treatment Times"].str.contains("Treatment")]
        for col in range(12):
            jaws_pre_avg_treatment[jc, col] = np.mean(df_pre[df_pre.columns[col]])
            jaws_avg_treatment[jc, col] = np.mean(df_stim[df_stim.columns[col]])
            jaws_post_treatment[jc, col] = np.mean(df_post[df_post.columns[col]])

    # Save JAWS features
    pd.DataFrame(
        jaws_pre_avg_treatment, columns=jaws_feature_tables[0].columns[0:12]
    ).to_csv(f"{jaws_features_save_dir}/pre_treatment_average.csv")
    pd.DataFrame(
        jaws_avg_treatment, columns=jaws_feature_tables[0].columns[0:12]
    ).to_csv(f"{jaws_features_save_dir}/treatment_average.csv")
    pd.DataFrame(
        jaws_post_treatment, columns=jaws_feature_tables[0].columns[0:12]
    ).to_csv(f"{jaws_features_save_dir}/post_treatment_average.csv")

    # Initialize arrays
    output_pre_stim_stats = []
    output_pre_post_stats = []
    output_stim_post_stats = []
    output_pre_data = []
    output_stim_data = []
    output_post_data = []
    output_labels = []

    # Plot pre stim and post results for each feature
    for col in range(12):
        fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=300, tight_layout=True)
        output_pre_data.extend(
            [
                np.mean(jaws_pre_avg_treatment[:, col]),
                stats.sem(jaws_pre_avg_treatment[:, col]),
                np.std(jaws_pre_avg_treatment[:, col]),
            ]
        )
        output_stim_data.extend(
            [
                np.mean(jaws_avg_treatment[:, col]),
                stats.sem(jaws_avg_treatment[:, col]),
                np.std(jaws_avg_treatment[:, col]),
            ]
        )
        output_post_data.extend(
            [
                np.mean(jaws_post_treatment[:, col]),
                stats.sem(jaws_post_treatment[:, col]),
                np.std(jaws_post_treatment[:, col]),
            ]
        )
        output_labels.extend(
            [
                f"{jaws_feature_tables[0].columns[col]} Avg",
                f"{jaws_feature_tables[0].columns[col]} SEM",
                f"{jaws_feature_tables[0].columns[col]} STD",
            ]
        )

        output_pre_post_stats.append(
            stats.ttest_rel(
                jaws_pre_avg_treatment[:, col],
                jaws_post_treatment[:, col],
            ).pvalue
        )

        output_pre_stim_stats.append(
            stats.ttest_rel(
                jaws_pre_avg_treatment[:, col],
                jaws_avg_treatment[:, col],
            ).pvalue
        )

        output_stim_post_stats.append(
            stats.ttest_rel(
                jaws_post_treatment[:, col],
                jaws_avg_treatment[:, col],
            ).pvalue
        )

        ax.errorbar(
            x=[0, 1, 2],
            y=[
                np.mean(jaws_pre_avg_treatment[:, col]),
                np.mean(jaws_avg_treatment[:, col]),
                np.mean(jaws_post_treatment[:, col]),
            ],
            yerr=[
                stats.sem(jaws_pre_avg_treatment[:, col]),
                stats.sem(jaws_avg_treatment[:, col]),
                stats.sem(jaws_post_treatment[:, col]),
            ],
            marker="o",
            color="k",
            capsize=5,
        )
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Pre", "Stim", "Post"])
        ax.set_ylabel(jaws_feature_tables[0].columns[col])
        makeNice(ax)
        fig.savefig(
            f"{jaws_feature_plots}/{jaws_feature_tables[0].columns[col]}.pdf",
            bbox_inches="tight",
        )
        plt.close()

    # Determine DD probability statistics
    output_pre_data.extend(
        [
            np.mean(jaws_prob_dd_pre),
            stats.sem(jaws_prob_dd_pre),
            np.std(jaws_prob_dd_pre),
        ]
    )
    output_stim_data.extend(
        [
            np.mean(jaws_prob_dd_stim),
            stats.sem(jaws_prob_dd_stim),
            np.std(jaws_prob_dd_stim),
        ]
    )
    output_post_data.extend(
        [
            np.mean(jaws_prob_dd_post),
            stats.sem(jaws_prob_dd_post),
            np.std(jaws_prob_dd_post),
        ]
    )
    output_labels.extend(
        [
            f"DD Prob Avg",
            f"DD Prob SEM",
            f"DD Prob STD",
        ]
    )

    # Run statistics on DD probabilities
    output_pre_post_stats.append(
        stats.ttest_rel(
            jaws_prob_dd_pre,
            jaws_prob_dd_post,
        ).pvalue
    )

    output_pre_stim_stats.append(
        stats.ttest_rel(
            jaws_prob_dd_pre,
            jaws_prob_dd_stim,
        ).pvalue
    )

    output_stim_post_stats.append(
        stats.ttest_rel(
            jaws_prob_dd_stim,
            jaws_prob_dd_post,
        ).pvalue
    )

    cols = list(jaws_feature_tables[0].columns[0:12])
    cols.append("DD Prob")

    # Save summary statistics
    pd.DataFrame(
        data=[output_pre_stim_stats, output_pre_post_stats, output_stim_post_stats],
        columns=cols,
        index=[
            "Paired T-test p-value Pre vs Stim",
            "Paired T-test p-value Pre vs Post",
            "Paired T-test p-value Stim vs Post",
        ],
    ).to_csv(f"{jaws_features_save_dir}/jaws_feature_stats_summary.csv")
    pd.DataFrame(
        data=[output_pre_data, output_stim_data, output_post_data],
        columns=output_labels,
        index=["Pre", "Stim", "Post"],
    ).to_csv(f"{jaws_features_save_dir}/jaws_feature_summary.csv")

    ### NPAS Feature Pre Post
    npas_pre_avg_treatment = np.zeros((npas_cells, 12))  # 12 features
    npas_avg_treatment = np.zeros((npas_cells, 12))  # 12 features
    npas_post_treatment = np.zeros((npas_cells, 12))  # 12 features

    # Calculate average features for each cell
    for jc in range(npas_cells):
        df = npas_feature_tables[jc]
        df_pre = df[df["Treatment Times"].isin(pre_times)]
        df_post = df[df["Treatment Times"].isin(post_times)]
        df_stim = df[df["Treatment Times"].str.contains("Treatment")]
        for col in range(12):
            npas_pre_avg_treatment[jc, col] = np.mean(df_pre[df_pre.columns[col]])
            npas_avg_treatment[jc, col] = np.mean(df_stim[df_stim.columns[col]])
            npas_post_treatment[jc, col] = np.mean(df_post[df_post.columns[col]])

    # Save NPAS features
    pd.DataFrame(
        npas_pre_avg_treatment, columns=npas_feature_tables[0].columns[0:12]
    ).to_csv(f"{npas_features_save_dir}/pre_treatment_average.csv")
    pd.DataFrame(
        npas_avg_treatment, columns=npas_feature_tables[0].columns[0:12]
    ).to_csv(f"{npas_features_save_dir}/treatment_average.csv")
    pd.DataFrame(
        npas_post_treatment, columns=npas_feature_tables[0].columns[0:12]
    ).to_csv(f"{npas_features_save_dir}/post_treatment_average.csv")

    # Initialize arrays
    output_pre_stim_stats = []
    output_pre_post_stats = []
    output_stim_post_stats = []
    output_pre_data = []
    output_stim_data = []
    output_post_data = []
    output_labels = []

    # Plot pre stim and post results for each feature
    for col in range(12):
        fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=300, tight_layout=True)
        output_pre_data.extend(
            [
                np.mean(npas_pre_avg_treatment[:, col]),
                stats.sem(npas_pre_avg_treatment[:, col]),
                np.std(npas_pre_avg_treatment[:, col]),
            ]
        )
        output_stim_data.extend(
            [
                np.mean(npas_avg_treatment[:, col]),
                stats.sem(npas_avg_treatment[:, col]),
                np.std(npas_avg_treatment[:, col]),
            ]
        )
        output_post_data.extend(
            [
                np.mean(npas_post_treatment[:, col]),
                stats.sem(npas_post_treatment[:, col]),
                np.std(npas_post_treatment[:, col]),
            ]
        )
        output_labels.extend(
            [
                f"{npas_feature_tables[0].columns[col]} Avg",
                f"{npas_feature_tables[0].columns[col]} SEM",
                f"{npas_feature_tables[0].columns[col]} STD",
            ]
        )

        output_pre_post_stats.append(
            stats.ttest_rel(
                npas_pre_avg_treatment[:, col],
                npas_post_treatment[:, col],
            ).pvalue
        )

        output_pre_stim_stats.append(
            stats.ttest_rel(
                npas_pre_avg_treatment[:, col],
                npas_avg_treatment[:, col],
            ).pvalue
        )

        output_stim_post_stats.append(
            stats.ttest_rel(
                npas_post_treatment[:, col],
                npas_avg_treatment[:, col],
            ).pvalue
        )
        ax.errorbar(
            x=[0, 1, 2],
            y=[
                np.mean(npas_pre_avg_treatment[:, col]),
                np.mean(npas_avg_treatment[:, col]),
                np.mean(npas_post_treatment[:, col]),
            ],
            yerr=[
                stats.sem(npas_pre_avg_treatment[:, col]),
                stats.sem(npas_avg_treatment[:, col]),
                stats.sem(npas_post_treatment[:, col]),
            ],
            marker="o",
            color="k",
            capsize=5,
        )
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Pre", "Stim", "Post"])
        ax.set_ylabel(npas_feature_tables[0].columns[col])
        makeNice(ax)
        fig.savefig(
            f"{npas_feature_plots}/{npas_feature_tables[0].columns[col]}.pdf",
            bbox_inches="tight",
        )
        plt.close()

    # Determine DD probability statistics
    output_pre_data.extend(
        [
            np.mean(npas_prob_dd_pre),
            stats.sem(npas_prob_dd_pre),
            np.std(npas_prob_dd_pre),
        ]
    )
    output_stim_data.extend(
        [
            np.mean(npas_prob_dd_stim),
            stats.sem(npas_prob_dd_stim),
            np.std(npas_prob_dd_stim),
        ]
    )
    output_post_data.extend(
        [
            np.mean(npas_prob_dd_post),
            stats.sem(npas_prob_dd_post),
            np.std(npas_prob_dd_post),
        ]
    )
    output_labels.extend(
        [
            f"DD Prob Avg",
            f"DD Prob SEM",
            f"DD Prob STD",
        ]
    )

    # Run statistics on DD probabilities
    output_pre_post_stats.append(
        stats.ttest_rel(
            npas_prob_dd_pre,
            npas_prob_dd_post,
        ).pvalue
    )

    output_pre_stim_stats.append(
        stats.ttest_rel(
            npas_prob_dd_pre,
            npas_prob_dd_stim,
        ).pvalue
    )

    output_stim_post_stats.append(
        stats.ttest_rel(
            npas_prob_dd_stim,
            npas_prob_dd_post,
        ).pvalue
    )

    cols = list(npas_feature_tables[0].columns[0:12])
    cols.append("DD Prob")

    # Save summary statistics
    pd.DataFrame(
        data=[output_pre_stim_stats, output_pre_post_stats, output_stim_post_stats],
        columns=cols,
        index=[
            "Paired T-test p-value Pre vs Stim",
            "Paired T-test p-value Pre vs Post",
            "Paired T-test p-value Stim vs Post",
        ],
    ).to_csv(f"{npas_features_save_dir}/npas_feature_stats_summary.csv")
    pd.DataFrame(
        data=[output_pre_data, output_stim_data, output_post_data],
        columns=output_labels,
        index=["Pre", "Stim", "Post"],
    ).to_csv(f"{npas_features_save_dir}/npas_feature_summary.csv")


def create_ephys_tracked_units(fixed_length, renewal_power=False):
    """
    Create and plot tracked units for ephys data.

    Parameters
    ----------
    \t fixed_length (int) : fixed length of the time window

    \t renewal_power=False (bool) : whether to use renewal power
    """

    # Create save directories and gather data
    jaws_spikes, jaws_light, jaws_is_medial, jaws_mouse, jaws_folder, jaws_cell_name = (
        get_ephys_tracked_data(jaws_path, f"{save_dir}/jaws_neurons", fixed_length)
    )

    npas_spikes, npas_light, npas_is_medial, npas_mouse, npas_folder, npas_cell_name = (
        get_ephys_tracked_data(npas_path, f"{save_dir}/npas_neurons", fixed_length)
    )

    # Save data
    save_spikes_light_ephys_tracked_data(jaws_spikes, jaws_light, jaws_cell_save_dir)
    save_spikes_light_ephys_tracked_data(npas_spikes, npas_light, npas_cell_save_dir)

    # Calculate firing rates, CVs, and burst statistics
    jaws_rates = calc_firing_rates_tracked(jaws_spikes, fixed_length)
    npas_rates = calc_firing_rates_tracked(npas_spikes, fixed_length)

    x_values = np.arange(jaws_rates.shape[1])
    treatment_times = np.arange(6, 76, 7)
    x_values = [i for i in x_values if i not in treatment_times]

    save_rates_ephys_tracked_data(jaws_rates, jaws_cell_save_dir)
    save_rates_ephys_tracked_data(npas_rates, npas_cell_save_dir)

    jaws_cvs = calc_cv_tracked(jaws_spikes)
    jaws_burst_stats = calc_burst_statistics_tracked(
        jaws_spikes,
        jaws_rates,
        fixed_length,
        min_spikes=3,
        surprise_threshold=3,
        window=2,
    )

    npas_cvs = calc_cv_tracked(npas_spikes)
    npas_burst_stats = calc_burst_statistics_tracked(
        npas_spikes,
        npas_rates,
        fixed_length,
        min_spikes=3,
        surprise_threshold=3,
        window=2,
    )

    # Run oscillation code
    run_cmd(
        '/Applications/MATLAB_R2021b.app/bin/matlab -nojvm -nodesktop -batch "get_osc_data_ephys_tracked"'
    )

    # Create save directories
    run_cmd(f"mkdir -p {jaws_features_save_dir}")
    run_cmd(f"mkdir -p {npas_features_save_dir}")

    # Gather data and reformat for saving
    all_jaws_data = np.zeros(
        (jaws_rates.shape[1], 12, jaws_rates.shape[0])
    )  # intervals, features, cells
    print(all_jaws_data.shape)
    for i in range(all_jaws_data.shape[2]):
        all_jaws_data[:, 0, i] = jaws_rates[i, :]
        all_jaws_data[:, 1, i] = jaws_cvs[i, :]
        osc = np.loadtxt(
            f"{jaws_cell_save_dir}/cell_{i+1:04d}/osc_data.txt",
            delimiter=",",
        )
        all_jaws_data[:, 2, i] = osc[:, 1 if renewal_power else 2]
        all_jaws_data[:, 3, i] = osc[:, 5 if renewal_power else 6]
        count = 4
        for stat in [0, 1, 2, 3, 4, 5, 8, 9]:
            scale = 30 if stat == 0 else 1
            all_jaws_data[:, count, i] = jaws_burst_stats[i, :, stat] / scale
            count += 1

    all_npas_data = np.zeros((npas_rates.shape[1], 12, npas_rates.shape[0]))
    print(all_npas_data.shape)
    for i in range(all_npas_data.shape[2]):
        all_npas_data[:, 0, i] = npas_rates[i, :]
        all_npas_data[:, 1, i] = npas_cvs[i, :]
        osc = np.loadtxt(
            f"{npas_cell_save_dir}/cell_{i+1:04d}/osc_data.txt",
            delimiter=",",
        )
        all_npas_data[:, 2, i] = osc[:, 1 if renewal_power else 2]
        all_npas_data[:, 3, i] = osc[:, 5 if renewal_power else 6]
        count = 4
        for stat in [0, 1, 2, 3, 4, 5, 8, 9]:
            scale = 30 if stat == 0 else 1
            all_npas_data[:, count, i] = npas_burst_stats[i, :, stat] / scale
            count += 1

    # Set up treatment index for saving
    treat = 1
    treatment_index = []
    for i in range(jaws_rates.shape[1]):
        if i not in treatment_times:
            treatment_index.append(i * 30 - 180)
        else:
            treatment_index.append(f"Treatment {treat}")
            treat += 1
    treatment_index = np.asarray(treatment_index)

    # Feature labels
    col_labels = [
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

    # Save data
    for k in range(all_npas_data.shape[2]):
        npas_export_df = pd.DataFrame(
            data=all_npas_data[:, :, k],
            columns=col_labels,
        )
        """npas_export_df["DD Probability"] = np.loadtxt(
            f"../data/tracked_units/npas_treatment_data/npas_treatment_cell_{k+1}_probabilities.txt",
            usecols=0,
        )
        npas_export_df["Naive Probability"] = np.loadtxt(
            f"../data/tracked_units/npas_treatment_data/npas_treatment_cell_{k+1}_probabilities.txt",
            usecols=1,
        )"""
        npas_export_df["Treatment Times"] = treatment_index
        npas_export_df["mouse"] = npas_mouse[k]
        npas_export_df["medial"] = npas_is_medial[k]
        npas_export_df["folder"] = npas_folder[k]
        npas_export_df["name"] = npas_cell_name[k]
        npas_export_df.to_csv(
            f"{npas_features_save_dir}/npas_treatment_cell_{k+1:04d}.csv"
        )

    # Save data
    for k in range(all_jaws_data.shape[2]):
        jaws_export_df = pd.DataFrame(
            data=all_jaws_data[:, :, k],
            columns=col_labels,
        )
        """jaws_export_df["DD Probability"] = np.loadtxt(
            f"../data/tracked_units/jaws_treatment_data/jaws_treatment_cell_{k+1}_probabilities.txt",
            usecols=0,
        )
        jaws_export_df["Naive Probability"] = np.loadtxt(
            f"../data/tracked_units/jaws_treatment_data/jaws_treatment_cell_{k+1}_probabilities.txt",
            usecols=1,
        )"""

        jaws_export_df["Treatment Times"] = treatment_index
        jaws_export_df["mouse"] = jaws_mouse[k]
        jaws_export_df["medial"] = jaws_is_medial[k]
        jaws_export_df["folder"] = jaws_folder[k]
        jaws_export_df["name"] = jaws_cell_name[k]
        jaws_export_df.to_csv(
            f"{jaws_features_save_dir}/jaws_treatment_cell_{k+1:04d}.csv"
        )

    sys.exit()

    # Load data
    delta_jaws = np.zeros(jaws_rates.shape)
    beta_jaws = np.zeros(jaws_rates.shape)
    for k in range(jaws_rates.shape[0]):
        osc = np.loadtxt(
            f"{save_dir}/jaws_neurons/tracked_units/cell_{k+1:04d}/osc_data.txt",
            delimiter=",",
        )
        delta_jaws[k, :] = osc[:, 1]
        beta_jaws[k, :] = osc[:, 4]

    delta_npas = np.zeros(npas_rates.shape)
    beta_npas = np.zeros(npas_rates.shape)
    for k in range(npas_rates.shape[0]):
        osc = np.loadtxt(
            f"{save_dir}/npas_neurons/tracked_units/cell_{k+1:04d}/osc_data.txt",
            delimiter=",",
        )
        delta_npas[k, :] = osc[:, 1]
        beta_npas[k, :] = osc[:, 4]

    # Save data
    jaws_treatment_data = np.zeros((jaws_rates.shape[1], 12 * 3))
    col_labels = []
    ### EXPORT MEANS ####
    col_labels.append(f"FR Mean")
    jaws_treatment_data[:, 0] = np.mean(jaws_rates, axis=0)
    ### EXPORT SEM ####
    col_labels.append(f"FR SEM")
    jaws_treatment_data[:, 1] = stats.sem(jaws_rates, axis=0)
    ### EXPORT STD ####
    col_labels.append(f"FR STD")
    jaws_treatment_data[:, 2] = np.std(jaws_rates, axis=0)
    ### EXPORT MEANS ####
    col_labels.append(f"CV Mean")
    jaws_treatment_data[:, 3] = np.mean(jaws_cvs, axis=0)
    ### EXPORT SEM ####
    col_labels.append(f"CV SEM")
    jaws_treatment_data[:, 4] = stats.sem(jaws_cvs, axis=0)
    ### EXPORT STD ####
    col_labels.append(f"CV STD")
    jaws_treatment_data[:, 5] = np.std(jaws_cvs, axis=0)
    ### EXPORT MEANS ####
    col_labels.append(f"Delta Power Mean")
    jaws_treatment_data[:, 6] = np.mean(delta_jaws, axis=0)
    ### EXPORT SEM ####
    col_labels.append(f"Delta Power SEM")
    jaws_treatment_data[:, 7] = stats.sem(delta_jaws, axis=0)
    ### EXPORT STD ####
    col_labels.append(f"Delta Power STD")
    jaws_treatment_data[:, 8] = np.std(delta_jaws, axis=0)
    ### EXPORT MEANS ####
    col_labels.append(f"Beta Power Mean")
    jaws_treatment_data[:, 9] = np.mean(beta_jaws, axis=0)
    ### EXPORT SEM ####
    col_labels.append(f"Beta Power SEM")
    jaws_treatment_data[:, 10] = stats.sem(beta_jaws, axis=0)
    ### EXPORT STD ####
    col_labels.append(f"Beta Power STD")
    jaws_treatment_data[:, 11] = np.std(beta_jaws, axis=0)
    count = 12
    burst_columns = [
        "Num Bursts",
        "Avg Burst Firing Rate",
        "Percent Time Bursting",
        "Percent of Spikes in Bursts",
        "Avg Burst Duration",
        "Avg Interburst Interval",
        "Non Bursting Firing Rate",
        "Burst Firing Rate Increase",
    ]
    label_count = 0
    for stat in [0, 1, 2, 3, 4, 5, 8, 9]:
        ### EXPORT MEANS ####
        col_labels.append(f"{burst_columns[label_count]} Mean")
        jaws_treatment_data[:, count] = np.mean(jaws_burst_stats[:, :, stat], axis=0)
        ### EXPORT SEM ####
        col_labels.append(f"{burst_columns[label_count]} SEM")
        jaws_treatment_data[:, count + 1] = stats.sem(
            jaws_burst_stats[:, :, stat], axis=0
        )
        ### EXPORT STD ####
        col_labels.append(f"{burst_columns[label_count]} STD")
        jaws_treatment_data[:, count + 2] = np.std(jaws_burst_stats[:, :, stat], axis=0)
        count += 3
        label_count += 1

    treat = 1
    treatment_index = []
    for i in range(jaws_rates.shape[1]):
        if i not in treatment_times:
            treatment_index.append(i * 30 - 180)
        else:
            treatment_index.append(f"Treatment {treat}")
            treat += 1
    treatment_index = np.asarray(treatment_index)

    sys.exit()

    delta_jaws = np.zeros(jaws_rates.shape)
    beta_jaws = np.zeros(jaws_rates.shape)
    for k in range(jaws_rates.shape[0]):
        osc = np.loadtxt(
            f"{save_dir}/jaws_neurons/tracked_units/cell_{k+1:04d}/osc_data.txt",
            delimiter=",",
        )
        delta_jaws[k, :] += osc[:, 1]
        beta_jaws[k, :] += osc[:, 4]

    delta_npas = np.zeros(npas_rates.shape)
    beta_npas = np.zeros(npas_rates.shape)
    for k in range(npas_rates.shape[0]):
        osc = np.loadtxt(
            f"{save_dir}/npas_neurons/tracked_units/cell_{k+1:04d}/osc_data.txt",
            delimiter=",",
        )
        delta_npas[k, :] = osc[:, 1]
        beta_npas[k, :] = osc[:, 4]

    jaws_treatment_data = np.zeros((jaws_rates.shape[1], 12 * 3))
    col_labels = []
    ### EXPORT MEANS ####
    col_labels.append(f"FR Mean")
    jaws_treatment_data[:, 0] = np.mean(jaws_rates, axis=0)
    ### EXPORT SEM ####
    col_labels.append(f"FR SEM")
    jaws_treatment_data[:, 1] = stats.sem(jaws_rates, axis=0)
    ### EXPORT STD ####
    col_labels.append(f"FR STD")
    jaws_treatment_data[:, 2] = np.std(jaws_rates, axis=0)
    ### EXPORT MEANS ####
    col_labels.append(f"CV Mean")
    jaws_treatment_data[:, 3] = np.mean(jaws_cvs, axis=0)
    ### EXPORT SEM ####
    col_labels.append(f"CV SEM")
    jaws_treatment_data[:, 4] = stats.sem(jaws_cvs, axis=0)
    ### EXPORT STD ####
    col_labels.append(f"CV STD")
    jaws_treatment_data[:, 5] = np.std(jaws_cvs, axis=0)
    ### EXPORT MEANS ####
    col_labels.append(f"Delta Power Mean")
    jaws_treatment_data[:, 6] = np.mean(delta_jaws, axis=0)
    ### EXPORT SEM ####
    col_labels.append(f"Delta Power SEM")
    jaws_treatment_data[:, 7] = stats.sem(delta_jaws, axis=0)
    ### EXPORT STD ####
    col_labels.append(f"Delta Power STD")
    jaws_treatment_data[:, 8] = np.std(delta_jaws, axis=0)
    ### EXPORT MEANS ####
    col_labels.append(f"Beta Power Mean")
    jaws_treatment_data[:, 9] = np.mean(beta_jaws, axis=0)
    ### EXPORT SEM ####
    col_labels.append(f"Beta Power SEM")
    jaws_treatment_data[:, 10] = stats.sem(beta_jaws, axis=0)
    ### EXPORT STD ####
    col_labels.append(f"Beta Power STD")
    jaws_treatment_data[:, 11] = np.std(beta_jaws, axis=0)
    count = 12
    burst_columns = [
        "Num Bursts",
        "Avg Burst Firing Rate",
        "Percent Time Bursting",
        "Percent of Spikes in Bursts",
        "Avg Burst Duration",
        "Avg Interburst Interval",
        "Non Bursting Firing Rate",
        "Burst Firing Rate Increase",
    ]
    label_count = 0
    for stat in [0, 1, 2, 3, 4, 5, 8, 9]:
        ### EXPORT MEANS ####
        col_labels.append(f"{burst_columns[label_count]} Mean")
        jaws_treatment_data[:, count] = np.mean(jaws_burst_stats[:, :, stat], axis=0)
        ### EXPORT SEM ####
        col_labels.append(f"{burst_columns[label_count]} SEM")
        jaws_treatment_data[:, count + 1] = stats.sem(
            jaws_burst_stats[:, :, stat], axis=0
        )
        ### EXPORT STD ####
        col_labels.append(f"{burst_columns[label_count]} STD")
        jaws_treatment_data[:, count + 2] = np.std(jaws_burst_stats[:, :, stat], axis=0)
        count += 3
        label_count += 1

    treat = 1
    treatment_index = []
    for i in range(jaws_rates.shape[1]):
        if i not in treatment_times:
            treatment_index.append(i * 30 - 180)
        else:
            treatment_index.append(f"Treatment {treat}")
            treat += 1
    treatment_index = np.asarray(treatment_index)

    jaws_export_df = pd.DataFrame(
        data=jaws_treatment_data, columns=col_labels, index=treatment_index
    )
    jaws_export_df.to_csv("../data/tracked_units/jaws_treatment_data.csv")

    npas_treatment_data = np.zeros((npas_rates.shape[1], 12 * 3))
    col_labels = []
    ### EXPORT MEANS ####
    col_labels.append(f"FR Mean")
    npas_treatment_data[:, 0] = np.mean(npas_rates, axis=0)
    ### EXPORT SEM ####
    col_labels.append(f"FR SEM")
    npas_treatment_data[:, 1] = stats.sem(npas_rates, axis=0)
    ### EXPORT STD ####
    col_labels.append(f"FR STD")
    npas_treatment_data[:, 2] = np.std(npas_rates, axis=0)
    ### EXPORT MEANS ####
    col_labels.append(f"CV Mean")
    npas_treatment_data[:, 3] = np.mean(npas_cvs, axis=0)
    ### EXPORT SEM ####
    col_labels.append(f"CV SEM")
    npas_treatment_data[:, 4] = stats.sem(npas_cvs, axis=0)
    ### EXPORT STD ####
    col_labels.append(f"CV STD")
    npas_treatment_data[:, 5] = np.std(npas_cvs, axis=0)
    ### EXPORT MEANS ####
    col_labels.append(f"Delta Power Mean")
    npas_treatment_data[:, 6] = np.mean(delta_npas, axis=0)
    ### EXPORT SEM ####
    col_labels.append(f"Delta Power SEM")
    npas_treatment_data[:, 7] = stats.sem(delta_npas, axis=0)
    ### EXPORT STD ####
    col_labels.append(f"Delta Power STD")
    npas_treatment_data[:, 8] = np.std(delta_npas, axis=0)
    ### EXPORT MEANS ####
    col_labels.append(f"Beta Power Mean")
    npas_treatment_data[:, 9] = np.mean(beta_npas, axis=0)
    ### EXPORT SEM ####
    col_labels.append(f"Beta Power SEM")
    npas_treatment_data[:, 10] = stats.sem(beta_npas, axis=0)
    ### EXPORT STD ####
    col_labels.append(f"Beta Power STD")
    npas_treatment_data[:, 11] = np.std(beta_npas, axis=0)
    count = 12
    burst_columns = [
        "Num Bursts",
        "Avg Burst Firing Rate",
        "Percent Time Bursting",
        "Percent of Spikes in Bursts",
        "Avg Burst Duration",
        "Avg Interburst Interval",
        "Non Bursting Firing Rate",
        "Burst Firing Rate Increase",
    ]
    label_count = 0
    for stat in [0, 1, 2, 3, 4, 5, 8, 9]:
        ### EXPORT MEANS ####
        col_labels.append(f"{burst_columns[label_count]} Mean")
        npas_treatment_data[:, count] = np.mean(npas_burst_stats[:, :, stat], axis=0)
        ### EXPORT SEM ####
        col_labels.append(f"{burst_columns[label_count]} SEM")
        npas_treatment_data[:, count + 1] = stats.sem(
            npas_burst_stats[:, :, stat], axis=0
        )
        ### EXPORT STD ####
        col_labels.append(f"{burst_columns[label_count]} STD")
        npas_treatment_data[:, count + 2] = np.std(npas_burst_stats[:, :, stat], axis=0)
        count += 3
        label_count += 1

    treat = 1
    treatment_index = []
    for i in range(jaws_rates.shape[1]):
        if i not in treatment_times:
            treatment_index.append(i * 30 - 180)
        else:
            treatment_index.append(f"Treatment {treat}")
            treat += 1
    treatment_index = np.asarray(treatment_index)

    npas_export_df = pd.DataFrame(
        data=npas_treatment_data, columns=col_labels, index=treatment_index
    )
    npas_export_df.to_csv("../data/tracked_units/npas_treatment_data.csv")

    plot_sem = False
    fig, ax = plt.subplots(4, 3, figsize=(14, 10), dpi=300, tight_layout=True)
    axes = [ax[i, j] for i in range(4) for j in range(3)]

    axes[0].errorbar(
        x_values,
        np.mean(jaws_rates[:, x_values], axis=0),
        yerr=stats.sem(jaws_rates[:, x_values], axis=0) if plot_sem else 0,
        marker="o",
        color="b",
        ls="dashed",
        capsize=2,
        alpha=0.25,
        markersize=2,
    )
    axes[0].errorbar(
        x_values,
        np.mean(npas_rates[:, x_values], axis=0),
        yerr=stats.sem(npas_rates[:, x_values], axis=0) if plot_sem else 0,
        marker="o",
        color="r",
        ls="dashed",
        capsize=2,
        alpha=0.25,
        markersize=2,
    )
    sns.regplot(
        x=x_values,
        y=np.mean(npas_rates[:, x_values], axis=0),
        color="r",
        ax=axes[0],
        scatter=False,
    )
    sns.regplot(
        x=x_values,
        y=np.mean(jaws_rates[:, x_values], axis=0),
        color="b",
        ax=axes[0],
        scatter=False,
    )

    axes[1].errorbar(
        x_values,
        np.mean(jaws_cvs[:, x_values], axis=0),
        yerr=stats.sem(jaws_cvs[:, x_values], axis=0) if plot_sem else 0,
        marker="o",
        color="b",
        ls="dashed",
        capsize=2,
        alpha=0.25,
        markersize=2,
    )
    axes[1].errorbar(
        x_values,
        np.mean(npas_cvs[:, x_values], axis=0),
        yerr=stats.sem(npas_cvs[:, x_values], axis=0) if plot_sem else 0,
        marker="o",
        color="r",
        ls="dashed",
        capsize=2,
        alpha=0.25,
        markersize=2,
    )

    sns.regplot(
        x=x_values,
        y=np.mean(npas_cvs[:, x_values], axis=0),
        color="r",
        ax=axes[1],
        scatter=False,
    )
    sns.regplot(
        x=x_values,
        y=np.mean(jaws_cvs[:, x_values], axis=0),
        color="b",
        ax=axes[1],
        scatter=False,
    )

    ### PLOT DELTA
    axes[2].errorbar(
        x_values,
        np.mean(delta_jaws[:, x_values], axis=0),
        yerr=stats.sem(delta_jaws[:, x_values], axis=0) if plot_sem else 0,
        marker="o",
        color="b",
        ls="dashed",
        capsize=2,
        alpha=0.25,
        markersize=2,
    )
    sns.regplot(
        x=x_values,
        y=np.mean(delta_jaws[:, x_values], axis=0),
        color="b",
        ax=axes[2],
        scatter=False,
    )
    axes[2].errorbar(
        x_values,
        np.mean(delta_npas[:, x_values], axis=0),
        yerr=stats.sem(delta_npas[:, x_values], axis=0) if plot_sem else 0,
        marker="o",
        color="r",
        ls="dashed",
        capsize=2,
        alpha=0.25,
        markersize=2,
    )
    sns.regplot(
        x=x_values,
        y=np.mean(delta_npas[:, x_values], axis=0),
        color="r",
        ax=axes[2],
        scatter=False,
    )

    ### PLOT BETA
    axes[3].errorbar(
        x_values,
        np.mean(beta_jaws[:, x_values], axis=0),
        yerr=stats.sem(beta_jaws[:, x_values], axis=0) if plot_sem else 0,
        marker="o",
        color="b",
        ls="dashed",
        capsize=2,
        alpha=0.25,
        markersize=2,
    )
    sns.regplot(
        x=x_values,
        y=np.mean(beta_jaws[:, x_values], axis=0),
        color="b",
        ax=axes[3],
        scatter=False,
    )
    axes[3].errorbar(
        x_values,
        np.mean(beta_npas[:, x_values], axis=0),
        yerr=stats.sem(beta_npas[:, x_values], axis=0) if plot_sem else 0,
        marker="o",
        color="r",
        ls="dashed",
        capsize=2,
        alpha=0.25,
        markersize=2,
    )
    sns.regplot(
        x=x_values,
        y=np.mean(beta_npas[:, x_values], axis=0),
        color="r",
        ax=axes[3],
        scatter=False,
    )

    axis_count = 4
    for stat in [0, 1, 2, 3, 4, 5, 8, 9]:
        sns.regplot(
            x=x_values,
            y=np.mean(jaws_burst_stats[:, x_values, stat], axis=0),
            color="b",
            ax=axes[axis_count],
            scatter=False,
        )
        sns.regplot(
            x=x_values,
            y=np.mean(npas_burst_stats[:, x_values, stat], axis=0),
            color="r",
            ax=axes[axis_count],
            scatter=False,
        )

        axes[axis_count].errorbar(
            x_values,
            np.mean(jaws_burst_stats[:, x_values, stat], axis=0),
            yerr=(
                stats.sem(jaws_burst_stats[:, x_values, stat], axis=0)
                if plot_sem
                else 0
            ),
            marker="o",
            color="b",
            ls="dashed",
            capsize=2,
            alpha=0.25,
            markersize=2,
        )
        axes[axis_count].errorbar(
            x_values,
            np.mean(npas_burst_stats[:, x_values, stat], axis=0),
            yerr=(
                stats.sem(npas_burst_stats[:, x_values, stat], axis=0)
                if plot_sem
                else 0
            ),
            marker="o",
            color="r",
            ls="dashed",
            capsize=2,
            alpha=0.25,
            markersize=2,
        )
        axis_count += 1
    ylabels = [
        "Firing Rate",
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

    for i in range(len(axes)):
        axes[i].set_ylabel(ylabels[i])
        axes[i].set_xlabel("Time (s)")
        ylims = axes[i].get_ylim()
        for stim in np.arange(6, jaws_rates.shape[1], 7):
            rect = patches.Rectangle(
                (stim - 0.25, ylims[0]),
                0.5,
                ylims[1] - ylims[0],
                color="orange",
                alpha=0.15,
            )
            axes[i].add_patch(rect)
        axes[i].set_ylim(ylims)

    makeNice(axes)
    fig.savefig("../data/tracked_units_features.pdf", bbox_inches="tight")
    plt.close()
    run_cmd("open data/tracked_units_features.pdf ")


training_fixed_length = 30
"""plot_tracked_healthy_dd(
    "../data/tracked_units/jaws_treatment_data", "../data/tracked_units/npas_treatment_data"
)"""


# create_ephys_tracked_units(training_fixed_length)

create_feature_results()
