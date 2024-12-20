"""
run_analysis.py

This script analyzes naive and DD spike trains to determine pathology level via MLPs and PCA.

Author: John E. Parker (2024)
"""

# python modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# user modules
import run_pca
import run_neural_net
from helpers import *
import plot_data

# Flag to change power as renewal fixed or not (Whalen et al, 2020)
renewal_power = False  # Default False
# Set font family and sizes
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.labelsize"] = 8


def get_naive_dd_training_data_set():
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
    dd_osc_data = np.loadtxt("../data/training/dd_neurons/osc_data.txt", delimiter=",")

    # Load naive spike train statistics
    naive_frs = np.loadtxt("../data/training/naive_neurons/cell_baseline_frs.txt")
    naive_cvs = np.loadtxt("../data/training/naive_neurons/cell_baseline_cvs.txt")
    naive_lengths = np.loadtxt(
        "../data/training/naive_neurons/cell_baseline_lengths.txt"
    )

    naive_burst_statistics = np.loadtxt(
        "../data/training/naive_neurons/cell_baseline_burst_statistics.txt"
    )

    # Load DD spike train statistics
    dd_frs = np.loadtxt("../data/training/dd_neurons/cell_baseline_frs.txt")
    dd_cvs = np.loadtxt("../data/training/dd_neurons/cell_baseline_cvs.txt")
    dd_lengths = np.loadtxt("../data/training/dd_neurons/cell_baseline_lengths.txt")
    dd_burst_statistics = np.loadtxt(
        "../data/training/dd_neurons/cell_baseline_burst_statistics.txt"
    )

    # Load naive spike train meta data
    naive_mouse = open(
        f"../data/training/naive_neurons/cell_mouse.txt", "r"
    ).readlines()
    naive_folder = open(
        f"../data/training/naive_neurons/cell_folders.txt", "r"
    ).readlines()
    naive_cell_names = open(
        f"../data/training/naive_neurons/cell_names.txt", "r"
    ).readlines()

    # Load DD spike train meta data
    dd_mouse = open(f"../data/training/dd_neurons/cell_mouse.txt", "r").readlines()
    dd_folder = open(f"../data/training/dd_neurons/cell_folders.txt", "r").readlines()
    dd_cell_names = open(f"../data/training/dd_neurons/cell_names.txt", "r").readlines()

    # Construct naive dataframe
    naive_df = pd.DataFrame(
        {
            "Type": 1,
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
            "mouse": naive_mouse,
            "folder": naive_folder,
            "name": naive_cell_names,
        }
    )

    # Constrcut DD dataframe
    dd_df = pd.DataFrame(
        {
            "Type": 0,
            "FR": dd_frs,
            "CV": dd_cvs,
            # "T": dd_lengths, # feature not used
            # "Delta Osc": dd_osc_data[:, 0], # feature not used
            "Delta Power": dd_osc_data[:, 1 if renewal_power else 2],
            # "Beta Osc": dd_osc_data[:, 4], # feature not used
            "Beta Power": dd_osc_data[:, 5 if renewal_power else 6],
            "Num Bursts": dd_burst_statistics[:, 0] / dd_lengths,
            "Avg Burst Firing Rate": dd_burst_statistics[:, 1],
            "Percent Time Bursting": dd_burst_statistics[:, 2],
            "Percent of Spikes in Bursts": dd_burst_statistics[:, 3],
            "Avg Burst Duration": dd_burst_statistics[:, 4],
            "Avg Interburst Interval": dd_burst_statistics[:, 5],
            # "CV Interburst Interval": dd_burst_statistics[:, 6], # feature not used
            # "Avg Surprise": dd_burst_statistics[:, 7], # feature not used
            "Non Bursting Firing Rate": dd_burst_statistics[:, 8],
            "Burst Firing Rate Increase": dd_burst_statistics[:, 9],
            "mouse": dd_mouse,
            "folder": dd_folder,
            "name": dd_cell_names,
        }
    )

    # Replace NaNs with 0s
    naive_df.fillna(0, inplace=True)
    dd_df.fillna(0, inplace=True)

    return naive_df, dd_df, pd.concat([naive_df, dd_df], ignore_index=True)


def get_jaws_npas_data_set():
    """
    Generates JAWS and NPAS combined dataframe of all spike train features.

    Returns
    -------
    \t combined_df (dataframe) : feature dataframe of all JAWS and NPAS data
    """

    # Define paths for JAWS and NPAS data
    source_path = "../data/training"
    dataframes = []
    paths = [
        "jaws_neurons/pre_opto",
        "jaws_neurons/post_opto",
        "npas_neurons/pre_opto",
        "npas_neurons/post_opto",
    ]
    is_medial = []

    # Loop through paths and generate dataframe, append to dataframe list
    path_count = 0
    for path in paths:

        # Load is_medial boolean list
        is_medial.append(
            np.loadtxt(f"{source_path}/{path}/is_medial.txt", delimiter=",")
        )

        # Load oscillation data
        osc_data = np.loadtxt(f"{source_path}/{path}/osc_data.txt", delimiter=",")

        # Load spike train features
        frs = np.loadtxt(f"{source_path}/{path}/cell_frs.txt")
        cvs = np.loadtxt(f"{source_path}/{path}/cell_cvs.txt")
        lengths = np.loadtxt(f"{source_path}/{path}/cell_lengths.txt")
        burst_statistics = np.loadtxt(f"{source_path}/{path}/cell_burst_statistics.txt")

        # Load spike train meta data
        mouse = open(f"{source_path}/{path}/cell_mouse.txt", "r").readlines()
        folder = open(f"{source_path}/{path}/cell_folders.txt", "r").readlines()
        cell_names = open(f"{source_path}/{path}/cell_names.txt", "r").readlines()

        # Load recording times for post-stim data
        post_times = np.zeros(len(frs))
        if path_count == 1 or path_count == 3:
            post_times = np.loadtxt(
                f"{source_path}/{'jaws' if path_count == 1 else 'npas'}_neurons/post_opto/cell_post_times.txt"
            )

        # Construct dataframe
        df = pd.DataFrame(
            {
                "FR": frs,
                "CV": cvs,
                # "T": lengths, # feature not used
                # "Delta Osc": osc_data[:, 0], # feature not used
                "Delta Power": osc_data[:, 1 if renewal_power else 2],
                # "Beta Osc": osc_data[:, 4], # feature not used
                "Beta Power": osc_data[:, 5 if renewal_power else 6],
                "Num Bursts": burst_statistics[:, 0] / lengths,
                "Avg Burst Firing Rate": burst_statistics[:, 1],
                "Percent Time Bursting": burst_statistics[:, 2],
                "Percent of Spikes in Bursts": burst_statistics[:, 3],
                "Avg Burst Duration": burst_statistics[:, 4],
                "Avg Interburst Interval": burst_statistics[:, 5],
                # "CV Interburst Interval": burst_statistics[:, 6], # feature not used
                # "Avg Surprise": burst_statistics[:, 7], # feature not used
                "Non Bursting Firing Rate": burst_statistics[:, 8],
                "Burst Firing Rate Increase": burst_statistics[:, 9],
                "mouse": "JAWS" if path_count < 2 else "NPAS",
                "folder": folder,
                "name": cell_names,
                "Post-Time": post_times,
                "Medial": is_medial[path_count],
            }
        )

        # Update dataframe list and list counter
        path_count += 1
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def generate_pre_post_dataframes(
    jaws_npas_df, time_chunks=[[0], [30], [60], [90, 120], [150, 180, 210]]
):
    """
    Splits dataframe a list of dataframes based on list of groups of times.

    Parameters
    ---------
    \t jaws_npas_df (pandas dataframe) :  Release date in string format.

    \t time_chunks (list) : List of lists of times to split dataframe

    Returns
    -------
    \t pre_post_dfs (list) : List of dataframes split by times
    """
    pre_post_dfs = []
    for chunk in time_chunks:
        pre_post_dfs.append(jaws_npas_df[jaws_npas_df["Post-Time"].isin(chunk)])
    return pre_post_dfs


# get cont and pulse naive dd datasets
naive_df, dd_df, df = get_naive_dd_training_data_set()

# get npas and jaws data sets, chunk into list of data frames pre and post
jaws_npas_df = get_jaws_npas_data_set()
jaws_npas_pre_post = generate_pre_post_dataframes(jaws_npas_df)

# combine pre data for jaws and npas to training data
training_df = df.copy()
training_df = training_df.drop(columns=["mouse", "folder", "name"])
jaws_npas_pre = jaws_npas_pre_post[0].copy()
jaws_npas_pre = jaws_npas_pre.drop(
    columns=["mouse", "folder", "name", "Post-Time", "Medial"]
)
# insert types column at beginning of dataframe
jaws_npas_pre.insert(0, "Type", np.zeros(len(jaws_npas_pre)))

# Combine training data and pre-stim jaws and npas data
combined_df = pd.concat([training_df, jaws_npas_pre], ignore_index=True)

# Extract types column and move to back of dataframe
types = combined_df["Type"]
combined_df = combined_df.drop(columns="Type")
combined_df["Type"] = types

### REMOVE OUTLIERS ###
# Set zscore outlier threshold, can adjust individual features if desired
zscore_threshold = 3
feature_outlier_strength = {
    "FR": zscore_threshold,
    "CV": zscore_threshold,
    "Delta Power": zscore_threshold,
    "Beta Power": zscore_threshold,
    "Num Bursts": zscore_threshold,
    "Avg Burst Firing Rate": zscore_threshold,
    "Percent Time Bursting": zscore_threshold,
    "Percent of Spikes in Bursts": zscore_threshold,
    "Avg Burst Duration": zscore_threshold,
    "Avg Interburst Interval": zscore_threshold,
    "Non Bursting Firing Rate": zscore_threshold,
    "Burst Firing Rate Increase": zscore_threshold,
}

### SELECT FEATURES TO USE IN ANALYSIS ###
use_feature = {
    "Type": False,  # Must be false
    "FR": True,
    "CV": True,
    "Delta Osc": False,  # Must be false
    "Delta Power": True,
    "Beta Osc": False,  # Must be false
    "Beta Power": True,
    "Num Bursts": True,
    "Avg Burst Firing Rate": True,
    "Percent Time Bursting": True,
    "Percent of Spikes in Bursts": True,
    "Avg Burst Duration": True,
    "Avg Interburst Interval": True,
    "CV Interburst Interval": False,
    "Avg Surprise": False,
    "Non Bursting Firing Rate": True,
    "Burst Firing Rate Increase": True,
    "mouse": False,  # Must be false
    "folder": False,  # Must be false
    "name": False,  # Must be false
}

# Plots feature histgorams and CDFs of training data
plot_data.plot_feature_histograms_separate(combined_df, feature_outlier_strength)

# Plots feature Box plots and prints out MWU results for pre v post comparisons
plot_data.plot_pre_post_boxplots(jaws_npas_df, show=False)

### GATHER THE FEATURES IN AN ARRAY ###
feature_array = []
count = 0
for col in combined_df.columns:
    if use_feature[col]:
        feature_array.append(count)
    count += 1

# Determine which analyses to run
mlp = True
pca = False
feature_removal = True

# Seeds (15) and train/test split (0.8)
num_seeds = 15
training_split = 0.8


# Run analyses requested
if mlp:
    run_neural_net.predict_motor_rescue(
        combined_df,
        feature_array,
        feature_outlier_strength,
        jaws_npas_pre_post,
        train_amount=training_split,
        seeds=np.arange(num_seeds),
        show=False,
        test_outliers=False,
        min_max_scale=False,
    )

if feature_removal:
    run_neural_net.feature_importance_clustered(
        combined_df,
        feature_array,
        feature_outlier_strength,
        train_amount=training_split,
        seeds=np.arange(num_seeds),
        show=False,
    )

if pca:
    run_pca.run_pca_for_figures(
        combined_df, feature_array, feature_outlier_strength, seeds=np.arange(num_seeds)
    )
