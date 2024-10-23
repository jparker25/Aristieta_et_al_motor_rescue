import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os, sys
from scipy import stats
from scipy.stats import ks_2samp

# user modules
import run_pca
import run_logistic_regression
import run_neural_net
from helpers import *
import clean_data

import plot_data

renewal_power = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.labelsize"] = 8


def get_naive_dd_training_data_set():
    naive_osc_data = np.loadtxt(
        "../data/training/naive_neurons/osc_data.txt", delimiter=","
    )
    dd_osc_data = np.loadtxt("../data/training/dd_neurons/osc_data.txt", delimiter=",")

    naive_frs = np.loadtxt("../data/training/naive_neurons/cell_baseline_frs.txt")
    naive_cvs = np.loadtxt("../data/training/naive_neurons/cell_baseline_cvs.txt")
    naive_lengths = np.loadtxt(
        "../data/training/naive_neurons/cell_baseline_lengths.txt"
    )

    naive_burst_statistics = np.loadtxt(
        "../data/training/naive_neurons/cell_baseline_burst_statistics.txt"
    )

    dd_frs = np.loadtxt("../data/training/dd_neurons/cell_baseline_frs.txt")
    dd_cvs = np.loadtxt("../data/training/dd_neurons/cell_baseline_cvs.txt")
    dd_lengths = np.loadtxt("../data/training/dd_neurons/cell_baseline_lengths.txt")
    dd_burst_statistics = np.loadtxt(
        "../data/training/dd_neurons/cell_baseline_burst_statistics.txt"
    )

    naive_mouse = open(
        f"../data/training/naive_neurons/cell_mouse.txt", "r"
    ).readlines()
    naive_folder = open(
        f"../data/training/naive_neurons/cell_folders.txt", "r"
    ).readlines()
    naive_cell_names = open(
        f"../data/training/naive_neurons/cell_names.txt", "r"
    ).readlines()

    dd_mouse = open(f"../data/training/dd_neurons/cell_mouse.txt", "r").readlines()
    dd_folder = open(f"../data/training/dd_neurons/cell_folders.txt", "r").readlines()
    dd_cell_names = open(f"../data/training/dd_neurons/cell_names.txt", "r").readlines()

    naive_df = pd.DataFrame(
        {
            "Type": 1,
            "FR": naive_frs,
            "CV": naive_cvs,
            # "T": naive_lengths,
            # "Delta Osc": naive_osc_data[:, 0],
            "Delta Power": naive_osc_data[:, 1 if renewal_power else 2],
            # "Beta Osc": naive_osc_data[:, 4],
            "Beta Power": naive_osc_data[:, 5 if renewal_power else 6],
            "Num Bursts": naive_burst_statistics[:, 0] / naive_lengths,
            "Avg Burst Firing Rate": naive_burst_statistics[:, 1],
            "Percent Time Bursting": naive_burst_statistics[:, 2],
            "Percent of Spikes in Bursts": naive_burst_statistics[:, 3],
            "Avg Burst Duration": naive_burst_statistics[:, 4],
            "Avg Interburst Interval": naive_burst_statistics[:, 5],
            # "CV Interburst Interval": naive_burst_statistics[:, 6],
            # "Avg Surprise": naive_burst_statistics[:, 7],
            "Non Bursting Firing Rate": naive_burst_statistics[:, 8],
            "Burst Firing Rate Increase": naive_burst_statistics[:, 9],
            "mouse": naive_mouse,
            "folder": naive_folder,
            "name": naive_cell_names,
        }
    )

    dd_df = pd.DataFrame(
        {
            "Type": 0,
            "FR": dd_frs,
            "CV": dd_cvs,
            # "T": dd_lengths,
            # "Delta Osc": dd_osc_data[:, 0],
            "Delta Power": dd_osc_data[:, 1 if renewal_power else 2],
            # "Beta Osc": dd_osc_data[:, 4],
            "Beta Power": dd_osc_data[:, 5 if renewal_power else 6],
            "Num Bursts": dd_burst_statistics[:, 0] / dd_lengths,
            "Avg Burst Firing Rate": dd_burst_statistics[:, 1],
            "Percent Time Bursting": dd_burst_statistics[:, 2],
            "Percent of Spikes in Bursts": dd_burst_statistics[:, 3],
            "Avg Burst Duration": dd_burst_statistics[:, 4],
            "Avg Interburst Interval": dd_burst_statistics[:, 5],
            # "CV Interburst Interval": dd_burst_statistics[:, 6],
            # "Avg Surprise": dd_burst_statistics[:, 7],
            "Non Bursting Firing Rate": dd_burst_statistics[:, 8],
            "Burst Firing Rate Increase": dd_burst_statistics[:, 9],
            "mouse": dd_mouse,
            "folder": dd_folder,
            "name": dd_cell_names,
        }
    )

    naive_df.fillna(0, inplace=True)
    dd_df.fillna(0, inplace=True)

    return naive_df, dd_df, pd.concat([naive_df, dd_df], ignore_index=True)


def get_jaws_npas_data_set():
    source_path = "../data/training"
    dataframes = []
    paths = [
        "jaws_neurons/pre_opto",
        "jaws_neurons/post_opto",
        "npas_neurons/pre_opto",
        "npas_neurons/post_opto",
    ]
    is_medial = []
    path_count = 0
    for path in paths:
        is_medial.append(
            np.loadtxt(f"{source_path}/{path}/is_medial.txt", delimiter=",")
        )
        osc_data = np.loadtxt(f"{source_path}/{path}/osc_data.txt", delimiter=",")

        frs = np.loadtxt(f"{source_path}/{path}/cell_frs.txt")
        cvs = np.loadtxt(f"{source_path}/{path}/cell_cvs.txt")
        lengths = np.loadtxt(f"{source_path}/{path}/cell_lengths.txt")

        burst_statistics = np.loadtxt(f"{source_path}/{path}/cell_burst_statistics.txt")

        mouse = open(f"{source_path}/{path}/cell_mouse.txt", "r").readlines()
        folder = open(f"{source_path}/{path}/cell_folders.txt", "r").readlines()
        cell_names = open(f"{source_path}/{path}/cell_names.txt", "r").readlines()

        post_times = np.zeros(len(frs))
        if path_count == 1 or path_count == 3:
            post_times = np.loadtxt(
                f"{source_path}/{'jaws' if path_count == 1 else 'npas'}_neurons/post_opto/cell_post_times.txt"
            )

        df = pd.DataFrame(
            {
                "FR": frs,
                "CV": cvs,
                # "T": lengths,
                # "Delta Osc": osc_data[:, 0],
                "Delta Power": osc_data[:, 1 if renewal_power else 2],
                # "Beta Osc": osc_data[:, 4],
                "Beta Power": osc_data[:, 5 if renewal_power else 6],
                "Num Bursts": burst_statistics[:, 0] / lengths,
                "Avg Burst Firing Rate": burst_statistics[:, 1],
                "Percent Time Bursting": burst_statistics[:, 2],
                "Percent of Spikes in Bursts": burst_statistics[:, 3],
                "Avg Burst Duration": burst_statistics[:, 4],
                "Avg Interburst Interval": burst_statistics[:, 5],
                # "CV Interburst Interval": burst_statistics[:, 6],
                # "Avg Surprise": burst_statistics[:, 7],
                "Non Bursting Firing Rate": burst_statistics[:, 8],
                "Burst Firing Rate Increase": burst_statistics[:, 9],
                "mouse": "JAWS" if path_count < 2 else "NPAS",
                "folder": folder,
                "name": cell_names,
                "Post-Time": post_times,
                "Medial": is_medial[path_count],
            }
        )
        path_count += 1
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def get_jaws_npas_tracked_data():
    source_path = "../data/training"
    dataframes = []
    paths = [
        "jaws_neurons/pre_opto",
        "jaws_neurons/post_opto",
        "npas_neurons/pre_opto",
        "npas_neurons/post_opto",
    ]
    return 0


def generate_post_dataframes(dataframe, times, time_chunks, is_medial):
    dataframe["Post-Time"] = times
    dataframe["Medial"] = is_medial
    dataframe = dataframe.sort_values(by="Post-Time")
    chunked_data = []
    for i in range(len(time_chunks)):
        if type(time_chunks[i]) is not list:
            start_index = dataframe["Post-Time"].searchsorted(
                time_chunks[i], side="left"
            )
            stop_index = dataframe["Post-Time"].searchsorted(
                time_chunks[i], side="right"
            )

            chunked_data.append(dataframe.iloc[start_index:stop_index, :])
        else:
            start_index = dataframe["Post-Time"].searchsorted(
                time_chunks[i][0], side="left"
            )
            stop_index = dataframe["Post-Time"].searchsorted(
                time_chunks[i][-1], side="right"
            )
            chunked_data.append(dataframe.iloc[start_index:stop_index, :])

    return chunked_data


def generate_pre_post_dataframes(
    jaws_npas_df, time_chunks=[[0], [30], [60], [90, 120], [150, 180, 210]]
):
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
jaws_npas_pre.insert(0, "Type", np.zeros(len(jaws_npas_pre)))

combined_df = pd.concat([training_df, jaws_npas_pre], ignore_index=True)

types = combined_df["Type"]
combined_df = combined_df.drop(columns="Type")
combined_df["Type"] = types

### REMOVE OUTLIERS ###
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
    "Type": False,
    "FR": True,
    "CV": True,
    "Delta Osc": False,
    "Delta Power": True,
    "Beta Osc": False,
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
    "mouse": False,
    "folder": False,
    "name": False,
}

plot_data.plot_feature_histograms_separate(combined_df, feature_outlier_strength)


### GATHER THE FEATURES IN AN ARRAY ###
feature_array = []
count = 0
for col in combined_df.columns:
    if use_feature[col]:
        feature_array.append(count)
    count += 1

mlp = False
pca = False
feature_removal = True
num_seeds = 15
training_split = 0.8

if mlp:
    run_neural_net.predict_motor_rescue(
        combined_df,
        feature_array,
        feature_outlier_strength,
        jaws_npas_pre_post,
        train_amount=training_split,
        seeds=np.arange(num_seeds),
        show=True,
        test_outliers=False,
        nonlinear_transforms=False,
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

    sys.exit()
    run_neural_net.feature_importance_selected(
        combined_df,
        feature_array,
        feature_outlier_strength,
        train_amount=training_split,
        seeds=np.arange(num_seeds),
        show=False,
    )

if pca:
    run_pca.run_pca_for_figures(combined_df, feature_array, feature_outlier_strength)
