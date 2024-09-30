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


def get_jaws_npas_data_set(source_path):
    dataframes = []
    paths = [
        "jaws_neurons/pre_opto",
        "jaws_neurons/post_opto",
        "npas_neurons/pre_opto",
        "npas_neurons/post_opto",
    ]
    is_medial = []
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

        df = pd.DataFrame(
            {
                "FR": frs,
                "CV": cvs,
                # "T": lengths,
                "Delta Osc": osc_data[:, 0],
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
                "mouse": mouse,
                "folder": folder,
                "name": cell_names,
            }
        )
        dataframes.append(df)
    jaws_post_times = np.loadtxt(
        f"{source_path}/jaws_neurons/post_opto/cell_post_times.txt"
    )
    npas_post_times = np.loadtxt(
        f"{source_path}/npas_neurons/post_opto/cell_post_times.txt"
    )
    return (
        dataframes[0],
        dataframes[1],
        dataframes[2],
        dataframes[3],
        jaws_post_times,
        npas_post_times,
        is_medial,
    )


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


### TRAINING DATA SET ###
naive_df, dd_df, df = get_naive_dd_training_data_set()

### TRAINING DATA SET FOR EPHYS ###
(
    jaws_pre_df,
    jaws_post_df,
    npas_pre_df,
    npas_post_df,
    jaws_post_times,
    npas_post_times,
    is_medial,
) = get_jaws_npas_data_set("../data/training")

### CONSTRUCT VARIABLES FOR EASIER USAGE ##
motor_rescue_dfs = [jaws_pre_df, jaws_post_df, npas_pre_df, npas_post_df]

"""jaws_post_df["Post-Time"] = jaws_post_times
jaws_post_df["Medial"] = is_medial[1]
jaws_pre_df["Post-Time"] = 0
jaws_pre_df["Medial"] = is_medial[0]

npas_post_df["Post-Time"] = npas_post_times
npas_post_df["Medial"] = is_medial[3]
npas_pre_df["Post-Time"] = 0
npas_pre_df["Medial"] = is_medial[2]"""

### CHUNKING DATA
chunked = True
time_chunks = [30, 60, 90, 120, 150, 180, 210]  # time intervals
if chunked:  # create groups of time intervals
    # time_chunks = [30, 60, [90, 120], [150, 180, 210]]
    time_chunks = [30, 60, [90, 120], [150, 180, 210]]


"""jaws_tmp_df = pd.concat([jaws_pre_df, jaws_post_df], ignore_index=True)
print(jaws_tmp_df.columns)
jaws_pre_post_dfs = []
for chunk in time_chunks:
    jaws_pre_post_dfs.append(jaws_tmp_df[jaws_tmp_df["Post-Time"].isin(chunk)])

print([len(x) for x in jaws_pre_post_dfs])
sys.exit()"""

### GET POST DATA BASED ON TIME CHUNKS
post_npas_dfs = generate_post_dataframes(
    npas_post_df, npas_post_times, time_chunks, is_medial[3]
)
post_jaws_dfs = generate_post_dataframes(
    jaws_post_df, jaws_post_times, time_chunks, is_medial[1]
)


### GET OSCILLATIONS
motor_rescue_dfs_osc = [
    jaws_pre_df["Delta Osc"],
    jaws_post_df["Delta Osc"],
    npas_pre_df["Delta Osc"],
    npas_post_df["Delta Osc"],
]

### GET POST OSCILLATIONS
post_npas_dfs_osc = [x["Delta Osc"] for x in post_npas_dfs]
post_jaws_dfs_osc = [x["Delta Osc"] for x in post_jaws_dfs]

### PARSE DFS WITHOUT OSCILLATION DATA
motor_rescue_dfs = [x.drop(["Delta Osc"], axis=1) for x in motor_rescue_dfs]
post_npas_dfs = [x.drop(["Delta Osc"], axis=1) for x in post_npas_dfs]
post_jaws_dfs = [x.drop(["Delta Osc"], axis=1) for x in post_jaws_dfs]

### COMBINE PRE STIM MOTOR DFS FOR TRAINING
pre_motor_df = pd.concat([motor_rescue_dfs[0], motor_rescue_dfs[2]])
pre_motor_df["Type"] = 0

### LABELS FOR PLOTTING CHUNKS
show = False
xlabels = [
    f"{30*i+30}\nn={len(post_jaws_dfs[i])}\nn={len(post_npas_dfs[i])}"
    for i in range(len(post_jaws_dfs))
]
xlabels.insert(0, "Pre-Stim")
ylabels = [int(x) for x in np.unique(jaws_post_times)]

if chunked:
    xlabels = [
        f"Pre-Stim\nJAWS n={len(motor_rescue_dfs[0])}\nNpas n={len(motor_rescue_dfs[2])}",
        f"30\nn={len(post_jaws_dfs[0])}\nn={len(post_npas_dfs[0])}",
        f"60\nn={len(post_jaws_dfs[1])}\nn={len(post_npas_dfs[1])}",
        f"90-120\nn={len(post_jaws_dfs[2])}\nn={len(post_npas_dfs[2])}",
        f"150-210\nn={len(post_jaws_dfs[3])}\nn={len(post_npas_dfs[3])}",
    ]
    ylabels = [30, 60, "90-120", "150-210"]


### COMBINE NAIVE AND DD DATAFRAMES INTO ONE
df = pd.concat([naive_df, dd_df])
df_type = df.pop("Type")
df.insert(df.shape[1], "Type", df_type)
df = pd.concat([df, pre_motor_df], ignore_index=True)

### ASSIGN LABELS FOR NAIVE AND DD
combined_naive_df = df[df["Type"] == 1]
combined_dd_df = df[df["Type"] == 0]

df = pd.concat([combined_naive_df, combined_dd_df])

print(df)


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

df = clean_data.remove_outliers_by_group_zscore_independent(
    combined_naive_df, combined_dd_df, feature_outlier_strength
)

df.to_csv("../data/df.csv")

# plot bursts vs delta power
plot_data.bursts_delta(df)

### PLOT FEATURES HISTOGRAMS FOR TRAINING ###
plot_data.plot_feature_histograms(df)

# Plot pre and post JAWS and Npas features
plot_data.feature_time(
    motor_rescue_dfs, post_jaws_dfs, post_npas_dfs, xlabels, show=show
)

# Plot pre and post histograms for Jaws and Npas
plot_data.plot_pre_post_histograms(
    motor_rescue_dfs, post_jaws_dfs, post_npas_dfs, show=show
)

# plot corr matrix
plot_data.plot_corr_matrix(df)

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

### GATHER THE FEATURES IN AN ARRAY ###
feature_array = []
count = 0
for col in df.columns:
    if use_feature[col]:
        feature_array.append(count)
    count += 1

### SELECT WHICH ANALYSIS TO RUN ###
plot_pca = True
plot_logistic = False
show = False

train_amount = 0.8
grid_search = False

motor_rescue_predict = False
motor_rescue_feature_importance = False
motor_rescue_detec_osc = False


if motor_rescue_feature_importance:
    run_neural_net.feature_importance_selected(
        df, feature_array, train_amount, seeds=10, show=True
    )

if motor_rescue_predict:
    run_neural_net.predict_motor_rescue_archive(
        df,
        feature_array,
        train_amount,
        motor_rescue_dfs,
        post_jaws_dfs,
        post_npas_dfs,
        jaws_post_times,
        npas_post_times,
        is_medial,
        show=show,
    )
    sys.exit()
    plot_data.plot_features_probabilities(combined_naive_df, combined_dd_df)
    plot_data.plot_feature_percent_correct(combined_naive_df, combined_dd_df, bins=20)

if motor_rescue_detec_osc:
    run_neural_net.plot_osc_over_time(
        feature_array,
        motor_rescue_dfs,
        post_jaws_dfs,
        post_npas_dfs,
        motor_rescue_dfs_osc,
        jaws_post_times,
        post_jaws_dfs_osc,
        npas_post_times,
        post_npas_dfs_osc,
        is_medial,
        show=show,
    )

if plot_logistic:
    run_logistic_regression.logistic_regression_on_df(
        df,
        feature_array,
        train_amount,
        motor_rescue_dfs,
        post_jaws_dfs,
        post_npas_dfs,
        jaws_post_times,
        npas_post_times,
        is_medial,
        show=show,
    )

if plot_pca:
    run_pca.pca_on_df(
        df,
        feature_array,
        motor_rescue_dfs,
        post_jaws_dfs,
        post_npas_dfs,
        xlabels,
        chunked,
        n_components_to_display=3,
        show=show,
    )


### OLD CODE

"""export_data = np.zeros(((len(time_chunks) + 1) * 2, 12 * 3))
col_labels = []
count = 0
for col in df.columns:
    if col != "Type" and col != "mouse" and col != "folder" and col != "name":
        ### EXPORT MEANS ####
        col_labels.append(f"{col} Mean")
        export_data[0, count * 3] = np.mean(motor_rescue_dfs[0][col])
        export_data[1 : export_data.shape[0] // 2, count * 3] = [
            np.mean(x[col]) for x in post_jaws_dfs
        ]
        export_data[export_data.shape[0] // 2, count * 3] = np.mean(
            motor_rescue_dfs[2][col]
        )
        export_data[export_data.shape[0] // 2 + 1 :, count * 3] = [
            np.mean(x[col]) for x in post_npas_dfs
        ]
        ### EXPORT SEM ###
        col_labels.append(f"{col} SEM")
        export_data[0, count * 3 + 1] = stats.sem(motor_rescue_dfs[0][col])
        export_data[1 : export_data.shape[0] // 2, count * 3 + 1] = [
            stats.sem(x[col]) for x in post_jaws_dfs
        ]
        export_data[export_data.shape[0] // 2, count * 3 + 1] = stats.sem(
            motor_rescue_dfs[2][col]
        )
        export_data[export_data.shape[0] // 2 + 1 :, count * 3 + 1] = [
            stats.sem(x[col]) for x in post_npas_dfs
        ]
        ### EXPORT STD ###
        col_labels.append(f"{col} STD")
        export_data[0, count * 3 + 2] = np.std(motor_rescue_dfs[0][col])
        export_data[1 : export_data.shape[0] // 2, count * 3 + 2] = [
            np.std(x[col]) for x in post_jaws_dfs
        ]
        export_data[export_data.shape[0] // 2, count * 3 + 2] = np.std(
            motor_rescue_dfs[2][col]
        )
        export_data[export_data.shape[0] // 2 + 1 :, count * 3 + 2] = [
            np.std(x[col]) for x in post_npas_dfs
        ]
        count += 1

export_df = pd.DataFrame(
    data=export_data,
    columns=col_labels,
    index=[
        "Pre-Stim JAWS",
        "Post 30 JAWS",
        "Post 60 JAWS",
        "Post 90-120 JAWS",
        "Post 150-210 JAWS",
        "Pre-Stim Npas",
        "Post 30 Npas",
        "Post 60 Npas",
        "Post 90-120 Npas",
        "Post 150-210 Npas",
    ],
)
n_counts = [len(motor_rescue_dfs[0])]
n_counts.extend([len(post_jaws_dfs[i]) for i in range(len(post_jaws_dfs))])
n_counts.extend([len(motor_rescue_dfs[2])])
n_counts.extend([len(post_npas_dfs[i]) for i in range(len(post_npas_dfs))])
export_df["n"] = n_counts

export_df.to_csv("../data/features_over_time.csv")

export_data = np.zeros(((len(time_chunks) + 1) * 2, 12 * 3))
col_labels = []
count = 0
for col in df.columns:
    if col != "Type" and col != "mouse" and col != "folder" and col != "name":
        ### EXPORT MEANS ####
        col_labels.append(f"{col} Mean")
        export_data[0, count * 3] = np.mean(motor_rescue_dfs[0][col])
        export_data[1 : export_data.shape[0] // 2, count * 3] = [
            np.mean(x[col]) for x in post_jaws_dfs
        ]
        export_data[export_data.shape[0] // 2, count * 3] = np.mean(
            motor_rescue_dfs[2][col]
        )
        export_data[export_data.shape[0] // 2 + 1 :, count * 3] = [
            np.mean(x[col]) for x in post_npas_dfs
        ]
        ### EXPORT SEM ###
        col_labels.append(f"{col} SEM")
        export_data[0, count * 3 + 1] = stats.sem(motor_rescue_dfs[0][col])
        export_data[1 : export_data.shape[0] // 2, count * 3 + 1] = [
            stats.sem(x[col]) for x in post_jaws_dfs
        ]
        export_data[export_data.shape[0] // 2, count * 3 + 1] = stats.sem(
            motor_rescue_dfs[2][col]
        )
        export_data[export_data.shape[0] // 2 + 1 :, count * 3 + 1] = [
            stats.sem(x[col]) for x in post_npas_dfs
        ]
        ### EXPORT STD ###
        col_labels.append(f"{col} STD")
        export_data[0, count * 3 + 2] = np.std(motor_rescue_dfs[0][col])
        export_data[1 : export_data.shape[0] // 2, count * 3 + 2] = [
            np.std(x[col]) for x in post_jaws_dfs
        ]
        export_data[export_data.shape[0] // 2, count * 3 + 2] = np.std(
            motor_rescue_dfs[2][col]
        )
        export_data[export_data.shape[0] // 2 + 1 :, count * 3 + 2] = [
            np.std(x[col]) for x in post_npas_dfs
        ]
        count += 1

export_df = pd.DataFrame(
    data=export_data,
    columns=col_labels,
    index=[
        "Pre-Stim JAWS",
        "Post 30 JAWS",
        "Post 60 JAWS",
        "Post 90-120 JAWS",
        "Post 150-210 JAWS",
        "Pre-Stim Npas",
        "Post 30 Npas",
        "Post 60 Npas",
        "Post 90-120 Npas",
        "Post 150-210 Npas",
    ],
)
n_counts = [len(motor_rescue_dfs[0])]
n_counts.extend([len(post_jaws_dfs[i]) for i in range(len(post_jaws_dfs))])
n_counts.extend([len(motor_rescue_dfs[2])])
n_counts.extend([len(post_npas_dfs[i]) for i in range(len(post_npas_dfs))])
export_df["n"] = n_counts

export_df.to_csv("../data/features_over_time.csv")

# export_data[export_data.shape[0]//2,count*3+1] = np.mean(motor_rescue_dfs[0][col])

"""


"""
jaws_pre_df["Medial"] = is_medial[0]
npas_pre_df["Medial"] = is_medial[2]

jaws_pre_df["Distance Healthy Score"] = np.loadtxt(
    f"../data/pca/jaws_pre_distance_healthy_score.txt"
)
jaws_pre_df["Centroid Healthy Score"] = np.loadtxt(
    f"../data/pca/jaws_pre_centroid_healthy_score.txt"
)
jaws_pre_df["Percentile Healthy Score"] = np.loadtxt(
    f"../data/pca/jaws_pre_percentile_healthy_score.txt"
)

jaws_pre_df["Naive Probability"] = np.loadtxt(
    f"../data/neural_net/pre_jaws_probabilities.txt", usecols=1
)

jaws_pre_df["DD Probability"] = np.loadtxt(
    f"../data/neural_net/pre_jaws_probabilities.txt", usecols=0
)

npas_pre_df["Distance Healthy Score"] = np.loadtxt(
    f"../data/pca/npas_pre_distance_healthy_score.txt"
)
npas_pre_df["Centroid Healthy Score"] = np.loadtxt(
    f"../data/pca/npas_pre_centroid_healthy_score.txt"
)
npas_pre_df["Percentile Healthy Score"] = np.loadtxt(
    f"../data/pca/npas_pre_percentile_healthy_score.txt"
)

npas_pre_df["Naive Probability"] = np.loadtxt(
    f"../data/neural_net/pre_npas_probabilities.txt", usecols=1
)

npas_pre_df["DD Probability"] = np.loadtxt(
    f"../data/neural_net/pre_npas_probabilities.txt", usecols=0
)

fig, ax = plt.subplots(4, 3, figsize=(12, 10), dpi=300, tight_layout=True)
axes = [ax[i, j] for i in range(4) for j in range(3)]
count = 0
for col in npas_pre_df.columns[:13]:
    if col != "Delta Osc":
        sns.regplot(
            data=npas_pre_df,
            x="Naive Probability",
            y=col,
            color="red",
            ax=axes[count],
            scatter_kws={"s": 8},
        )
        sns.regplot(
            data=npas_pre_df,
            x="DD Probability",
            y=col,
            color="blue",
            ax=axes[count],
            scatter_kws={"s": 8},
        )
        axes[count].set_xlabel("Probability")
        count += 1
makeNice(axes)
fig.savefig(f"../data/feature_probabilities_plot.pdf", bbox_inches="tight")
plt.close()
run_cmd("open ../data/feature_probabilities_plot.pdf")
"""

sys.exit()
jaws_pre_df.to_csv("../data/jaws_pre_time.csv")
npas_pre_df.to_csv("../data/npas_pre_time.csv")
time_chunk_labels = [30, 60, "90_120", "150_180_210"]
count = 0
for pjdf in post_jaws_dfs:
    """pjdf["Distance Healthy Score"] = np.loadtxt(
        f"../data/pca/jaws_post_{time_chunk_labels[count]}_distance_healthy_score.txt"
    )
    pjdf["Centroid Healthy Score"] = np.loadtxt(
        f"../data/pca/jaws_post_{time_chunk_labels[count]}_centroid_healthy_score.txt"
    )
    pjdf["Percentile Healthy Score"] = np.loadtxt(
        f"../data/pca/jaws_post_{time_chunk_labels[count]}_percentile_healthy_score.txt"
    )"""
    pjdf["Naive Probability"] = np.loadtxt(
        f"../data/neural_net/post_{time_chunk_labels[count]}_jaws_probabilities.txt",
        usecols=1,
    )

    pjdf["DD Probability"] = np.loadtxt(
        f"../data/neural_net/post_{time_chunk_labels[count]}_jaws_probabilities.txt",
        usecols=0,
    )
    pjdf.to_csv(f"../data/jaws_post_time_{time_chunk_labels[count]}.csv")
    count += 1
count = 0
for pjdf in post_npas_dfs:
    """pjdf["Distance Healthy Score"] = np.loadtxt(
        f"../data/pca/npas_post_{time_chunk_labels[count]}_distance_healthy_score.txt"
    )
    pjdf["Centroid Healthy Score"] = np.loadtxt(
        f"../data/pca/npas_post_{time_chunk_labels[count]}_centroid_healthy_score.txt"
    )
    pjdf["Percentile Healthy Score"] = np.loadtxt(
        f"../data/pca/npas_post_{time_chunk_labels[count]}_percentile_healthy_score.txt"
    )"""
    pjdf["Naive Probability"] = np.loadtxt(
        f"../data/neural_net/post_{time_chunk_labels[count]}_npas_probabilities.txt",
        usecols=1,
    )

    pjdf["DD Probability"] = np.loadtxt(
        f"../data/neural_net/post_{time_chunk_labels[count]}_npas_probabilities.txt",
        usecols=0,
    )
    pjdf.to_csv(f"../data/npas_post_time_{time_chunk_labels[count]}.csv")
    count += 1
