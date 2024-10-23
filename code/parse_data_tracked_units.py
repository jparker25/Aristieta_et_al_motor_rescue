import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os, sys
import seaborn as sns
import json
from scipy import stats
import matplotlib.patches as patches
from PIL import Image
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# user modules
import poisson_surprise
from helpers import *
import parse_data

# global variables


def get_ephys_tracked_data(source_path, save_path, length):
    run_cmd(f"mkdir -p {save_path}")
    all_spikes = []
    light_on_off = []
    is_medial = []
    mouse = []
    cell_name = []
    folder = []
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
            collect_spikes = []
            for i in range(6):
                collect_spikes.append(
                    spikes[
                        (light_on[0, 0] - (5 - i) * 30 > spikes)
                        & (spikes >= light_on[0, 0] - (6 - i) * length)
                    ]
                )

            """collect_spikes.append(
                spikes[(light_on[0, 0] > spikes) & (spikes >= light_on[0, 0] - length)]
            )"""
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
    rates = []
    for cell_spikes in spikes:
        cell_rates = parse_data.calc_firing_rates(
            cell_spikes, np.ones(len(cell_spikes)) * length
        )
        rates.append(cell_rates)
    return np.asarray(rates)


def calc_cv_tracked(spikes):
    cv = []
    for cell_spikes in spikes:
        cell_cvs = parse_data.calc_cv(cell_spikes)
        cv.append(cell_cvs)
    return np.asarray(cv)


def calc_burst_statistics_tracked(
    spikes, rates, length, min_spikes=3, surprise_threshold=3, window=0.25
):
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


def create_ephys_tracked_units(fixed_length):
    save_dir = "../data/tracked_units"
    jaws_path = (
        "/Users/johnparker/UPitt_Data/SNr_motor_rescue_project/jaws_pre_processed"
    )
    jaws_spikes, jaws_light, jaws_is_medial, jaws_mouse, jaws_folder, jaws_cell_name = (
        get_ephys_tracked_data(
            jaws_path, f"{save_dir}/jaws_neurons/tracked_units", fixed_length
        )
    )

    npas_path = (
        "/Users/johnparker/UPitt_Data/SNr_motor_rescue_project/npas_pre_processed"
    )
    npas_spikes, npas_light, npas_is_medial, npas_mouse, npas_folder, npas_cell_name = (
        get_ephys_tracked_data(
            npas_path, f"{save_dir}/npas_neurons/tracked_units", fixed_length
        )
    )

    save_spikes_light_ephys_tracked_data(
        jaws_spikes, jaws_light, f"{save_dir}/jaws_neurons/tracked_units"
    )
    save_spikes_light_ephys_tracked_data(
        npas_spikes, npas_light, f"{save_dir}/npas_neurons/tracked_units"
    )

    jaws_rates = calc_firing_rates_tracked(jaws_spikes, fixed_length)
    npas_rates = calc_firing_rates_tracked(npas_spikes, fixed_length)

    x_values = np.arange(jaws_rates.shape[1])
    treatment_times = np.arange(6, 76, 7)
    x_values = [i for i in x_values if i not in treatment_times]

    save_rates_ephys_tracked_data(jaws_rates, f"{save_dir}/jaws_neurons/tracked_units")
    save_rates_ephys_tracked_data(npas_rates, f"{save_dir}/npas_neurons/tracked_units")

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

    run_cmd(
        '/Applications/MATLAB_R2021b.app/bin/matlab -nojvm -nodesktop -batch "get_osc_data_ephys_tracked"'
    )
    all_jaws_data = np.zeros((jaws_rates.shape[1], 12, jaws_rates.shape[0]))
    print(all_jaws_data.shape)
    for i in range(all_jaws_data.shape[2]):
        all_jaws_data[:, 0, i] = jaws_rates[i, :]
        all_jaws_data[:, 1, i] = jaws_cvs[i, :]
        osc = np.loadtxt(
            f"{save_dir}/jaws_neurons/tracked_units/cell_{i+1:04d}/osc_data.txt",
            delimiter=",",
        )
        all_jaws_data[:, 2, i] = osc[:, 1]
        all_jaws_data[:, 3, i] = osc[:, 4]
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
            f"{save_dir}/npas_neurons/tracked_units/cell_{i+1:04d}/osc_data.txt",
            delimiter=",",
        )
        all_npas_data[:, 2, i] = osc[:, 1]
        all_npas_data[:, 3, i] = osc[:, 4]
        count = 4
        for stat in [0, 1, 2, 3, 4, 5, 8, 9]:
            scale = 30 if stat == 0 else 1
            all_npas_data[:, count, i] = npas_burst_stats[i, :, stat] / scale
            count += 1

    treat = 1
    treatment_index = []
    for i in range(jaws_rates.shape[1]):
        if i not in treatment_times:
            treatment_index.append(i * 30 - 180)
        else:
            treatment_index.append(f"Treatment {treat}")
            treat += 1
    treatment_index = np.asarray(treatment_index)

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

    for k in range(all_npas_data.shape[2]):
        npas_export_df = pd.DataFrame(
            data=all_npas_data[:, :, k], columns=col_labels, index=treatment_index
        )
        npas_export_df["DD Probability"] = np.loadtxt(
            f"../data/tracked_units/npas_treatment_data/npas_treatment_cell_{k+1}_probabilities.txt",
            usecols=0,
        )
        npas_export_df["Naive Probability"] = np.loadtxt(
            f"../data/tracked_units/npas_treatment_data/npas_treatment_cell_{k+1}_probabilities.txt",
            usecols=1,
        )
        npas_export_df["mouse"] = npas_mouse[k]
        npas_export_df["medial"] = npas_is_medial[k]
        npas_export_df["folder"] = npas_folder[k]
        npas_export_df["name"] = npas_cell_name[k]
        npas_export_df.to_csv(
            f"../data/tracked_units/npas_treatment_data/npas_treatment_cell_{k+1}.csv"
        )

    for k in range(all_jaws_data.shape[2]):
        jaws_export_df = pd.DataFrame(
            data=all_jaws_data[:, :, k], columns=col_labels, index=treatment_index
        )
        jaws_export_df["DD Probability"] = np.loadtxt(
            f"../data/tracked_units/jaws_treatment_data/jaws_treatment_cell_{k+1}_probabilities.txt",
            usecols=0,
        )
        jaws_export_df["Naive Probability"] = np.loadtxt(
            f"../data/tracked_units/jaws_treatment_data/jaws_treatment_cell_{k+1}_probabilities.txt",
            usecols=1,
        )

        jaws_export_df["mouse"] = jaws_mouse[k]
        jaws_export_df["medial"] = jaws_is_medial[k]
        jaws_export_df["folder"] = jaws_folder[k]
        jaws_export_df["name"] = jaws_cell_name[k]
        jaws_export_df.to_csv(
            f"../data/tracked_units/jaws_treatment_data/jaws_treatment_cell_{k+1}.csv"
        )

    sys.exit()

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
    jaws_export_df.to_csv("data/tracked_units/jaws_treatment_data.csv")

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
    npas_export_df.to_csv("data/tracked_units/npas_treatment_data.csv")

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
        """axes[axis_count].plot(
            x_values,
            np.mean(jaws_burst_stats[:, x_values, stat], axis=0),
            marker="o",
            color="b",
            ls="dashed",
            lw=0.5,
            markersize=2,
            alpha=0.25,
        )
        axes[axis_count].plot(
            x_values,
            np.mean(npas_burst_stats[:, x_values, stat], axis=0),
            marker="o",
            color="r",
            ls="dashed",
            lw=0.5,
            markersize=2,
            alpha=0.25,
        )"""
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
    fig.savefig("data/tracked_units_features.pdf", bbox_inches="tight")
    plt.close()
    run_cmd("open data/tracked_units_features.pdf ")


def plot_tracked_healthy_dd(jaws_path, npas_path, jaws_units=23, npas_units=10):
    jaws_data = []
    for i in range(jaws_units):
        jaws_data.append(pd.read_csv(f"{jaws_path}/jaws_treatment_cell_{i+1}.csv"))
    jaws_dd = np.zeros(76)
    jaws_dd_prob = np.zeros(76)
    for i in range(len(jaws_dd)):
        for data in jaws_data:
            jaws_dd_prob[i] += (
                data.iloc[i, data.columns.get_loc("DD Probability")] / jaws_units
            )
            if data.iloc[i, data.columns.get_loc("DD Probability")] > 0.5:
                jaws_dd[i] += 1 / jaws_units
    npas_data = []
    for i in range(npas_units):
        npas_data.append(pd.read_csv(f"{npas_path}/npas_treatment_cell_{i+1}.csv"))
    npas_dd = np.zeros(76)
    npas_dd_prob = np.zeros(76)
    for i in range(len(npas_dd)):
        for data in npas_data:
            npas_dd_prob[i] += (
                data.iloc[i, data.columns.get_loc("DD Probability")] / npas_units
            )
            if data.iloc[i, data.columns.get_loc("DD Probability")] > 0.5:
                npas_dd[i] += 1 / npas_units
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
    fig.savefig(f"data/tracked_units/dd_probabilities.pdf", bbox_inches="tight")
    plt.close()
    run_cmd("open data/tracked_units/dd_probabilities.pdf")


training_fixed_length = 30
"""plot_tracked_healthy_dd(
    "data/tracked_units/jaws_treatment_data", "data/tracked_units/npas_treatment_data"
)"""
create_ephys_tracked_units(training_fixed_length)
