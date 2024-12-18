"""
parse_data.py

Gathers and processes data for training and testing of the motor rescue model.

Author: John E. Parker (2024)
"""

import numpy as np
import json

# user modules
import poisson_surprise
from helpers import *

#### GLOBAL VARIABLES ####

# Paths to pre-processed data
jaws_path = "/Users/johnparker/UPitt_Data/SNr_motor_rescue_project/jaws_pre_processed"
npas_path = "/Users/johnparker/UPitt_Data/SNr_motor_rescue_project/npas_pre_processed"
cont_gpe_data = (
    "/Users/johnparker/UPitt_Data/continuous_light/gpe_pv_hsyn/pre_processed_data"
)
cont_d1_data = (
    "/Users/johnparker/UPitt_Data/continuous_light/d1_msns/pre_processed_data"
)

pulse_gpe_data = (
    "/Users/johnparker/UPitt_Data/pulsed_light_terminal_20Hz/gpe_pv/pre_processed_data"
)
pulse_d1_data = (
    "/Users/johnparker/UPitt_Data/pulsed_light_terminal_20Hz/d1_msns/pre_processed_data"
)

dd_cont_gpe_data = (
    "/Users/johnparker/UPitt_Data/continuous_light/gpe_pv_hsyn/pre_processed_data"
)
dd_cont_d1_data = (
    "/Users/johnparker/UPitt_Data/continuous_light/d1_msns/pre_processed_data"
)

dd_pulse_gpe_data = (
    "/Users/johnparker/UPitt_Data/pulsed_light_terminal_20Hz/gpe_pv/pre_processed_data"
)
dd_pulse_d1_data = (
    "/Users/johnparker/UPitt_Data/pulsed_light_terminal_20Hz/d1_msns/pre_processed_data"
)

# length of spike trains to analyze
training_fixed_length = 30


def read_in_meta_data(file_path):
    """
    Read in meta data from a file.

    Parameters
    ----------
    \t file_path (str) : path read in meta data

    Returns
    -------
    \t meta_data (dict) : dictionary of meta data
    """
    meta_data = {}
    with open(file_path, "r") as file:
        for line in file.readlines():
            str_split = line.split(":\t")
            meta_data[str_split[0]] = str_split[1][0:-1]
    return meta_data


def get_naive_baseline_data(length, *paths):
    """
    Get baseline spike trains for naive neurons.

    Parameters
    ----------
    \t length (int) : length of baseline to analyze

    \t paths (list) : list of paths to naive neuron data

    Returns
    -------
    \t baseline_spike_trains (list) : list of baseline spike trains

    \t baseline_lengths (list) : list of lengths of baseline spike trains

    \t mouse (list) : list of mouse names

    \t folder (list) : list of folder names

    \t name (list) : list of channel names
    """

    # Initialize lists
    baseline_spike_trains = []
    baseline_lengths = []
    mouse = []
    folder = []
    name = []

    # Loop through paths and grab spike train data
    for path in paths:
        for dir in os.listdir(path):
            if "Naive" in dir and not "hsyn" in dir:
                for subdir in sorted(os.listdir(f"{path}/{dir}")):
                    if "Neuron" in subdir:
                        meta_data = read_in_meta_data(
                            f"{path}/{dir}/{subdir}/meta_data.txt"
                        )
                        mouse.append(meta_data["mouse"])
                        folder.append(meta_data["src"])
                        name.append(meta_data["channel"])
                        spikes = np.loadtxt(f"{path}/{dir}/{subdir}/spikes.txt")
                        light_on = np.loadtxt(f"{path}/{dir}/{subdir}/light_on.txt")
                        if length > 0 and light_on[0] >= length:
                            baseline_spike_trains.append(
                                spikes[
                                    (spikes < light_on[0])
                                    & (spikes > light_on[0] - length)
                                ]
                                - (light_on[0] - length)
                            )
                        else:
                            baseline_spike_trains.append(spikes[spikes < light_on[0]])
                        baseline_lengths.append(
                            light_on[0] if light_on[0] < length else length
                        )

    return baseline_spike_trains, baseline_lengths, mouse, folder, name


def get_dd_baseline_data(length, *paths):
    """
    Get baseline spike trains for DD neurons.

    Parameters
    ----------
    \t length (int) : length of baseline to analyze

    \t paths (list) : list of paths to DD neuron data

    Returns
    -------
    \t baseline_spike_trains (list) : list of baseline spike trains

    \t baseline_lengths (list) : list of lengths of baseline spike trains

    \t mouse (list) : list of mouse names

    \t folder (list) : list of folder names

    \t name (list) : list of channel names
    """

    # Initialize lists
    baseline_spike_trains = []
    baseline_lengths = []
    mouse = []
    folder = []
    name = []

    # Loop through paths and grab spike train data
    for path in paths:
        for dir in os.listdir(path):
            if "6-OHDA" in dir and not "hsyn" in dir:
                for subdir in sorted(os.listdir(f"{path}/{dir}")):
                    if "Neuron" in subdir:
                        meta_data = read_in_meta_data(
                            f"{path}/{dir}/{subdir}/meta_data.txt"
                        )
                        mouse.append(meta_data["mouse"])
                        folder.append(meta_data["src"])
                        name.append(meta_data["channel"])
                        spikes = np.loadtxt(f"{path}/{dir}/{subdir}/spikes.txt")
                        light_on = np.loadtxt(f"{path}/{dir}/{subdir}/light_on.txt")
                        if length > 0 and light_on[0] >= length:
                            baseline_spike_trains.append(
                                spikes[
                                    (spikes < light_on[0])
                                    & (spikes > light_on[0] - length)
                                ]
                                - (light_on[0] - length)
                            )
                        else:
                            baseline_spike_trains.append(spikes[spikes < light_on[0]])
                        baseline_lengths.append(
                            light_on[0] if light_on[0] < length else length
                        )
    return baseline_spike_trains, baseline_lengths, mouse, folder, name


def save_naive_baseline_spikes(spikes):
    """
    Saves naive baseline spikes to file.

    Parameters
    ----------
    \t spikes (list) : list of naive baseline spikes
    """
    run_cmd("mkdir -p ../data/training/naive_neurons/spikes")
    for i in range(len(spikes)):
        np.savetxt(
            f"../data/training/naive_neurons/spikes/cell_{i:04d}_baseline_spikes.txt",
            spikes[i],
            fmt="%f",
            newline="\n",
            delimiter="\t",
        )


def save_dd_baseline_spikes(spikes):
    """
    Saves DD baseline spikes to file.

    Parameters
    ----------
    \t spikes (list) : list of DD baseline spikes
    """
    run_cmd("mkdir -p ../data/training/dd_neurons/spikes")
    for i in range(len(spikes)):
        np.savetxt(
            f"../data/training/dd_neurons/spikes/cell_{i:04d}_baseline_spikes.txt",
            spikes[i],
            fmt="%f",
            newline="\n",
            delimiter="\t",
        )


def calc_firing_rates(spikes, lengths):
    """
    Calculate firing rates for a list of spike trains.

    Parameters
    ----------
    \t spikes (list) : list of spike trains

    \t lengths (list) : list of lengths of spike trains

    Returns
    -------
    \t firing_rates (np.array) : array of firing rates
    """
    return np.asarray([spikes[i].shape[0] / lengths[i] for i in range(len(spikes))])


def calc_cv(spikes):
    """
    Calculate coefficient of variation for a list of spike trains.

    Parameters
    ----------
    \t spikes (list) : list of spike trains

    Returns
    -------
    \t cvs (np.array) : array of coefficients of variation
    """
    cvs = np.zeros(len(spikes))
    for i in range(len(spikes)):
        cv_spikes_diff = np.diff(spikes[i])
        cvs[i] = (
            np.std(cv_spikes_diff) / np.mean(cv_spikes_diff)
            if len(cv_spikes_diff) > 3
            else 0
        )
    return cvs


def calc_burst_statistics(
    spikes, rates, lengths, min_spikes=3, surprise_threshold=3, window=0.25
):
    """
    Calculate burst statistics for a list of spike trains.

    Parameters
    ----------
    \t spikes (list) : list of spike trains

    \t rates (np.array) : array of firing rates

    \t lengths (list) : list of lengths of spike trains

    \t min_spikes=3 (int) : minimum number of spikes for a burst

    \t surprise_threshold=3 (int) : surprise threshold for a burst

    \t window=0.25 (float) : window for a burst

    Returns
    -------
    \t burst_statistics (np.array) : array of burst statistics
    """

    # initialize burst statistics array
    burst_statistics = np.zeros((len(spikes), 10))

    # iterate through spike trains and calculate burst statistics
    for i in range(len(spikes)):
        bursts = poisson_surprise.run_poisson_surprise(
            spikes[i],
            min_spikes=min_spikes,
            surprise_threshold=surprise_threshold,
            window=window,
        )
        burst_statistics[i, :] = (
            poisson_surprise.burst_statistics(spikes[i], bursts, lengths[i])
            if len(bursts) > 0
            else np.zeros(10)
        )
    return burst_statistics


def save_stat(stat_data, path):
    """
    Save stat_data to file.

    Parameters
    ----------
    \t stat_data (np.array) : array of data to save

    \t path (str) : path to save data
    """
    np.savetxt(path, stat_data, fmt="%f", newline="\n")


def get_baseline_ephys_data(fixed_length, path):
    """
    Get baseline ephys data.

    Parameters
    ----------
    \t fixed_length (int) : length of spike trains to analyze

    \t path (str) : path to ephys data

    Returns
    -------
    \t pre_spikes (list) : list of pre opto spike trains

    \t is_medial (list) : list of medial boolean values

    \t mouse (list) : list of mouse names

    \t folder (list) : list of folder names

    \t cell_name (list) : list of cell names
    """

    # Initialize lists
    pre_spikes = []
    is_medial = []
    mouse = []
    folder = []
    cell_name = []

    # Iterate through files and generate data
    for neuron in sorted(os.listdir(f"{path}/pre_opto")):
        if neuron != ".DS_Store":
            meta_file = open(f"{path}/pre_opto/{neuron}/meta_data.txt", "r")
            meta_data = json.load(meta_file)
            is_medial.append(meta_data["medial"])
            mouse.append(meta_data["mouse"])
            cell_name.append(meta_data["cell_name"])
            folder.append(meta_data["folder"])
            spikes = np.loadtxt(f"{path}/pre_opto/{neuron}/spikes.txt")
            spikes = spikes[spikes < fixed_length]
            pre_spikes.append(spikes)
    return pre_spikes, is_medial, mouse, folder, cell_name


def get_post_ephys_data(fixed_length, path):
    """
    Get post opto ephys data.

    Parameters
    ----------
    \t fixed_length (int) : length of spike trains to analyze

    \t path (str) : path to ephys data

    Returns
    -------
    \t post_spikes (list) : list of post opto spike trains

    \t post_times (list) : list of post opto times

    \t is_medial (list) : list of medial boolean values

    \t mouse (list) : list of mouse names

    \t folder (list) : list of folder names

    \t cell_name (list) : list of cell names
    """

    # initialize lists
    post_spikes = []
    post_times = []
    is_medial = []
    mouse = []
    folder = []
    cell_name = []

    # Iterate through files and generate data
    for neuron in sorted(os.listdir(f"{path}/post_opto")):
        if neuron != ".DS_Store":
            meta_file = open(f"{path}/post_opto/{neuron}/meta_data.txt", "r")
            meta_data = json.load(meta_file)
            post_times.append(meta_data["post_time"])
            is_medial.append(meta_data["medial"])
            mouse.append(meta_data["mouse"])
            cell_name.append(meta_data["cell_name"])
            folder.append(meta_data["folder"])
            spikes = np.loadtxt(f"{path}/post_opto/{neuron}/spikes.txt")
            spikes = spikes[spikes < fixed_length]
            post_spikes.append(spikes)
    return post_spikes, post_times, is_medial, mouse, folder, cell_name


def save_motor_rescue_spikes(spikes, direc):
    """
    Save motor rescue spikes to file.

    Parameters
    ----------
    \t spikes (list) : list of motor rescue spikes

    \t direc (str) : directory to save spikes
    """

    # Create directory if it doesn't exist
    run_cmd(f"mkdir -p {direc}/spikes")

    # Save spikes to file
    for i in range(len(spikes)):
        np.savetxt(
            f"{direc}/spikes/cell_{i:04d}.txt",
            spikes[i],
            fmt="%f",
            newline="\n",
            delimiter="\t",
        )


def find_pre_post_matches(path):
    """
    Find pre/post matches.

    Parameters
    ----------
    \t path (str) : path to ephys data

    Returns
    -------
    \t pre_post_units (np.array) : array of pre/post units
    """

    # Initialize list of matches
    matches = ["mouse", "folder", "cell_name", "type"]
    pre_post_units = []

    # Iterate through pre/post directories and find matches
    for pre_units in os.listdir(f"{path}/pre_opto"):
        if pre_units != ".DS_Store":
            pre_meta_file = open(f"{path}/pre_opto/{pre_units}/meta_data.txt", "r")
            pre_meta_data = json.load(pre_meta_file)
            for post_units in os.listdir(f"{path}/post_opto"):
                if post_units != ".DS_Store":
                    post_meta_file = open(
                        f"{path}/post_opto/{post_units}/meta_data.txt", "r"
                    )
                    post_meta_data = json.load(post_meta_file)
                    if (
                        np.sum(
                            [
                                pre_meta_data[match] == post_meta_data[match]
                                for match in matches
                            ]
                        )
                        == 4
                    ):

                        pre_post_units.append(
                            [
                                eval(pre_units.split("_")[1].lstrip("0")),
                                eval(post_units.split("_")[1].lstrip("0")),
                            ]
                        )
    return np.asarray(pre_post_units)


def save_meta_list(stat_list, path):
    """
    Save meta list to file.

    Parameters
    ----------
    \t stat_list (list) : list of meta data

    \t path (str) : path to save meta data
    """
    with open(path, "w") as f:
        for line in stat_list:
            f.write(f"{line}\n")


def get_ephys_data(source_path, save_path, fixed_length):
    """
    Get ephys data.

    Parameters
    ----------
    \t source_path (str) : path to ephys data

    \t save_path (str) : path to save ephys data

    \t fixed_length (int) : length of spike trains to analyze
    """

    # Create directories
    run_cmd(f"mkdir -p {save_path}")
    run_cmd(f"mkdir -p {save_path}/pre_opto")
    run_cmd(f"mkdir -p {save_path}/post_opto")

    # Find pre/post matches
    pre_post_jaws = find_pre_post_matches(source_path)

    # Save pre/post matches
    save_stat(pre_post_jaws, f"{save_path}/pre_post_units.txt")

    # Get baseline ephys data
    (
        jaws_pre_spikes,
        jaws_pre_medial,
        jaws_pre_mouse,
        jaws_pre_folder,
        jaws_pre_cell_name,
    ) = get_baseline_ephys_data(fixed_length, source_path)

    # Get post opto ephys data
    (
        jaws_post_spikes,
        jaws_post_times,
        jaws_post_medial,
        jaws_post_mouse,
        jaws_post_folder,
        jaws_post_cell_name,
    ) = get_post_ephys_data(fixed_length, source_path)

    # Calculate firing rates for jaws pre/post data
    jaws_pre_spikes_frs = calc_firing_rates(
        jaws_pre_spikes, np.ones(len(jaws_pre_spikes)) * fixed_length
    )

    jaws_post_spikes_frs = calc_firing_rates(
        jaws_post_spikes, np.ones(len(jaws_post_spikes)) * fixed_length
    )

    # Calculate coefficient of variation for jaws pre/post data
    jaws_pre_spikes_cvs = calc_cv(jaws_pre_spikes)

    jaws_post_spikes_cvs = calc_cv(jaws_post_spikes)

    # Calculate burst statistics for jaws pre/post data
    jaws_pre_burst_statistics = calc_burst_statistics(
        jaws_pre_spikes,
        jaws_pre_spikes_frs,
        np.ones(len(jaws_pre_spikes)) * fixed_length,
        min_spikes=3,
        surprise_threshold=3,
        window=2,
    )

    jaws_post_burst_statistics = calc_burst_statistics(
        jaws_post_spikes,
        jaws_post_spikes_frs,
        np.ones(len(jaws_post_spikes)) * fixed_length,
        min_spikes=3,
        surprise_threshold=3,
        window=2,
    )

    # Save data
    save_motor_rescue_spikes(jaws_pre_spikes, f"{save_path}/pre_opto")
    save_motor_rescue_spikes(jaws_post_spikes, f"{save_path}/post_opto")

    save_stat(jaws_post_times, f"{save_path}/post_opto/cell_post_times.txt")

    save_stat(jaws_pre_medial, f"{save_path}/pre_opto/is_medial.txt")
    save_stat(jaws_post_medial, f"{save_path}/post_opto/is_medial.txt")

    save_stat(
        np.ones(len(jaws_pre_spikes)) * fixed_length,
        f"{save_path}/pre_opto/cell_lengths.txt",
    )
    save_stat(jaws_pre_spikes_frs, f"{save_path}/pre_opto/cell_frs.txt")
    save_stat(jaws_pre_spikes_cvs, f"{save_path}/pre_opto/cell_cvs.txt")
    save_stat(
        jaws_pre_burst_statistics,
        f"{save_path}/pre_opto/cell_burst_statistics.txt",
    )

    save_stat(
        np.ones(len(jaws_post_spikes)) * fixed_length,
        f"{save_path}/post_opto/cell_lengths.txt",
    )
    save_stat(jaws_post_spikes_frs, f"{save_path}/post_opto/cell_frs.txt")
    save_stat(jaws_post_spikes_cvs, f"{save_path}/post_opto/cell_cvs.txt")
    save_stat(
        jaws_post_burst_statistics,
        f"{save_path}/post_opto/cell_burst_statistics.txt",
    )

    save_meta_list(jaws_pre_mouse, f"{save_path}/pre_opto/cell_mouse.txt")
    save_meta_list(jaws_pre_folder, f"{save_path}/pre_opto/cell_folders.txt")
    save_meta_list(jaws_pre_cell_name, f"{save_path}/pre_opto/cell_names.txt")

    save_meta_list(jaws_post_mouse, f"{save_path}/post_opto/cell_mouse.txt")
    save_meta_list(jaws_post_folder, f"{save_path}/post_opto/cell_folders.txt")
    save_meta_list(jaws_post_cell_name, f"{save_path}/post_opto/cell_names.txt")


def create_ephys_training_data(fixed_length):
    """
    Create ephys training data.

    Parameters
    ----------

    \t fixed_length (int) : length of spike trains to analyze
    """

    # Create directories
    save_dir = "../data/training"

    get_ephys_data(jaws_path, f"{save_dir}/jaws_neurons", fixed_length)
    get_ephys_data(npas_path, f"{save_dir}/npas_neurons", fixed_length)

    run_cmd(
        "/Applications/MATLAB_R2021b.app/bin/matlab -nojvm -nodesktop -batch 'get_osc_data_ephys_training'"
    )


def create_pulse_cont_training_data(fixed_length):
    """
    Create pulse/cont training data.

    Parameters
    ----------
    \t fixed_length (int) : length of spike trains to analyze
    """

    # Create directories
    save_dir = "../data/training"
    run_cmd(f"mkdir -p {save_dir}")
    run_cmd(f"mkdir -p {save_dir}/naive_neurons")
    run_cmd(f"mkdir -p {save_dir}/dd_neurons")

    # Check if pulse data exists and generate spike train data
    pulse = True
    if pulse:
        (
            naive_baseline_spikes,
            naive_baseline_lengths,
            naive_mouse,
            naive_folder,
            naive_name,
        ) = get_naive_baseline_data(
            fixed_length, cont_gpe_data, cont_d1_data, pulse_gpe_data, pulse_d1_data
        )

        dd_baseline_spikes, dd_baseline_lengths, dd_mouse, dd_folder, dd_name = (
            get_dd_baseline_data(
                fixed_length,
                dd_cont_gpe_data,
                dd_cont_d1_data,
                dd_pulse_gpe_data,
                dd_pulse_d1_data,
            )
        )
    else:
        (
            naive_baseline_spikes,
            naive_baseline_lengths,
            naive_mouse,
            naive_folder,
            naive_name,
        ) = get_naive_baseline_data(fixed_length, cont_gpe_data, cont_d1_data)

        dd_baseline_spikes, dd_baseline_lengths, dd_mouse, dd_folder, dd_name = (
            get_dd_baseline_data(
                fixed_length,
                dd_cont_gpe_data,
                dd_cont_d1_data,
            )
        )

    # Calculate firing rates, CVs, and burst statistics
    naive_frs = calc_firing_rates(naive_baseline_spikes, naive_baseline_lengths)
    dd_frs = calc_firing_rates(dd_baseline_spikes, dd_baseline_lengths)

    naive_cvs = calc_cv(naive_baseline_spikes)
    dd_cvs = calc_cv(dd_baseline_spikes)

    naive_burst_statistics = calc_burst_statistics(
        naive_baseline_spikes,
        naive_frs,
        naive_baseline_lengths,
        min_spikes=3,
        surprise_threshold=3,
        window=2,
    )

    dd_burst_statistics = calc_burst_statistics(
        dd_baseline_spikes,
        dd_frs,
        dd_baseline_lengths,
        min_spikes=3,
        surprise_threshold=3,
        window=2,
    )

    # Save data
    save_dd_baseline_spikes(dd_baseline_spikes)
    save_naive_baseline_spikes(naive_baseline_spikes)
    save_stat(
        naive_baseline_lengths,
        "../data/training/naive_neurons/cell_baseline_lengths.txt",
    )
    save_stat(
        dd_baseline_lengths, "../data/training/dd_neurons/cell_baseline_lengths.txt"
    )
    save_stat(naive_frs, "../data/training/naive_neurons/cell_baseline_frs.txt")
    save_stat(dd_frs, "../data/training/dd_neurons/cell_baseline_frs.txt")
    save_stat(naive_cvs, "../data/training/naive_neurons/cell_baseline_cvs.txt")
    save_stat(dd_cvs, "../data/training/dd_neurons/cell_baseline_cvs.txt")

    save_stat(
        naive_burst_statistics,
        "../data/training/naive_neurons/cell_baseline_burst_statistics.txt",
    )
    save_stat(
        dd_burst_statistics,
        "../data/training/dd_neurons/cell_baseline_burst_statistics.txt",
    )

    save_meta_list(naive_name, "../data/training/naive_neurons/cell_names.txt")
    save_meta_list(naive_folder, "../data/training/naive_neurons/cell_folders.txt")
    save_meta_list(naive_mouse, "../data/training/naive_neurons/cell_mouse.txt")

    save_meta_list(dd_name, "../data/training/dd_neurons/cell_names.txt")
    save_meta_list(dd_folder, "../data/training/dd_neurons/cell_folders.txt")
    save_meta_list(dd_mouse, "../data/training/dd_neurons/cell_mouse.txt")

    run_cmd(
        f"/Applications/MATLAB_R2021b.app/bin/matlab -nojvm -nodesktop -batch 'get_osc_data_pulse_cont'"
    )


# Create data and generate statistics for further analysis

create_ephys_training_data(training_fixed_length)
create_pulse_cont_training_data(training_fixed_length)
