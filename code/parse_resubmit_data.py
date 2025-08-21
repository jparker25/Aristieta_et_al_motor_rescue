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
npas_unilateral = (
    "/Users/johnparker/UPitt_Data/SNr_motor_rescue_project/npas_cre_unidd_pre_processed"
)
pv_bilateral = "/Users/johnparker/UPitt_Data/SNr_motor_rescue_project/pv_bilateral_dd_pre_processed"

# length of spike trains to analyze
fixed_length = 30


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


def get_baseline_data(fixed_length, path):
    """
    Get baseline data.

    Parameters
    ----------
    \t fixed_length (int) : length of spike trains to analyze

    \t path (str) : path to data

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


def get_post_data(fixed_length, path):
    """
    Get post opto data.

    Parameters
    ----------
    \t fixed_length (int) : length of spike trains to analyze

    \t path (str) : path to data

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
    pre_post = find_pre_post_matches(source_path)

    # Save pre/post matches
    save_stat(pre_post, f"{save_path}/pre_post_units.txt")

    # Get baseline ephys data
    (
        pre_spikes,
        pre_medial,
        pre_mouse,
        pre_folder,
        pre_cell_name,
    ) = get_baseline_data(fixed_length, source_path)

    # Get post opto ephys data
    (
        post_spikes,
        post_times,
        post_medial,
        post_mouse,
        post_folder,
        post_cell_name,
    ) = get_post_data(fixed_length, source_path)

    # Calculate firing rates for jaws pre/post data
    pre_spikes_frs = calc_firing_rates(
        pre_spikes, np.ones(len(pre_spikes)) * fixed_length
    )

    post_spikes_frs = calc_firing_rates(
        post_spikes, np.ones(len(post_spikes)) * fixed_length
    )

    # Calculate coefficient of variation for jaws pre/post data
    pre_spikes_cvs = calc_cv(pre_spikes)

    post_spikes_cvs = calc_cv(post_spikes)

    # Calculate burst statistics for jaws pre/post data
    pre_burst_statistics = calc_burst_statistics(
        pre_spikes,
        pre_spikes_frs,
        np.ones(len(pre_spikes)) * fixed_length,
        min_spikes=3,
        surprise_threshold=3,
        window=2,
    )

    post_burst_statistics = calc_burst_statistics(
        post_spikes,
        post_spikes_frs,
        np.ones(len(post_spikes)) * fixed_length,
        min_spikes=3,
        surprise_threshold=3,
        window=2,
    )

    # Save data
    save_motor_rescue_spikes(pre_spikes, f"{save_path}/pre_opto")
    save_motor_rescue_spikes(post_spikes, f"{save_path}/post_opto")

    save_stat(post_times, f"{save_path}/post_opto/cell_post_times.txt")

    save_stat(pre_medial, f"{save_path}/pre_opto/is_medial.txt")
    save_stat(post_medial, f"{save_path}/post_opto/is_medial.txt")

    save_stat(
        np.ones(len(pre_spikes)) * fixed_length,
        f"{save_path}/pre_opto/cell_lengths.txt",
    )
    save_stat(pre_spikes_frs, f"{save_path}/pre_opto/cell_frs.txt")
    save_stat(pre_spikes_cvs, f"{save_path}/pre_opto/cell_cvs.txt")
    save_stat(
        pre_burst_statistics,
        f"{save_path}/pre_opto/cell_burst_statistics.txt",
    )

    save_stat(
        np.ones(len(post_spikes)) * fixed_length,
        f"{save_path}/post_opto/cell_lengths.txt",
    )
    save_stat(post_spikes_frs, f"{save_path}/post_opto/cell_frs.txt")
    save_stat(post_spikes_cvs, f"{save_path}/post_opto/cell_cvs.txt")
    save_stat(
        post_burst_statistics,
        f"{save_path}/post_opto/cell_burst_statistics.txt",
    )

    save_meta_list(pre_mouse, f"{save_path}/pre_opto/cell_mouse.txt")
    save_meta_list(pre_folder, f"{save_path}/pre_opto/cell_folders.txt")
    save_meta_list(pre_cell_name, f"{save_path}/pre_opto/cell_names.txt")

    save_meta_list(post_mouse, f"{save_path}/post_opto/cell_mouse.txt")
    save_meta_list(post_folder, f"{save_path}/post_opto/cell_folders.txt")
    save_meta_list(post_cell_name, f"{save_path}/post_opto/cell_names.txt")


def create_ephys_training_data(fixed_length):
    """
    Create ephys training data.

    Parameters
    ----------

    \t fixed_length (int) : length of spike trains to analyze
    """

    # Create directories
    save_dir = "../resub_data"

    get_ephys_data(npas_unilateral, f"{save_dir}/npas_uni_dd_neurons", fixed_length)
    get_ephys_data(pv_bilateral, f"{save_dir}/pv_bilateral_dd_neurons", fixed_length)

    run_cmd(
        "/Applications/MATLAB_R2021b.app/bin/matlab -nojvm -nodesktop -batch 'get_osc_data_resubmission'"
    )


# Create data and generate statistics for further analysis

create_ephys_training_data(fixed_length)
