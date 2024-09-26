import numpy as np
import json, sys

# user modules
import poisson_surprise
from helpers import *

#### GLOBAL VARIABLES ####
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

training_fixed_length = 30


def read_in_meta_data(file_path):
    meta_data = {}
    with open(file_path, "r") as file:
        for line in file.readlines():
            str_split = line.split(":\t")
            meta_data[str_split[0]] = str_split[1][0:-1]
    return meta_data


def get_naive_baseline_data(length, *paths):
    baseline_spike_trains = []
    baseline_lengths = []
    mouse = []
    folder = []
    name = []
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
    baseline_spike_trains = []
    baseline_lengths = []
    mouse = []
    folder = []
    name = []
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
    run_cmd("mkdir -p data/training/naive_neurons/spikes")
    for i in range(len(spikes)):
        np.savetxt(
            f"data/training/naive_neurons/spikes/cell_{i:04d}_baseline_spikes.txt",
            spikes[i],
            fmt="%f",
            newline="\n",
            delimiter="\t",
        )


def save_dd_baseline_spikes(spikes):
    run_cmd("mkdir -p data/training/dd_neurons/spikes")
    for i in range(len(spikes)):
        np.savetxt(
            f"data/training/dd_neurons/spikes/cell_{i:04d}_baseline_spikes.txt",
            spikes[i],
            fmt="%f",
            newline="\n",
            delimiter="\t",
        )


def calc_firing_rates(spikes, lengths):
    return np.asarray([spikes[i].shape[0] / lengths[i] for i in range(len(spikes))])


def calc_cv(spikes):
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
    burst_statistics = np.zeros((len(spikes), 10))
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
    np.savetxt(path, stat_data, fmt="%f", newline="\n")


def get_baseline_ephys_data(fixed_length, path):
    pre_spikes = []
    is_medial = []
    mouse = []
    folder = []
    cell_name = []
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
    post_spikes = []
    post_times = []
    is_medial = []
    mouse = []
    folder = []
    cell_name = []
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
    run_cmd(f"mkdir -p {direc}/spikes")
    for i in range(len(spikes)):
        np.savetxt(
            f"{direc}/spikes/cell_{i:04d}.txt",
            spikes[i],
            fmt="%f",
            newline="\n",
            delimiter="\t",
        )


def find_pre_post_matches(path):
    matches = ["mouse", "folder", "cell_name", "type"]
    pre_post_units = []
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
    with open(path, "w") as f:
        for line in stat_list:
            f.write(f"{line}\n")


def get_ephys_data(source_path, save_path, fixed_length):
    run_cmd(f"mkdir -p {save_path}")
    run_cmd(f"mkdir -p {save_path}/pre_opto")
    run_cmd(f"mkdir -p {save_path}/post_opto")
    pre_post_jaws = find_pre_post_matches(source_path)
    save_stat(pre_post_jaws, f"{save_path}/pre_post_units.txt")
    (
        jaws_pre_spikes,
        jaws_pre_medial,
        jaws_pre_mouse,
        jaws_pre_folder,
        jaws_pre_cell_name,
    ) = get_baseline_ephys_data(fixed_length, source_path)
    (
        jaws_post_spikes,
        jaws_post_times,
        jaws_post_medial,
        jaws_post_mouse,
        jaws_post_folder,
        jaws_post_cell_name,
    ) = get_post_ephys_data(fixed_length, source_path)

    jaws_pre_spikes_frs = calc_firing_rates(
        jaws_pre_spikes, np.ones(len(jaws_pre_spikes)) * fixed_length
    )

    jaws_post_spikes_frs = calc_firing_rates(
        jaws_post_spikes, np.ones(len(jaws_post_spikes)) * fixed_length
    )

    jaws_pre_spikes_cvs = calc_cv(jaws_pre_spikes)

    jaws_post_spikes_cvs = calc_cv(jaws_post_spikes)

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
    save_dir = "data/training"

    get_ephys_data(jaws_path, f"{save_dir}/jaws_neurons", fixed_length)
    get_ephys_data(npas_path, f"{save_dir}/npas_neurons", fixed_length)

    run_cmd(
        "/Applications/MATLAB_R2021b.app/bin/matlab -nojvm -nodesktop -batch 'get_osc_data_ephys_training'"
    )


def create_pulse_cont_training_data(fixed_length):
    save_dir = "data/training"
    run_cmd(f"mkdir -p {save_dir}")
    run_cmd(f"mkdir -p {save_dir}/naive_neurons")
    run_cmd(f"mkdir -p {save_dir}/dd_neurons")

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

    save_dd_baseline_spikes(dd_baseline_spikes)
    save_naive_baseline_spikes(naive_baseline_spikes)
    save_stat(
        naive_baseline_lengths, "data/training/naive_neurons/cell_baseline_lengths.txt"
    )
    save_stat(dd_baseline_lengths, "data/training/dd_neurons/cell_baseline_lengths.txt")
    save_stat(naive_frs, "data/training/naive_neurons/cell_baseline_frs.txt")
    save_stat(dd_frs, "data/training/dd_neurons/cell_baseline_frs.txt")
    save_stat(naive_cvs, "data/training/naive_neurons/cell_baseline_cvs.txt")
    save_stat(dd_cvs, "data/training/dd_neurons/cell_baseline_cvs.txt")

    save_stat(
        naive_burst_statistics,
        "data/training/naive_neurons/cell_baseline_burst_statistics.txt",
    )
    save_stat(
        dd_burst_statistics,
        "data/training/dd_neurons/cell_baseline_burst_statistics.txt",
    )

    save_meta_list(naive_name, "data/training/naive_neurons/cell_names.txt")
    save_meta_list(naive_folder, "data/training/naive_neurons/cell_folders.txt")
    save_meta_list(naive_mouse, "data/training/naive_neurons/cell_mouse.txt")

    save_meta_list(dd_name, "data/training/dd_neurons/cell_names.txt")
    save_meta_list(dd_folder, "data/training/dd_neurons/cell_folders.txt")
    save_meta_list(dd_mouse, "data/training/dd_neurons/cell_mouse.txt")

    run_cmd(
        f"/Applications/MATLAB_R2021b.app/bin/matlab -nojvm -nodesktop -batch 'get_osc_data_pulse_cont'"
    )


# Create data
create_ephys_training_data(training_fixed_length)
create_pulse_cont_training_data(training_fixed_length)
