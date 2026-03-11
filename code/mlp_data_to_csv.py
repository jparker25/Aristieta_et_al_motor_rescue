import numpy as np
import os, sys, json
import pandas as pd

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


def dd_data_set_to_csv(paths, new_path):
    # cell_count = 0
    # light_on = 0
    spike_trains = []
    light_on = []
    data_set = []
    unit_name = []

    for path in paths:
        for subdir in os.listdir(path):
            if subdir != ".DS_Store" and "6-OHDA" in subdir and not "hsyn" in subdir:
                for neuro_dir in os.listdir(f"{path}/{subdir}"):
                    if neuro_dir != ".DS_Store":
                        # cell_count += 1
                        spike_trains.append(
                            np.loadtxt(f"{path}/{subdir}/{neuro_dir}/spikes.txt")
                        )
                        light_on.append(
                            np.loadtxt(f"{path}/{subdir}/{neuro_dir}/light_on.txt")
                        )
                        data_set.append(subdir)
                        unit_name.append(neuro_dir)

    light_on_data = {
        f"{data_set[i]}: {unit_name[i]}": pd.Series(arr)
        for i, arr in enumerate(light_on)
    }
    # Create the DataFrame and export to CSV
    df = pd.DataFrame(light_on_data)
    df.to_csv(f"{new_path}/DD_data_light_on.csv", index=False)

    baseline_data = {
        f"{data_set[i]}: {unit_name[i]}": pd.Series(arr)
        for i, arr in enumerate(spike_trains)
    }

    # Create the DataFrame and export to CSV
    df = pd.DataFrame(baseline_data)
    df.to_csv(f"{new_path}/DD_data.csv", index=False)


def naive_data_set_to_csv(paths, new_path):
    # cell_count = 0
    # light_on = 0
    spike_trains = []
    light_on = []
    data_set = []
    unit_name = []

    for path in paths:
        for subdir in os.listdir(path):
            if subdir != ".DS_Store" and "Naive" in subdir and not "hsyn" in subdir:
                for neuro_dir in os.listdir(f"{path}/{subdir}"):
                    if neuro_dir != ".DS_Store":
                        # cell_count += 1
                        spike_trains.append(
                            np.loadtxt(f"{path}/{subdir}/{neuro_dir}/spikes.txt")
                        )
                        light_on.append(
                            np.loadtxt(f"{path}/{subdir}/{neuro_dir}/light_on.txt")
                        )
                        data_set.append(subdir)
                        unit_name.append(neuro_dir)

    light_on_data = {
        f"{data_set[i]}: {unit_name[i]}": pd.Series(arr)
        for i, arr in enumerate(light_on)
    }
    # Create the DataFrame and export to CSV
    df = pd.DataFrame(light_on_data)
    df.to_csv(f"{new_path}/naive_data_light_on.csv", index=False)

    baseline_data = {
        f"{data_set[i]}: {unit_name[i]}": pd.Series(arr)
        for i, arr in enumerate(spike_trains)
    }

    # Create the DataFrame and export to CSV
    df = pd.DataFrame(baseline_data)
    df.to_csv(f"{new_path}/naive_data.csv", index=False)


def jaws_npas_data_set_to_csv(paths, new_path):
    spike_trains = []
    cell_name = []
    mouse = []
    exp_type = []
    for path in paths:
        for neuron in sorted(os.listdir(f"{path}/pre_opto")):
            if neuron != ".DS_Store":
                meta_file = open(f"{path}/pre_opto/{neuron}/meta_data.txt", "r")
                meta_data = json.load(meta_file)
                mouse.append(meta_data["mouse"])
                cell_name.append(f"{meta_data["cell_name"]} {neuron}")
                exp_type.append(meta_data["type"])
                spike_trains.append(np.loadtxt(f"{path}/pre_opto/{neuron}/spikes.txt"))
    data = {
        f"{exp_type[i]} {mouse[i]} {cell_name[i]}": pd.Series(arr)
        for i, arr in enumerate(spike_trains)
    }

    # Create the DataFrame and export to CSV
    df = pd.DataFrame(data)
    df.to_csv(f"{new_path}/jaws_npas_data.csv", index=False)


dd_training_paths = [
    dd_cont_d1_data,
    dd_cont_gpe_data,
    dd_pulse_d1_data,
    dd_pulse_gpe_data,
]

dd_data_set_to_csv(dd_training_paths, "/Users/johnparker/Desktop")

naive_training_paths = [
    cont_d1_data,
    cont_gpe_data,
    pulse_d1_data,
    pulse_gpe_data,
]

naive_data_set_to_csv(naive_training_paths, "/Users/johnparker/Desktop")

jaws_npas_training_paths = [jaws_path, npas_path]


jaws_npas_data_set_to_csv(jaws_npas_training_paths, "/Users/johnparker/Desktop")
