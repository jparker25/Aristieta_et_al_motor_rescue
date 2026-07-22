"""
read_in_motor_rescue_recordings.py

Reads in experimental data and parses into format to be processed for analysis.

Author: John E. Parker (2024)
"""

# python modules
import numpy as np
import pandas as pd
import os
import json
import pandas as pd

# user modules
from helpers import *


def read_in_jaws():
    """
    Reads in JAWS data and parses into format to be processed for analysis.
    """

    # Set directories
    direc = "../mlp_data/SNr_motor_rescue_project/hsyn-Jaws_in_SNr_6-OHDA_mouse"
    save_direc = (
        "../mlp_data/SNr_motor_rescue_project/jaws_pre_processed"
    )

    # Create save directory
    run_cmd(f"mkdir -p {save_direc}")

    # Initialize variables
    pre_opto_cell_count = 0
    post_opto_cell_count = 0
    pre_post_opto_cell_count = 0
    post_times = []
    missed_folders = []
    all_distances = []

    # Loop through all mice and experiments
    for mouse in os.listdir(direc):
        # Skip hidden files
        if mouse != ".DS_Store":
            # Loop through all experiments
            for exp in os.listdir(f"{direc}/{mouse}"):
                if exp != ".DS_Store":
                    # Extract distance from experiment name
                    distance = exp.split("_")
                    dist_index = 0
                    for entry in distance:
                        if "mm" in entry:
                            break
                        else:
                            dist_index += 1
                    distance = eval(distance[dist_index][:-2])
                    all_distances.append(distance)
                    # Determine if medial or lateral
                    medial = True if distance < 1.3 else False

                    # Check if experiment is opto-stim or 10x30 and not post
                    if ("opto-stim" in exp or "10x30" in exp) and "post" not in exp:
                        for csv_path in os.listdir(f"{direc}/{mouse}/{exp}"):
                            # Check if baseline or post
                            if "baseline" in csv_path:
                                csv = pd.read_csv(f"{direc}/{mouse}/{exp}/{csv_path}")
                                for col in csv.columns:
                                    # Grab values and remove NaNs
                                    spikes = csv[col].values[~np.isnan(csv[col].values)]

                                    # Set spike times from 0 to 200
                                    spikes = spikes - np.floor(np.min(csv.iloc[0, :]))

                                    pre_opto_cell_count += 1
                                    cell_direc = f"{save_direc}/pre_opto/Neuron_{pre_opto_cell_count:04d}"
                                    run_cmd(f"mkdir -p {cell_direc}", print_out=False)

                                    # Save meta data
                                    meta_data = {
                                        "mouse": mouse,
                                        "folder": exp,
                                        "cell_name": col,
                                        "type": "jaws",
                                        "medial": medial,
                                    }
                                    with open(
                                        f"{cell_direc}/meta_data.txt", "w"
                                    ) as file:
                                        json.dump(meta_data, file)
                                    np.savetxt(
                                        f"{cell_direc}/spikes.txt",
                                        spikes,
                                        fmt="%f",
                                        newline="\n",
                                        delimiter=",",
                                    )

                            # Check if post
                            elif "post" in csv_path:
                                csv = pd.read_csv(f"{direc}/{mouse}/{exp}/{csv_path}")
                                spl = csv_path.split("_")

                                # Find time of post
                                min_index = 0
                                for entry in spl:
                                    if "min" in entry:
                                        break
                                    else:
                                        min_index += 1
                                if spl[min_index] not in post_times:
                                    post_times.append(spl[min_index])
                                for col in csv.columns:
                                    # Grab values and remove NaNs
                                    spikes = csv[col].values[~np.isnan(csv[col].values)]

                                    # Set spike times from 0 to 200
                                    spikes = spikes - np.floor(np.min(csv.iloc[0, :]))

                                    # Save post opto data
                                    post_opto_cell_count += 1
                                    cell_direc = f"{save_direc}/post_opto/Neuron_{post_opto_cell_count:04d}"
                                    run_cmd(f"mkdir -p {cell_direc}", print_out=False)
                                    meta_data = {
                                        "mouse": mouse,
                                        "folder": exp,
                                        "cell_name": col,
                                        "type": "jaws",
                                        "post_time": eval(
                                            spl[min_index][: spl[min_index].index("m")]
                                        ),
                                        "medial": medial,
                                    }
                                    with open(
                                        f"{cell_direc}/meta_data.txt", "w"
                                    ) as file:
                                        json.dump(meta_data, file)
                                    np.savetxt(
                                        f"{cell_direc}/spikes.txt",
                                        spikes,
                                        fmt="%f",
                                        newline="\n",
                                        delimiter=",",
                                    )
                            # Check if pre-post opto
                            else:

                                # Read in excel file
                                df = pd.read_excel(f"{direc}/{mouse}/{exp}/{csv_path}")
                                if (
                                    np.sum(["Unnamed" in col for col in df.columns])
                                    == df.shape[1]
                                ):
                                    df.columns = df.iloc[0]
                                    df = df[1:]

                                df = df.iloc[:, ~pd.isna(df.columns)]
                                df = df.astype(float)

                                # Drop columns with no data
                                dropcols = []
                                for col in df.columns:
                                    if "Unnamed" in col:
                                        dropcols.append(col)
                                # Drop columns
                                df = df.drop(columns=dropcols)
                                spikes = []
                                channel = None
                                # Loop through columns
                                for col in df.columns:
                                    if "STIM" in col and "ON" in col:
                                        light_on = df[col].values[
                                            ~np.isnan(df[col].values)
                                        ]
                                    elif "STIM" in col and "OFF" in col:
                                        light_off = df[col].values[
                                            ~np.isnan(df[col].values)
                                        ]
                                    elif "CHANNEL" in col:
                                        spikes.append(
                                            df[col].values[~np.isnan(df[col].values)]
                                        )
                                        channel = col
                                # Save data
                                light_on = np.column_stack((light_on, light_off))
                                for i in range(len(spikes)):
                                    pre_post_opto_cell_count += 1
                                    cell_direc = f"{save_direc}/pre_post_opto/Neuron_{pre_post_opto_cell_count:04d}"
                                    run_cmd(f"mkdir -p {cell_direc}", print_out=False)
                                    meta_data = {
                                        "mouse": mouse,
                                        "folder": exp,
                                        "cell_name": channel,
                                        "type": "npas",
                                        "medial": medial,
                                    }
                                    with open(
                                        f"{cell_direc}/meta_data.txt", "w"
                                    ) as file:
                                        json.dump(meta_data, file)
                                    np.savetxt(
                                        f"{cell_direc}/spikes.txt",
                                        spikes[i],
                                        fmt="%f",
                                        newline="\n",
                                        delimiter="\t",
                                    )
                                    np.savetxt(
                                        f"{cell_direc}/light_on.txt",
                                        light_on,
                                        fmt="%f",
                                        newline="\n",
                                        delimiter="\t",
                                    )

                    # Check if post
                    elif "post" in exp:

                        spl = exp.split("_")
                        min_index = 0
                        for entry in spl:
                            if "min" in entry:
                                break
                            else:
                                min_index += 1
                        # Find time of post
                        if spl[min_index] not in post_times:
                            post_times.append(spl[min_index])

                        # Loop through all columns
                        for csv_path in os.listdir(f"{direc}/{mouse}/{exp}"):
                            csv = pd.read_csv(f"{direc}/{mouse}/{exp}/{csv_path}")
                            for col in csv.columns:
                                # Grab values and remove NaNs
                                spikes = csv[col].values[~np.isnan(csv[col].values)]

                                # Set spike times from 0 to 200
                                spikes = spikes - np.floor(np.min(csv.iloc[0, :]))

                                # Save post opto data
                                post_opto_cell_count += 1
                                cell_direc = f"{save_direc}/post_opto/Neuron_{post_opto_cell_count:04d}"
                                run_cmd(f"mkdir -p {cell_direc}", print_out=False)
                                meta_data = {
                                    "mouse": mouse,
                                    "folder": exp,
                                    "cell_name": col,
                                    "type": "jaws",
                                    "post_time": eval(
                                        spl[min_index][: spl[min_index].index("m")]
                                    ),
                                    "medial": medial,
                                }
                                with open(f"{cell_direc}/meta_data.txt", "w") as file:
                                    json.dump(meta_data, file)
                                np.savetxt(
                                    f"{cell_direc}/spikes.txt",
                                    spikes,
                                    fmt="%f",
                                    newline="\n",
                                    delimiter=",",
                                )

                    # Check if pre-opto
                    elif "post" not in exp and "opto" not in exp:
                        # Loop through all columns
                        for csv_path in os.listdir(f"{direc}/{mouse}/{exp}"):
                            csv = pd.read_csv(f"{direc}/{mouse}/{exp}/{csv_path}")
                            for col in csv.columns:
                                # Grab values and remove NaNs
                                spikes = csv[col].values[~np.isnan(csv[col].values)]

                                # Set spike times from 0 to 200
                                spikes = spikes - np.floor(np.min(csv.iloc[0, :]))

                                # Save pre opto data
                                pre_opto_cell_count += 1
                                cell_direc = f"{save_direc}/pre_opto/Neuron_{pre_opto_cell_count:04d}"
                                run_cmd(f"mkdir -p {cell_direc}", print_out=False)
                                meta_data = {
                                    "mouse": mouse,
                                    "folder": exp,
                                    "cell_name": col,
                                    "type": "jaws",
                                    "medial": medial,
                                }
                                with open(f"{cell_direc}/meta_data.txt", "w") as file:
                                    json.dump(meta_data, file)
                                np.savetxt(
                                    f"{cell_direc}/spikes.txt",
                                    spikes,
                                    fmt="%f",
                                    newline="\n",
                                    delimiter=",",
                                )
                    # If not pre-opto, post-opto, or pre-post opto, then missed
                    else:
                        missed_folders.append([mouse, exp])
    # Print out results
    print(f"From JAWS dataset found: ")
    print(f"\tPre-opto units: {pre_opto_cell_count}")
    print(f"\tPost-opto units: {post_opto_cell_count}")
    print(f"\tPost-opto times: {sorted(post_times)}")
    print(f"\tPre-Post Units: {pre_post_opto_cell_count}")
    print(f"\tFolders missed: {missed_folders}")


def read_in_npas():
    """
    Reads in NPAS data and parses into format to be processed for analysis.
    """

    # Set directories
    direc = "../mlp_data/SNr_motor_rescue_project/Npas-cre_mouse_DIO-ChR2_in_SNr_6-OHDA"
    save_direc = (
        "../mlp_data/SNr_motor_rescue_project/npas_pre_processed"
    )

    # Create save directory
    run_cmd(f"mkdir -p {save_direc}")

    # Initialize variables
    pre_opto_cell_count = 0
    post_opto_cell_count = 0
    post_times = []
    missed_folders = []
    opto_tagging = []
    all_distances = []
    pre_post_opto_cell_count = 0

    # Loop through all mice and experiments
    for mouse in os.listdir(direc):
        # Skip hidden files
        if mouse != ".DS_Store" and "Mouse" in mouse:
            # Loop through all experiments
            for exp in os.listdir(f"{direc}/{mouse}"):
                # Skip hidden files
                if exp != ".DS_Store":
                    # Extract distance from experiment name
                    distance = exp.split("_")
                    dist_index = 0
                    for entry in distance:
                        if "mm" in entry:
                            break
                        else:
                            dist_index += 1
                    distance = eval(distance[dist_index][:-2])
                    all_distances.append(distance)
                    # Determine if medial or lateral
                    medial = True if distance < 1.3 else False

                    # Check if experiment is opto-stim or 10x30 and not post
                    if ("opto-stim" in exp or "10x30" in exp) and "post" not in exp:
                        for csv_path in os.listdir(f"{direc}/{mouse}/{exp}"):
                            # Check if baseline or post
                            if "baseline" in csv_path:
                                csv = pd.read_csv(f"{direc}/{mouse}/{exp}/{csv_path}")
                                for col in csv.columns:
                                    # Grab values and remove NaNs
                                    spikes = csv[col].values[~np.isnan(csv[col].values)]

                                    # Set spike times from 0 to 200
                                    spikes = spikes - np.floor(np.min(csv.iloc[0, :]))

                                    # Save pre opto data
                                    pre_opto_cell_count += 1
                                    cell_direc = f"{save_direc}/pre_opto/Neuron_{pre_opto_cell_count:04d}"
                                    run_cmd(f"mkdir -p {cell_direc}", print_out=False)
                                    meta_data = {
                                        "mouse": mouse,
                                        "folder": exp,
                                        "cell_name": col,
                                        "type": "npas",
                                        "medial": medial,
                                    }
                                    with open(
                                        f"{cell_direc}/meta_data.txt", "w"
                                    ) as file:
                                        json.dump(meta_data, file)
                                    np.savetxt(
                                        f"{cell_direc}/spikes.txt",
                                        spikes,
                                        fmt="%f",
                                        newline="\n",
                                        delimiter=",",
                                    )

                            # Check if post
                            elif "post" in csv_path:
                                csv = pd.read_csv(f"{direc}/{mouse}/{exp}/{csv_path}")
                                spl = csv_path.split("_")
                                # Find time of post
                                min_index = 0
                                for entry in spl:
                                    if "min" in entry:
                                        break
                                    else:
                                        min_index += 1
                                if spl[min_index] not in post_times:
                                    post_times.append(spl[min_index])
                                # Loop through all columns
                                for col in csv.columns:
                                    # Grab values and remove NaNs
                                    spikes = csv[col].values[~np.isnan(csv[col].values)]

                                    # Set spike times from 0 to 200
                                    spikes = spikes - np.floor(np.min(csv.iloc[0, :]))

                                    # Save post opto data
                                    post_opto_cell_count += 1
                                    cell_direc = f"{save_direc}/post_opto/Neuron_{post_opto_cell_count:04d}"
                                    run_cmd(f"mkdir -p {cell_direc}", print_out=False)
                                    meta_data = {
                                        "mouse": mouse,
                                        "folder": exp,
                                        "cell_name": col,
                                        "type": "npas",
                                        "post_time": eval(
                                            spl[min_index][: spl[min_index].index("m")]
                                        ),
                                        "medial": medial,
                                    }
                                    with open(
                                        f"{cell_direc}/meta_data.txt", "w"
                                    ) as file:
                                        json.dump(meta_data, file)
                                    np.savetxt(
                                        f"{cell_direc}/spikes.txt",
                                        spikes,
                                        fmt="%f",
                                        newline="\n",
                                        delimiter=",",
                                    )
                            # Check if pre-post opto
                            else:
                                # Read in excel file
                                df = pd.read_excel(f"{direc}/{mouse}/{exp}/{csv_path}")
                                # Check if all columns are unnamed
                                if (
                                    np.sum(["Unnamed" in col for col in df.columns])
                                    == df.shape[1]
                                ):
                                    df.columns = df.iloc[0]
                                    df = df[1:]
                                # Drop columns with no data
                                df = df.iloc[:, ~pd.isna(df.columns)]
                                df = df.astype(float)

                                dropcols = []
                                for col in df.columns:
                                    if "Unnamed" in col:
                                        dropcols.append(col)

                                df = df.drop(columns=dropcols)
                                spikes = []
                                # Loop through columns
                                for col in df.columns:
                                    if "STIM" in col and "ON" in col:
                                        light_on = df[col].values[
                                            ~np.isnan(df[col].values)
                                        ]
                                    elif "STIM" in col and "OFF" in col:
                                        light_off = df[col].values[
                                            ~np.isnan(df[col].values)
                                        ]
                                    elif "CHANNEL" in col:
                                        spikes.append(
                                            df[col].values[~np.isnan(df[col].values)]
                                        )
                                # Save data
                                print(spikes)
                                pre_post_opto_cell_count += len(spikes)

                    # Check if post
                    elif "post" in exp:
                        # Loop through all columns
                        for csv_path in os.listdir(f"{direc}/{mouse}/{exp}"):
                            csv = pd.read_csv(f"{direc}/{mouse}/{exp}/{csv_path}")
                            spl = csv_path.split("_")
                            # Find time of post
                            min_index = 0
                            for entry in spl:
                                if "min" in entry:
                                    break
                                else:
                                    min_index += 1
                            if spl[min_index] not in post_times:
                                post_times.append(spl[min_index])
                            # Loop through all columns
                            for col in csv.columns:
                                # Grab values and remove NaNs
                                spikes = csv[col].values[~np.isnan(csv[col].values)]

                                # Set spike times from 0 to 200
                                spikes = spikes - np.floor(np.min(csv.iloc[0, :]))

                                # Save post opto data
                                post_opto_cell_count += 1
                                cell_direc = f"{save_direc}/post_opto/Neuron_{post_opto_cell_count:04d}"
                                run_cmd(f"mkdir -p {cell_direc}", print_out=False)
                                meta_data = {
                                    "mouse": mouse,
                                    "folder": exp,
                                    "cell_name": col,
                                    "type": "npas",
                                    "post_time": eval(
                                        spl[min_index][: spl[min_index].index("m")]
                                    ),
                                    "medial": medial,
                                }
                                with open(f"{cell_direc}/meta_data.txt", "w") as file:
                                    json.dump(meta_data, file)
                                np.savetxt(
                                    f"{cell_direc}/spikes.txt",
                                    spikes,
                                    fmt="%f",
                                    newline="\n",
                                    delimiter=",",
                                )

                    # Check if pre-opto
                    elif "post" not in exp and "opto" not in exp:
                        # Loop through all columns
                        for csv_path in os.listdir(f"{direc}/{mouse}/{exp}"):
                            csv = pd.read_csv(f"{direc}/{mouse}/{exp}/{csv_path}")
                            # Check if all columns are unnamed
                            for col in csv.columns:
                                # Grab values and remove NaNs
                                spikes = csv[col].values[~np.isnan(csv[col].values)]

                                # Set spike times from 0 to 200
                                spikes = spikes - np.floor(np.min(csv.iloc[0, :]))

                                # Save pre opto data
                                pre_opto_cell_count += 1
                                cell_direc = f"{save_direc}/pre_opto/Neuron_{pre_opto_cell_count:04d}"
                                run_cmd(f"mkdir -p {cell_direc}", print_out=False)
                                meta_data = {
                                    "mouse": mouse,
                                    "folder": exp,
                                    "cell_name": col,
                                    "type": "npas",
                                    "medial": medial,
                                }
                                if col.split(" ")[1] not in opto_tagging:
                                    opto_tagging.append(col.split(" ")[1])
                                with open(f"{cell_direc}/meta_data.txt", "w") as file:
                                    json.dump(meta_data, file)
                                np.savetxt(
                                    f"{cell_direc}/spikes.txt",
                                    spikes,
                                    fmt="%f",
                                    newline="\n",
                                    delimiter=",",
                                )
                    # If not pre-opto, post-opto, or pre-post opto, then missed
                    else:
                        missed_folders.append([mouse, exp])
        # If not mouse, then must be a file
        elif mouse != ".DS_Store":
            # Extract distance from experiment name
            for csv_path in os.listdir(f"{direc}/{mouse}"):
                # Check if baseline or post
                df = pd.read_excel(f"{direc}/{mouse}/{csv_path}")
                if np.sum(["Unnamed" in col for col in df.columns]) == df.shape[1]:
                    df.columns = df.iloc[0]
                    df = df[1:]

                df = df.iloc[:, ~pd.isna(df.columns)]
                df = df.astype(float)

                dropcols = []
                for col in df.columns:
                    if "Unnamed" in col:
                        dropcols.append(col)

                df = df.drop(columns=dropcols)
                spikes = []
                # Loop through columns
                channel = None
                for col in df.columns:
                    if "Stim" in col and "ON" in col:
                        light_on = df[col].values[~np.isnan(df[col].values)]
                    elif "Stim" in col and "OFF" in col:
                        light_off = df[col].values[~np.isnan(df[col].values)]
                    elif "CHANNEL" in col:
                        spikes.append(df[col].values[~np.isnan(df[col].values)])
                        channel = col
                # Save data
                light_on = np.column_stack((light_on, light_off))
                for i in range(len(spikes)):
                    pre_post_opto_cell_count += 1
                    cell_direc = f"{save_direc}/pre_post_opto/Neuron_{pre_post_opto_cell_count:04d}"
                    run_cmd(f"mkdir -p {cell_direc}", print_out=False)
                    meta_data = {
                        "mouse": mouse,
                        "folder": exp,
                        "cell_name": channel,
                        "type": "npas",
                        "medial": medial,
                    }
                    with open(f"{cell_direc}/meta_data.txt", "w") as file:
                        json.dump(meta_data, file)
                    np.savetxt(
                        f"{cell_direc}/spikes.txt",
                        spikes[i],
                        fmt="%f",
                        newline="\n",
                        delimiter="\t",
                    )
                    np.savetxt(
                        f"{cell_direc}/light_on.txt",
                        light_on,
                        fmt="%f",
                        newline="\n",
                        delimiter="\t",
                    )
    # Print out results
    print(f"From NPAS dataset found: ")
    print(f"\tPre-opto units: {pre_opto_cell_count}")
    print(f"\tPost-opto units: {post_opto_cell_count}")
    print(f"\tPost-opto times: {sorted(post_times)}")
    print(f"\tPre-Post Units: {pre_post_opto_cell_count}")
    print(f"\tFolders missed: {missed_folders}")


# Run functions to read in data for further processing
read_in_jaws()
read_in_npas()
