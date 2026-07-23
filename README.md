This repo contains data and figures for the paper by Aristieta et al., under review. [biorxiv](https://www.biorxiv.org/content/10.1101/2024.12.09.627637v1)

A Python virtual environment is advised and is assumed in all instructions in this README. This code was developed and tested on an Apple Silicon M1, 2020 (16 GB).

For any questions or issues please contact the owner of this repository or the corresponding author.

# Overview

## Data breakdown
Data is stored in `mlp_data` directory. Below is a breakdown of where each experimental dataset exists within `mlp_data`.

- Npas_cre_UniDD: `../mlp_data/SNr_motor_rescue_project/Data_for_resubmission/Npas_cre_UniDD`
- Npas_cre_BilateralDD: `../mlp_data/SNr_motor_rescue_project/Data_for_resubmission/Npas_cre_BilateralDD`
- PV_Bilateral_DD: `../mlp_data/SNr_motor_rescue_project/Data_for_resubmission/SNr-PV_bilateral_DD`
- JAWS DD: `../mlp_data/SNr_motor_rescue_project/hsyn-Jaws_in_SNr_6-OHDA_mouse`
- NPAS DD: `../mlp_data/SNr_motor_rescue_project/Npas-cre_mouse_DIO-ChR2_in_SNr_6-OHDA`

## Running MLP on data set for figures.
Steps to reproduce all figures are as follows and assumes the correct virtual environment with appropriate dependencies (See following section). 

1) Read in new mouse recordings data stored in `../mlp_data/SNr_motor_rescue_project`. Run 
- `$ python read_in_motor_rescue_recordings.py` 
- `$ python read_in_resubmission_data.py`
2) Parse datasets into usable formats for analysis pipeline (e.g. extract features and place into appropriate files)
- `$ python parse_data.py`
- `$ python parse_resubmit_data.py`
- `$ python parse_unilateral_data.py`
3) Run analysis. The script `run_analysis.py` is the main script that sets up any analysis to run. By default, all analysis occurs. Highly recommend reading through that script before implementing.
- `$ python run_analysis.py`

## Python Virtual Environment
Python version 3.12.13 was used to implement this repository.

Please see [https://docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html) for instructions on how to create a virtual environment on your machine.

Then, read in required Python modules via `$ pip install -r requirements.txt`

For a brief overview on usage see below.

## MATLAB
Python scripts call MATLAB R2021b via the terminal. 

