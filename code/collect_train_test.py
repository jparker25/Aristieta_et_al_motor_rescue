import numpy as np
import pandas as pd


seeds = 15
dd_data_train = []
ctrl_data_train = []
dd_data_test = []
ctrl_data_test = []

for s in range(seeds):
    train_data = pd.read_csv(f"../data/neural_net/X_train_seed_{s:02d}.csv")
    test_data = pd.read_csv(f"../data/neural_net/X_test_seed_{s:02d}.csv")

    train_data["Seed"] = s
    test_data["Seed"] = s

    ctrl_data_train.append(train_data[train_data["Type"] == 1])
    dd_data_train.append(train_data[train_data["Type"] == 0])

    ctrl_data_test.append(test_data[test_data["Type"] == 1])
    dd_data_test.append(test_data[test_data["Type"] == 0])

full_train_dd = pd.concat(dd_data_train).to_csv(
    "../data/neural_net/dd_train_all_seeds.csv"
)
full_train_ctrl = pd.concat(ctrl_data_train).to_csv(
    "../data/neural_net/ctrl_train_all_seeds.csv"
)
full_test_dd = pd.concat(dd_data_test).to_csv(
    "../data/neural_net/dd_test_all_seeds.csv"
)
full_test_ctrl = pd.concat(ctrl_data_test).to_csv(
    "../data/neural_net/ctrl_test_all_seeds.csv"
)
