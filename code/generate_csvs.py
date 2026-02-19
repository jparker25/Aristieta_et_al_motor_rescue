import numpy as np
import pandas as pd
import pickle

from helpers import *
import clean_data

mlp_seeds = 15
df = pd.read_csv("/Users/johnparker/Desktop/jaws_npas_pre_post.csv")
df = df.drop(columns="Unnamed: 0")
print(df.columns)

partial_df = df.iloc[:, 0:12]

predicts = np.zeros(mlp_seeds)
probs = np.zeros((mlp_seeds, len(df)))

for i in range(mlp_seeds):
    seed_train_data = pd.read_csv(f"../data/neural_net/X_train_seed_{int(i):02d}.csv")
    seed_train_data = seed_train_data.drop(
        ["Unnamed: 0", "DD Probability", "Type"], axis=1
    )
    _, X_test_norm = clean_data.normalize_data(
        seed_train_data, partial_df, min_max=False
    )

    with open(f"../data/neural_net/MLP_seed_{int(i):02d}.pkl", "rb") as file:
        clf = pickle.load(file)
        predict_test = clf.predict(X_test_norm)
        predicts[i] = 1 - np.sum(predict_test) / len(X_test_norm)
        probs[i, :] = clf.predict_proba(X_test_norm)[:, 0]
df["DD Confidence"] = np.mean(probs, axis=0)

df.to_csv("/Users/johnparker/Desktop/jaws_npas_pre_post_conf.csv")
