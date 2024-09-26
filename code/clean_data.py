import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os, sys
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from scipy import stats


def normalize_data(df_train, df_test, min_max=False):
    scaler = (
        StandardScaler().fit(df_train) if not min_max else MinMaxScaler().fit(df_train)
    )
    norm_df_train = pd.DataFrame(
        data=scaler.transform(df_train), columns=df_train.columns, index=df_train.index
    )
    norm_df_test = pd.DataFrame(
        data=scaler.transform(df_test), columns=df_train.columns, index=df_test.index
    )

    # norm_df_train = norm_df_train.dropna(axis=0, how="any")
    # norm_df_test = norm_df_test.dropna(axis=0, how="any")
    return norm_df_train, norm_df_test


def remove_outliers_by_group_zscore_independent(naive_df, dd_df, zscore_dict):
    naive_outliers_index = []
    # naive_df = naive_df.iloc[:, 1:]
    for col in zscore_dict.keys():
        outliers = naive_df[col][
            (np.abs(stats.zscore(naive_df[col])) >= zscore_dict[col])
        ].index
        naive_outliers_index.extend(outliers)
    naive_outliers_index = np.unique(naive_outliers_index)

    dd_outliers_index = []
    # dd_df = dd_df.iloc[:, 1:]
    for col in zscore_dict.keys():
        outliers = dd_df[col][
            (np.abs(stats.zscore(dd_df[col])) >= zscore_dict[col])
        ].index
        dd_outliers_index.extend(outliers)
    dd_outliers_index = np.unique(dd_outliers_index)

    print(f"Removed {len(naive_outliers_index)} outliers from {len(naive_df)} samples")
    print(f"Removed {len(dd_outliers_index)} outliers from {len(dd_df)} samples")

    naive_df = naive_df.drop(naive_outliers_index)
    dd_df = dd_df.drop(dd_outliers_index)
    naive_df["Type"] = 1
    dd_df["Type"] = 0

    return pd.concat([naive_df, dd_df])


def split_data(feature_df, target_df, train_amount, seed=24):
    feature_naive_df = feature_df[target_df["Type"] == 1]
    feature_dd_df = feature_df[target_df["Type"] == 0]

    np.random.seed(seed)
    feature_naive_df = feature_naive_df.sample(frac=1)
    naive_test_size = int(feature_naive_df.shape[0] * train_amount)
    naive_train = feature_naive_df.iloc[0:naive_test_size, :]
    naive_test = feature_naive_df.iloc[naive_test_size:]
    naive_train_target = np.ones(len(naive_train))
    naive_test_target = np.ones(len(naive_test))

    feature_dd_df = feature_dd_df.sample(frac=1)
    dd_test_size = int(feature_dd_df.shape[0] * train_amount)
    dd_train = feature_dd_df.iloc[0:dd_test_size, :]
    dd_test = feature_dd_df.iloc[dd_test_size:]
    dd_train_target = np.zeros(len(dd_train))
    dd_test_target = np.zeros(len(dd_test))

    X_train = pd.concat([naive_train, dd_train], ignore_index=False)
    X_test = pd.concat([naive_test, dd_test], ignore_index=False)
    y_train = pd.DataFrame(
        data=np.concatenate(
            [naive_train_target, dd_train_target],
        ),
        columns=["Type"],
        index=X_train.index,
    )
    y_test = pd.DataFrame(
        data=np.concatenate(
            [naive_test_target, dd_test_target],
        ),
        columns=["Type"],
        index=X_test.index,
    )
    X_train = X_train.sample(frac=1, random_state=0, ignore_index=False)
    y_train = y_train.sample(frac=1, random_state=0, ignore_index=False)
    X_test = X_test.sample(frac=1, random_state=0, ignore_index=False)
    y_test = y_test.sample(frac=1, random_state=0, ignore_index=False)

    return X_train, X_test, y_train["Type"], y_test["Type"]
