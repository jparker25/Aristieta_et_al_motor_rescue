"""
clean_data.py

This script cleans and pre-processes via normalization, outlier removal, and train/test split of dataframes.

Author: John E. Parker (2024)
"""

# python modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


def normalize_data(df_train, df_test, min_max=False):
    """
    Normalizes both dataframes to df_train statistics

    Parameters
    -------
    \t df_train (dataframe) : feature dataframe of training dataframe

    \t df_test (dataframe) : feature dataframe of testing dataframe

    \t min_max=False (boolean) : Option to do min-max scaling instead of normal dist

    Returns
    -------
    \t norm_df_train (dataframe) : normalized feature dataframe of training dataframe

    \t norm_df_test (dataframe) : normalized feature dataframe of testing dataframe
    """

    # Create SKLearn scaler for normalization
    scaler = (
        StandardScaler().fit(df_train) if not min_max else MinMaxScaler().fit(df_train)
    )

    # Normalize training dataframes to df_train
    norm_df_train = pd.DataFrame(
        data=scaler.transform(df_train), columns=df_train.columns, index=df_train.index
    )

    # Normalize testing dataframes to df_train
    norm_df_test = pd.DataFrame(
        data=scaler.transform(df_test), columns=df_train.columns, index=df_test.index
    )

    return norm_df_train, norm_df_test


def remove_outliers_by_group_zscore_independent(naive_df, dd_df, zscore_dict):
    """
    Removes any outliers based on column features with z-score greater than value defined in zscore_dict.

    Parameters
    -------
    \t naive_df (dataframe) : feature dataframe of naive data

    \t dd_df (dataframe) : feature dataframe of DD data

    \t zscore_dict (dict) : dictionary of feature z-score thresholds for outlier removal

    Returns
    -------
    \t combined dataframe of naive_df and dd_df with outliers removed
    """

    # Find naive outliers based on each feature
    naive_outliers_index = []
    for col in zscore_dict.keys():
        outliers = naive_df[col][
            (np.abs(stats.zscore(naive_df[col])) >= zscore_dict[col])
        ].index
        naive_outliers_index.extend(outliers)

    # Find unique naive indicies for outliers
    naive_outliers_index = np.unique(naive_outliers_index)

    # Find DD outliers based on each feature
    dd_outliers_index = []
    for col in zscore_dict.keys():
        outliers = dd_df[col][
            (np.abs(stats.zscore(dd_df[col])) >= zscore_dict[col])
        ].index
        dd_outliers_index.extend(outliers)

    # Find unique naive indicies for outliers
    dd_outliers_index = np.unique(dd_outliers_index)

    # Print number of outliers removed
    print(f"Removed {len(naive_outliers_index)} outliers from {len(naive_df)} samples")
    print(f"Removed {len(dd_outliers_index)} outliers from {len(dd_df)} samples")

    # Drop outliers from naive_df and dd_df dataframes
    naive_df = naive_df.drop(naive_outliers_index)
    dd_df = dd_df.drop(dd_outliers_index)

    # Define types for each dataframe
    naive_df["Type"] = 1
    dd_df["Type"] = 0

    return pd.concat([naive_df, dd_df])


def split_data(feature_df, target_df, train_amount, seed=24):
    """
    Split feature and target dataframes by train_amount in to train/test splits.

    Parameters
    -------
    \t feature_df (dataframe) : feature dataframe of training data

    \t target_df (dataframe) : feature dataframe of target data (used to get class types)

    \t train_amount (double) : Decimal < 1 to determine train/test split of feature data

    \t seed=24 (integer) : Integer to set random seed

    Returns
    -------
    \t X_train (dataframe) : dataframe of naive and DD data for training MLPs

    \t X_test (dataframe) : dataframe of naive and DD data for testing MLPs

    \t y_train (dataframe) : single column dataframe of type classes for X_train

    \t y_test (dataframe) : single column dataframe of type classes for X_test
    """

    # define naive and DD feature dataframes based on type classes
    feature_naive_df = feature_df[target_df["Type"] == 1]
    feature_dd_df = feature_df[target_df["Type"] == 0]

    # Set up seed
    np.random.seed(seed)

    # Split naive feature by train and test amounts
    feature_naive_df = feature_naive_df.sample(frac=1)
    naive_test_size = int(feature_naive_df.shape[0] * train_amount)
    naive_train = feature_naive_df.iloc[0:naive_test_size, :]
    naive_test = feature_naive_df.iloc[naive_test_size:]
    naive_train_target = np.ones(len(naive_train))
    naive_test_target = np.ones(len(naive_test))

    # Split DD feature by train and test amounts
    feature_dd_df = feature_dd_df.sample(frac=1)
    dd_test_size = int(feature_dd_df.shape[0] * train_amount)
    dd_train = feature_dd_df.iloc[0:dd_test_size, :]
    dd_test = feature_dd_df.iloc[dd_test_size:]
    dd_train_target = np.zeros(len(dd_train))
    dd_test_target = np.zeros(len(dd_test))

    # Create train and test dataframes and class targets
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

    # Shuffle data appropriately
    X_train = X_train.sample(frac=1, random_state=0, ignore_index=False)
    y_train = y_train.sample(frac=1, random_state=0, ignore_index=False)
    X_test = X_test.sample(frac=1, random_state=0, ignore_index=False)
    y_test = y_test.sample(frac=1, random_state=0, ignore_index=False)

    return X_train, X_test, y_train["Type"], y_test["Type"]
