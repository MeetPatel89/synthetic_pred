from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def find_non_numeric_cols(df: pd.DataFrame) -> List[Tuple[str, np.generic]]:
    """Find columns in a DataFrame that are not numeric."""
    return [
        i[0]
        for i in df.dtypes.items()
        if not ("int" in i[1].name or "float" in i[1].name)
    ]


def find_null_cols(df: pd.DataFrame) -> List[str]:
    """Find columns in a DataFrame that contain null values."""
    return [col for col in df if df[col].isnull().any()]


def find_null_cols_pct(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "null_pct": [df[col].isnull().sum() / df.shape[0] for col in df],
            "null_count": [df[col].isnull().sum() for col in df],
        },
        index=df.columns,
    )


def classify_null_cols(
    df: pd.DataFrame, null_cols: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """Classify null columns as categorical or numerical."""
    if null_cols is None:
        null_cols = find_null_cols(df)
    if df is None:
        raise ValueError("DataFrame 'df' cannot be None")
    return {
        "categorical": [col for col in df[null_cols].select_dtypes(include="O")],
        "numerical": [col for col in df[null_cols].select_dtypes(exclude="O")],
    }


def drop_null_cols(
    df: pd.DataFrame,
    null_cols: Optional[List[str]] = None,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Drop columns in a DataFrame that contain null values."""
    if null_cols is None:
        null_cols = find_null_cols(df)

    cols_to_drop = []

    # if threshold for dropping null columns is not given assume 50% of dataframe length
    if threshold is None:
        cols_to_drop = [
            col for col in null_cols if df[col].isnull().sum() > df.shape[0] / 2
        ]
    else:
        cols_to_drop = [col for col in null_cols if df[col].isnull().sum() > threshold]

    return df.drop(columns=cols_to_drop)


def test_train_split_by_nulls(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into training and test sets based on null values."""
    null_mask = df.isnull().any(axis=1)
    train_df = df[~null_mask].copy()
    test_df = df[null_mask].copy()
    return train_df, test_df


def split_df_k_fold(
    df: pd.DataFrame, k: int, random_state: int = 42
) -> List[pd.DataFrame]:
    """Split a DataFrame into k folds."""
    if df is None:
        raise ValueError("DataFrame 'df' cannot be None")
    if k < 2:
        raise ValueError("k must be greater than 1")
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k > len(df):
        raise ValueError("k must be less than the number of rows in the DataFrame")
    # shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # use numpy array to split the dataframe
    folds = np.array_split(df_shuffled, k)
    return folds


# implement k-fold cross validation for linear regression, Ridge and Lasso models
def k_fold_cross_validation(
    df: pd.DataFrame,
    k: int,
    model: Union[LinearRegression, Lasso, Ridge],
    random_state: int = 42,
) -> float:
    """Perform k-fold cross validation for a given model."""
    if df is None:
        raise ValueError("DataFrame 'df' cannot be None")
    if k < 2:
        raise ValueError("k must be greater than 1")
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k > len(df):
        raise ValueError("k must be less than the number of rows in the DataFrame")
    if model is None:
        raise ValueError("Model cannot be None")
    # split the dataframe into k folds
    folds = split_df_k_fold(df, k, random_state=random_state)
    # initialize list to store the MSE for each fold
    fold_mse = []
    # iterate over each fold
    for i, fold in enumerate(folds):
        # get the training set by concatenating all folds except the current fold
        train_set = pd.concat([f for j, f in enumerate(folds) if j != i])
        # get the validation set
        val_set = fold
        # split the training and validation sets into X and y
        X_train = train_set.drop(columns=["target"])
        y_train = train_set["target"]
        X_val = val_set.drop(columns=["target"])
        y_val = val_set["target"]
        # standardize the data
        scalar = StandardScaler()
        X_train_scaled = scalar.fit_transform(X_train)
        X_val_scaled = scalar.transform(X_val)
        # fit the model
        model.fit(X_train_scaled, y_train)
        # make predictions
        y_pred = model.predict(X_val_scaled)
        # calculate the mean squared error
        mse = mean_squared_error(y_val, y_pred)
        # append the mse to the list
    fold_mse.append(mse)
    return float(np.mean(fold_mse))


def impute_missing_values_numeric(
    df: pd.DataFrame, null_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Impute missing values in numeric columns.
    Impute using Light Gradient Boosting Machine."""
    if null_cols is None:
        null_cols = find_null_cols(df)
    if df is None:
        raise ValueError("DataFrame 'df' cannot be None")
    for col in null_cols:
        data = df.copy()
        data["is_nan"] = data[col].isna()
        X_train = data[~data[col].isna()].drop(columns=[col, "is_nan"])
        y_train = data[~data[col].isna()][col]
        X_test = data[data[col].isna()].drop(columns=[col, "is_nan"])
        nan_indices = data[data[col].isna()].index

        model = LGBMRegressor(force_col_wise=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        df.loc[nan_indices, col] = y_pred
    return df


def impute_missing_values_ordinal(
    df: pd.DataFrame, null_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Impute missing values in ordinal columns."""
    if null_cols is None:
        null_cols = find_null_cols(df)
    if df is None:
        raise ValueError("DataFrame 'df' cannot be None")
    for col in null_cols:
        data = df.copy()
        X_train = data.loc[data[col] != -1].drop(columns=[col])
        y_train = data.loc[data[col] != -1][col]
        X_test = data.loc[data[col] == -1].drop(columns=[col])
        nan_indices = X_test.index

        model = LGBMClassifier(force_col_wise=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        df.loc[nan_indices, col] = y_pred

    return df


def plot_lasso_ridge_errors(
    lasso_mse: Dict[float, float], ridge_mse: Dict[float, float]
) -> None:
    """Plot mean squared error vs lambda for Lasso and Ridge."""
    lasso_lambdas = list(lasso_mse.keys())
    lasso_errors = list(lasso_mse.values())

    ridge_lambdas = list(ridge_mse.keys())
    ridge_errors = list(ridge_mse.values())

    plt.figure(figsize=(10, 6))
    plt.plot(lasso_lambdas, lasso_errors, label="Lasso", marker="o")
    plt.plot(ridge_lambdas, ridge_errors, label="Ridge", marker="o")
    plt.xlabel("Lambda Values (Log Scale)")
    plt.ylabel("Validation Mean Squared Error")
    plt.title("Validation Mean Squared Error vs Lambda for Lasso and Ridge")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()


def encode_nominal_features(
    df: pd.DataFrame, features: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Encode nominal features in a DataFrame."""
    one_hot_enc = OneHotEncoder(
        sparse_output=False, handle_unknown="infrequent_if_exist", drop="first"
    )
    one_hot_encoded = one_hot_enc.fit_transform(df.loc[:, features])
    one_hot_encoded_df = pd.DataFrame(
        one_hot_encoded, columns=one_hot_enc.get_feature_names_out(features)
    )
    df = pd.concat([df.drop(columns=features), one_hot_encoded_df], axis=1)
    encoded_features = one_hot_enc.get_feature_names_out(features)
    for col in encoded_features:
        df[col] = pd.to_numeric(df[col])
    return df, encoded_features


def imput_missing_values_nominal(
    df: pd.DataFrame,
    nominal_features_df: pd.DataFrame,
    encoded_features: List[str],
    null_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Impute missing values in nominal columns."""
    if null_cols is None:
        null_cols = find_null_cols(df)
    if df is None:
        raise ValueError("DataFrame 'df' cannot be None")
    for nom_col in null_cols:
        drop_cols = [col for col in encoded_features if col[: len(nom_col)] == nom_col]
        df = df.drop(columns=drop_cols)
        df = pd.concat([df, nominal_features_df[nom_col]], axis=1)
        data = df.copy()
        X_train = data.loc[data[nom_col].notna()].drop(columns=[nom_col])
        y_train = data.loc[data[nom_col].notna(), nom_col]
        X_test = data.loc[data[nom_col].isna()].drop(columns=[nom_col])
        nan_indices = X_test.index

        model = LGBMClassifier(force_col_wise=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        df.loc[nan_indices, nom_col] = y_pred
        df = encode_nominal_features(df, [nom_col])[0]
    return df
