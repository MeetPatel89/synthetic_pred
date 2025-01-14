from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import OneHotEncoder


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
