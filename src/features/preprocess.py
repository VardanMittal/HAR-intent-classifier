import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


def remove_duplicate_columns(X):
    return X.loc[:, ~X.columns.duplicated()]

def remove_constant_columns(X_train, X_test):
    selector = VarianceThreshold()

    X_train_new = selector.fit_transform(X_train)
    X_test_new = selector.transform(X_test)

    columns = X_train.columns[selector.get_support()]

    X_train = pd.DataFrame(
        X_train_new,
        columns=columns,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        X_test_new,
        columns=columns,
        index=X_test.index
    )

    return X_train, X_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train, X_test

def preprocess_data(
    X_train,
    X_test,
    y_train,
    y_test,
    config,
):
    X_train = remove_duplicate_columns(X_train)
    X_test = remove_duplicate_columns(X_test)

    if config["preprocessing"]["remove_constant"]:
        X_train, X_test = remove_constant_columns(
            X_train,
            X_test,
        )

    if config["preprocessing"]["scale"]:
        X_train, X_test = scale_features(
            X_train,
            X_test,)

    return X_train, X_test, y_train, y_test