"""
Baseline model: LightGBM classifier trained on triple-barrier labels.

Save this file as: src/models.py

Install: pip install lightgbm
"""

import numpy as np
import pandas as pd
import lightgbm as lgb


def prepare_dataset(features: pd.DataFrame, labels: pd.DataFrame):
    """
    Join features (indexed by date, from features.py) with triple-barrier
    labels (indexed by t0, from labeling.py) into an aligned X, y, plus
    the t1 column CPCV needs for purging.

    Only label values +1/-1 are kept for training a binary directional
    classifier here -- 0 (timeout) rows are dropped as the simplest
    starting point. Revisit this once you have a working baseline: you
    could instead train a 3-class model, or treat timeouts as a weak
    negative signal.

    Returns
    -------
    X : pd.DataFrame of features, aligned to valid labels
    y : pd.Series, 1 = profit-take hit, 0 = stop-loss hit
    t1 : pd.Series of label-resolution dates, aligned to X -- keep this
         around, it feeds CPCV's purge/embargo logic in validation.py
    """
    joined = features.join(labels[["t1", "label"]], how="inner")
    joined = joined.dropna()
    joined = joined[joined["label"] != 0]

    y = (joined["label"] == 1).astype(int)
    t1 = joined["t1"]
    X = joined.drop(columns=["label", "t1"])

    return X, y, t1


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    params: dict = None,
):
    """
    Train a LightGBM binary classifier. Pass X_val/y_val for early
    stopping -- strongly preferred over a fixed num_boost_round, since
    the right number of trees varies a lot across CPCV folds.
    """
    default_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_sets = [train_data]
    callbacks = []

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets.append(val_data)
        callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=500,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    return model


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    n = 500
    dates = pd.bdate_range("2022-01-01", periods=n)

    features = pd.DataFrame(
        rng.normal(0, 1, (n, 8)), index=dates,
        columns=[f"feat_{i}" for i in range(8)],
    )
    labels = pd.DataFrame(
        {
            "t1": dates + pd.Timedelta(days=5),
            "label": rng.choice([1, -1, 0], size=n, p=[0.4, 0.4, 0.2]),
        },
        index=dates,
    )

    X, y, t1 = prepare_dataset(features, labels)
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model = train_lightgbm(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    print(f"Trained on {len(X_train)} rows, validated on {len(X_val)} rows")
    print(f"Prediction range: [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"Best iteration (early stopping): {model.best_iteration}")