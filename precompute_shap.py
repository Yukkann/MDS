from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("MDS_model/MDS_model/processed_data.csv")
OUTPUT_PATH = Path("MDS_model/MDS_model/shap_dashboard_data.npz")
CONTINUOUS_COLS = ["GPS座標X", "GPS座標Y", "時刻(小時)", "月", "平日", "道路速限"]
SAMPLE_SIZE = 350
SEED = 42


def merge_duplicate_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate no-signal columns created by renamed signal features."""
    signal_cols = ["無號誌", "無號誌.1"]
    existing_signal_cols = [col for col in signal_cols if col in df.columns]
    if len(existing_signal_cols) <= 1:
        return df

    df = df.copy()
    df["無號誌"] = df[existing_signal_cols].max(axis=1)
    return df.drop(columns=[col for col in existing_signal_cols if col != "無號誌"])


def main() -> None:
    raw = merge_duplicate_signal_columns(pd.read_csv(DATA_PATH)).drop(
        columns=["序號"], errors="ignore"
    )
    X = raw.drop(columns=["事故嚴重度"]).fillna(0)
    y = raw["事故嚴重度"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    sample_n = min(SAMPLE_SIZE, len(X_test))
    X_sample = X_test.sample(n=sample_n, random_state=SEED)
    y_sample = y_test.loc[X_sample.index]

    preprocess = ColumnTransformer(
        transformers=[("scale", StandardScaler(), CONTINUOUS_COLS)],
        remainder="passthrough",
    )
    lr_model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    C=0.01,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    lr_model.fit(X_train, y_train)

    fitted_preprocess = lr_model.named_steps["preprocess"]
    fitted_lr = lr_model.named_steps["model"]
    passthrough_cols = [col for col in X_train.columns if col not in CONTINUOUS_COLS]
    lr_features = np.array(CONTINUOUS_COLS + passthrough_cols)
    X_train_lr = pd.DataFrame(
        fitted_preprocess.transform(X_train),
        columns=lr_features,
        index=X_train.index,
    )
    X_sample_lr = pd.DataFrame(
        fitted_preprocess.transform(X_sample),
        columns=lr_features,
        index=X_sample.index,
    )
    lr_explainer = shap.LinearExplainer(fitted_lr, X_train_lr)
    lr_values = lr_explainer.shap_values(X_sample_lr)
    if isinstance(lr_values, list):
        lr_values = lr_values[1]
    lr_base = lr_explainer.expected_value
    if isinstance(lr_base, (list, np.ndarray)):
        lr_base = np.ravel(lr_base)[-1]

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_split=10,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_explainer = shap.TreeExplainer(rf_model)
    rf_values_raw = rf_explainer.shap_values(X_sample)
    if isinstance(rf_values_raw, list):
        rf_values = rf_values_raw[1]
    elif getattr(rf_values_raw, "ndim", 0) == 3:
        rf_values = rf_values_raw[:, :, 1]
    else:
        rf_values = rf_values_raw
    rf_base = rf_explainer.expected_value
    if isinstance(rf_base, (list, np.ndarray)):
        rf_base = np.ravel(rf_base)[-1]

    fatal_positions = np.where(y_sample.to_numpy() == 1)[0]
    waterfall_idx = int(fatal_positions[0]) if len(fatal_positions) else 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        lr_features=lr_features,
        lr_values=np.asarray(lr_values),
        lr_data=X_sample_lr.to_numpy(),
        lr_base_value=float(lr_base),
        rf_features=np.array(X.columns.tolist()),
        rf_values=np.asarray(rf_values),
        rf_data=X_sample.to_numpy(),
        rf_base_value=float(rf_base),
        waterfall_idx=waterfall_idx,
    )
    print(f"Saved SHAP dashboard data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
