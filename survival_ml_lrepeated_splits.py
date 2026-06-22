#!/usr/bin/env python3
"""Repeated-split survival ML analysis for incremental LSR C-index.

This script pools training and validation CSV files, repeatedly splits the
pooled cohort into training and holdout sets, and compares:

- `NO_LSR`: clinicopathological variables only
- `WITH_LSR`: clinicopathological variables plus LSR

The primary endpoint is overall survival represented by a duration column
(`OS` by default) and event indicator (`OSstatus` by default). The primary
performance metric is Harrell's concordance index (C-index).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_CLINICAL_FEATURES = ["Age", "Gender", "Tumor Size", "Stage", "Tumor Site", "Histologic Grade"]
DEFAULT_MODELS = ["CoxPH", "GBS", "RSF", "SurvSVM", "XGB_Cox"]
SURVIVAL_FUNCTION_MODELS = {"CoxPH", "GBS", "RSF"}
DEFAULT_TIME_POINTS = [12.0, 36.0, 60.0]
DEFAULT_SEEDS = list(range(1, 21))


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated-split survival ML C-index analysis for incremental LSR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Directory containing train.csv and validation.csv. If this directory "
            "does not contain them directly, the script also checks data-root/LSR/."
        ),
    )
    input_group.add_argument("--train-csv", type=Path, help="Training CSV path. Must be used with --validation-csv.")
    parser.add_argument("--validation-csv", type=Path, help="Validation CSV path when --train-csv is used.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/survival_ml_lsr_cindex"))
    parser.add_argument("--duration-col", default="OS")
    parser.add_argument("--event-col", default="OSstatus")
    parser.add_argument("--id-col", default="FileName_HE")
    parser.add_argument("--lsr-col", default="LSR")
    parser.add_argument("--clinical-features", nargs="+", default=DEFAULT_CLINICAL_FEATURES)
    parser.add_argument("--time-points", nargs="+", type=float, default=DEFAULT_TIME_POINTS, help="Months.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    args = parser.parse_args()
    if args.train_csv is not None and args.validation_csv is None:
        parser.error("--validation-csv is required when --train-csv is used")
    return args


def resolve_input_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.train_csv is not None:
        return args.train_csv, args.validation_csv

    direct_train = args.data_root / "train.csv"
    direct_validation = args.data_root / "validation.csv"
    nested_train = args.data_root / "LSR" / "train.csv"
    nested_validation = args.data_root / "LSR" / "validation.csv"
    if direct_train.exists() and direct_validation.exists():
        return direct_train, direct_validation
    if nested_train.exists() and nested_validation.exists():
        return nested_train, nested_validation
    raise FileNotFoundError(
        f"Could not find train.csv and validation.csv in {args.data_root} or {args.data_root / 'LSR'}"
    )


def load_pooled_data(args: argparse.Namespace) -> pd.DataFrame:
    train_csv, validation_csv = resolve_input_paths(args)
    frames = []
    for split_name, path in [("train", train_csv), ("validation", validation_csv)]:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["source_split"] = split_name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    required = [args.duration_col, args.event_col, args.lsr_col, *args.clinical_features]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if args.id_col not in df.columns:
        df[args.id_col] = np.arange(len(df)).astype(str)

    df[args.duration_col] = pd.to_numeric(df[args.duration_col], errors="coerce")
    df[args.event_col] = pd.to_numeric(df[args.event_col], errors="coerce")
    for col in [c for c in [*args.clinical_features, args.lsr_col] if c != "Stage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df[args.duration_col].notna() & df[args.event_col].isin([0, 1]) & (df[args.duration_col] > 0)
    return df.loc[valid].reset_index(drop=True)


def survival_y(df: pd.DataFrame, duration_col: str, event_col: str) -> np.ndarray:
    try:
        from sksurv.util import Surv
    except ImportError as exc:
        raise ImportError("scikit-survival is required. Install scikit-survival to run this script.") from exc
    return Surv.from_arrays(
        event=df[event_col].astype(bool).to_numpy(),
        time=df[duration_col].astype(float).to_numpy(),
    )


def cindex(y_true: np.ndarray, risk_score: np.ndarray) -> float:
    from sksurv.metrics import concordance_index_censored

    return float(concordance_index_censored(y_true["event"], y_true["time"], np.asarray(risk_score, dtype=float))[0])


def numeric_feature_columns(df: pd.DataFrame, features: Sequence[str]) -> List[str]:
    numeric = []
    for col in features:
        if col == "Stage":
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        non_missing = df[col].notna().sum()
        if non_missing and converted.notna().sum() / non_missing >= 0.9:
            numeric.append(col)
    return numeric


def make_preprocessor(df: pd.DataFrame, features: Sequence[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_cols = numeric_feature_columns(df, features)
    categorical_cols = [c for c in features if c not in numeric_cols]
    numeric_pipe = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_pipe.append(("scaler", StandardScaler()))
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, categorical_cols),
            ("num", Pipeline(numeric_pipe), numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_model(model_name: str, seed: int) -> Any:
    if model_name == "CoxPH":
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        return CoxPHSurvivalAnalysis(alpha=1.0)
    if model_name == "GBS":
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        return GradientBoostingSurvivalAnalysis(learning_rate=0.05, n_estimators=600, max_depth=1, random_state=seed)
    if model_name == "RSF":
        from sksurv.ensemble import RandomSurvivalForest

        return RandomSurvivalForest(
            n_estimators=300,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "SurvSVM":
        from sksurv.svm import FastSurvivalSVM

        return FastSurvivalSVM(rank_ratio=1.0, alpha=1.0, max_iter=2000, random_state=seed)
    raise ValueError(f"Unsupported model: {model_name}")


def xgb_label(df: pd.DataFrame, duration_col: str, event_col: str) -> np.ndarray:
    time = df[duration_col].astype(float).to_numpy()
    event = df[event_col].astype(int).to_numpy()
    return np.where(event == 1, time, -time)


def safe_time_grid(y_train: np.ndarray, y_eval: np.ndarray, n_grid: int = 80) -> Optional[np.ndarray]:
    train_time = y_train["time"].astype(float)
    eval_time = y_eval["time"].astype(float)
    eps = 1e-6
    lo = max(float(np.percentile(train_time, 10)), float(np.min(eval_time)) + eps)
    hi = min(float(np.percentile(train_time, 90)), float(np.max(eval_time)) - eps, float(np.max(train_time)) - eps)
    if hi <= lo:
        lo = max(float(np.min(train_time)), float(np.min(eval_time))) + eps
        hi = min(float(np.max(train_time)), float(np.max(eval_time))) - eps
    if hi <= lo:
        return None
    return np.unique(np.linspace(lo, hi, n_grid))


def eval_risk(
    model_name: str,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    risk: np.ndarray,
    time_points: Sequence[float],
    model: Optional[Any],
    x_eval_t: Optional[np.ndarray],
) -> Dict[str, Any]:
    from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score

    risk = np.asarray(risk, dtype=float)
    out: Dict[str, Any] = {"cindex": cindex(y_eval, risk)}
    times = safe_time_grid(y_train, y_eval)
    if times is None:
        out["mean_auc"] = np.nan
    else:
        try:
            _, mean_auc = cumulative_dynamic_auc(y_train, y_eval, risk, times)
            out["mean_auc"] = float(mean_auc)
        except Exception:
            out["mean_auc"] = np.nan

    for time_point in time_points:
        key = f"auc_{int(time_point)}"
        try:
            auc, _ = cumulative_dynamic_auc(y_train, y_eval, risk, np.array([time_point], dtype=float))
            out[key] = float(auc[0])
        except Exception:
            out[key] = np.nan
    if model_name in SURVIVAL_FUNCTION_MODELS and model is not None and x_eval_t is not None and times is not None:
        try:
            funcs = model.predict_survival_function(x_eval_t)
            surv_mat = np.row_stack([fn(times) for fn in funcs])
            out["ibs"] = float(integrated_brier_score(y_train, y_eval, surv_mat, times))
        except Exception as exc:  # noqa: BLE001
            out["ibs"] = np.nan
            out["ibs_error"] = str(exc)
    else:
        out["ibs"] = np.nan
        if model_name not in SURVIVAL_FUNCTION_MODELS:
            out["ibs_error"] = "model does not provide individual survival functions in this implementation"
    return out


def fit_predict(
    args: argparse.Namespace,
    model_name: str,
    features: Sequence[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> Tuple[np.ndarray, Optional[Any], Optional[np.ndarray]]:
    scale_numeric = model_name in {"CoxPH", "SurvSVM"}
    pre = make_preprocessor(train_df, features, scale_numeric=scale_numeric)
    x_train = train_df[list(features)]
    x_test = test_df[list(features)]

    if model_name == "XGB_Cox":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost is required for --models XGB_Cox. Install xgboost or omit XGB_Cox.") from exc
        x_train_t = pre.fit_transform(x_train)
        x_test_t = pre.transform(x_test)
        model = xgb.XGBRegressor(
            objective="survival:cox",
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train_t, xgb_label(train_df, args.duration_col, args.event_col), verbose=False)
        return np.asarray(model.predict(x_test_t), dtype=float), None, None

    model = make_model(model_name, seed)
    x_train_t = pre.fit_transform(x_train)
    x_test_t = pre.transform(x_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train_t, survival_y(train_df, args.duration_col, args.event_col))
    return np.asarray(model.predict(x_test_t), dtype=float), model, x_test_t


def run_one_split(args: argparse.Namespace, pooled_df: pd.DataFrame, seed: int) -> List[Dict[str, Any]]:
    train_idx, test_idx = train_test_split(
        pooled_df.index,
        test_size=args.test_size,
        random_state=seed,
        stratify=pooled_df[args.event_col].astype(int).to_numpy(),
    )
    train_df = pooled_df.loc[train_idx].reset_index(drop=True)
    test_df = pooled_df.loc[test_idx].reset_index(drop=True)
    y_train = survival_y(train_df, args.duration_col, args.event_col)
    y_test = survival_y(test_df, args.duration_col, args.event_col)
    feature_sets = {
        "NO_LSR": args.clinical_features,
        "WITH_LSR": [*args.clinical_features, args.lsr_col],
    }
    rows = []
    for feature_set, features in feature_sets.items():
        for model_name in args.models:
            risk, model, x_test_t = fit_predict(args, model_name, features, train_df, test_df, seed)
            row = {
                "seed": seed,
                "feature_set": feature_set,
                "model": model_name,
                "split": "holdout",
                "n_all": int(len(pooled_df)),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "events_all": int(pooled_df[args.event_col].sum()),
                "events_train": int(train_df[args.event_col].sum()),
                "events_test": int(test_df[args.event_col].sum()),
                "features": "|".join(features),
            }
            row.update(eval_risk(model_name, y_train, y_test, risk, args.time_points, model, x_test_t))
            rows.append(row)
    return rows


def build_delta(metrics_df: pd.DataFrame, time_points: Sequence[float]) -> pd.DataFrame:
    rows = []
    metric_cols = ["cindex", "mean_auc", *[f"auc_{int(t)}" for t in time_points], "ibs"]
    for (seed, model), group in metrics_df.groupby(["seed", "model"]):
        values = group.set_index("feature_set")
        if "WITH_LSR" not in values.index or "NO_LSR" not in values.index:
            continue
        record = {"seed": seed, "model": model, "split": "holdout"}
        for col in metric_cols:
            with_lsr = values.loc["WITH_LSR", col]
            no_lsr = values.loc["NO_LSR", col]
            record[f"{col}_with_lsr"] = with_lsr
            record[f"{col}_no_lsr"] = no_lsr
            record[f"delta_{col}"] = with_lsr - no_lsr
            if col == "ibs":
                record["ibs_improvement"] = no_lsr - with_lsr
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["model", "seed"]).reset_index(drop=True)


def summarize_delta(delta_df: pd.DataFrame, time_points: Sequence[float]) -> pd.DataFrame:
    rows = []
    summary_cols = [
        "cindex_with_lsr",
        "cindex_no_lsr",
        "delta_cindex",
        "ibs_with_lsr",
        "ibs_no_lsr",
        "ibs_improvement",
        "mean_auc_with_lsr",
        "mean_auc_no_lsr",
        "delta_mean_auc",
        *[f"auc_{int(t)}_with_lsr" for t in time_points],
        *[f"auc_{int(t)}_no_lsr" for t in time_points],
        *[f"delta_auc_{int(t)}" for t in time_points],
    ]
    for model, group in delta_df.groupby("model"):
        record: Dict[str, Any] = {
            "model": model,
            "n_splits": int(len(group)),
            "positive_cindex_splits": int((group["delta_cindex"] > 0).sum()),
            "positive_cindex_rate": float((group["delta_cindex"] > 0).mean()),
            "positive_ibs_splits": int((group["ibs_improvement"] > 0).sum())
            if group["ibs_improvement"].notna().any()
            else np.nan,
        }
        for col in summary_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(values) == 0:
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            se = sd / float(np.sqrt(len(values))) if len(values) > 1 else 0.0
            record[f"{col}_mean"] = mean
            record[f"{col}_sd"] = sd
            record[f"{col}_ci95_low"] = mean - 1.96 * se
            record[f"{col}_ci95_high"] = mean + 1.96 * se
        rows.append(record)
    return pd.DataFrame(rows).sort_values("delta_cindex_mean", ascending=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled_df = load_pooled_data(args)
    all_rows = []
    for i, seed in enumerate(args.seeds, start=1):
        print(f"split {i}/{len(args.seeds)} seed={seed}")
        all_rows.extend(run_one_split(args, pooled_df, seed))

    metrics_df = pd.DataFrame(all_rows)
    delta_df = build_delta(metrics_df, args.time_points)
    summary_df = summarize_delta(delta_df, args.time_points)

    metrics_path = args.output_dir / "survival_ml_metrics.csv"
    delta_path = args.output_dir / "survival_ml_delta.csv"
    summary_path = args.output_dir / "cindex_ibs_delta_summary.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    delta_df.to_csv(delta_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    train_csv, validation_csv = resolve_input_paths(args)
    run_summary = {
        "description": "Repeated pooled-data holdout survival ML analysis comparing WITH_LSR vs NO_LSR. IBS is computed for models that generate individual survival functions.",
        "train_csv": str(train_csv),
        "validation_csv": str(validation_csv),
        "output_dir": str(args.output_dir),
        "seeds": args.seeds,
        "n_splits": len(args.seeds),
        "models": args.models,
        "survival_function_models_for_ibs": sorted(SURVIVAL_FUNCTION_MODELS),
        "test_size": args.test_size,
        "duration_col": args.duration_col,
        "event_col": args.event_col,
        "clinical_features": args.clinical_features,
        "lsr_col": args.lsr_col,
        "time_points_months": args.time_points,
        "n_all": int(len(pooled_df)),
        "events_all": int(pooled_df[args.event_col].sum()),
        "metrics_csv": str(metrics_path),
        "delta_csv": str(delta_path),
        "cindex_ibs_delta_summary_csv": str(summary_path),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    keep_cols = [
        "model",
        "n_splits",
        "cindex_with_lsr_mean",
        "cindex_no_lsr_mean",
        "delta_cindex_mean",
        "ibs_with_lsr_mean",
        "ibs_no_lsr_mean",
        "ibs_improvement_mean",
        "positive_cindex_splits",
        "positive_ibs_splits",
        "positive_cindex_rate",
    ]
    print("\nC-index and IBS delta summary WITH_LSR - NO_LSR:")
    print(summary_df[keep_cols].to_string(index=False))


if __name__ == "__main__":
    main()
