"""Script to train model."""
import argparse
import json
from time import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


PARAMS_FILEPATH = Path("params.yaml")
with PARAMS_FILEPATH.open("r") as f:
    PARAMS = yaml.safe_load(f)["train"]


def _get_args(argv=None):
    """Get arguments from user input."""
    parser = argparse.ArgumentParser(
        description="Train model."
    )
    parser.add_argument(
        "-d", "--data",
        type=str,
        help="Filepath to load training data. Example: data/train.csv"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Filepath to save the trained model. Example: model.joblib",
        default="outputs/models/model.joblib"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="Filepath to save the model performance. Example: metrics.json",
        default="outputs/metrics/train_metrics.json"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Experiment name",
        default="default-experiment"
    )
    args, _ = parser.parse_known_args(argv)
    print(args)
    return args


def _check_dir(directory: Path):
    if not directory.exists():
        print(f"{directory} doesn't exist. Creating one...")
        directory.mkdir(parents=True)

    print(f"{directory} exist!")


def create_model():
    model = Pipeline(
        [
            ("estimator", RandomForestClassifier(
                max_depth=PARAMS["hyperparameters"]["max_depth"],
                n_estimators=PARAMS["hyperparameters"]["n_estimators"],
                min_samples_split=PARAMS["hyperparameters"]["min_samples_split"],
                min_samples_leaf=PARAMS["hyperparameters"]["min_samples_leaf"],
                random_state=PARAMS["seed"]))
        ]
    )
    return model


def main(argv=None):
    """Main function to train model."""
    args = _get_args(argv)
    mlflow.set_experiment(args.experiment_name)

    print("Start training model")
    start = time()

    data_filepath = Path(args.data)
    _check_dir(data_filepath.parent)

    model_filepath = Path(args.model)
    _check_dir(model_filepath.parent)

    metrics_filepath = Path(args.metrics)
    _check_dir(metrics_filepath.parent)

    data = pd.read_csv(data_filepath)
    print("Loaded training data with shape:", data.shape)

    features = data.drop(columns=PARAMS["col_to_drop"])
    features = features.select_dtypes(include="number")
    target = data["loan_status"]

    mlflow.sklearn.autolog()
    model = create_model()
    model.fit(features, target)

    joblib.dump(model, model_filepath)

    predictions = model.predict(features)
    report = classification_report(target, predictions, output_dict=True)
    with open(metrics_filepath, "w") as f:
        json.dump(report, f)

    mlflow.sklearn.autolog(disable=True)

    print(f"Done training in {time()-start:3f}s")


if __name__ == "__main__":
    main()
