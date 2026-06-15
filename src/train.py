"""Training entrypoint: trains models across imbalance strategies and
logs each run to MLflow / DagsHub."""

import mlflow


def main():
    mlflow.set_experiment("project-06-fraud-detection")
    # TODO: load data, loop over (model, imbalance_strategy) combinations,
    # log params/metrics/artifacts to MLflow.
    pass


if __name__ == "__main__":
    main()