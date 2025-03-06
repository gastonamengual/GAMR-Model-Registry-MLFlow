from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from app.experiment_tracking.experiment_tracker import ExperimentTracker
from mlflow.models import infer_signature


def training() -> None:
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    params = {
        "solver": "lbfgs",
        "max_iter": 900,
        "multi_class": "auto",
        "random_state": 8778,
    }
    logistic_regression = LogisticRegression(**params)
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    experiment_tracker = ExperimentTracker()

    experiment_tracker.set_experiment("MLflow Afternoon")

    with experiment_tracker.start_run():
        run_info = experiment_tracker.get_run_info()
        experiment_tracker.log_params(run_info.run_id, params)
        experiment_tracker.log_metrics(run_info.run_id, {"accuracy": accuracy})
        experiment_tracker.set_tag(
            run_info.run_id,
            "Training Info",
            "Basic logistic_regression model for iris data",
        )

        # Infer the model signature
        signature = infer_signature(X_train, logistic_regression.predict(X_train))

        # Log the model
        experiment_tracker.log_model(
            sk_model=logistic_regression,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
        )
