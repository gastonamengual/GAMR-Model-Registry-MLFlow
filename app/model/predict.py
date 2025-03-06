from app.experiment_tracking.experiment_tracker import ExperimentTracker

def predict():
    model_name = "tracking-quickstart"
    experiment_tracker = ExperimentTracker()

    latest_version = experiment_tracker.get_latest_version(model_name)
    model_uri = experiment_tracker.model_uri(model_name, latest_version)
    model = experiment_tracker.load_model(model_uri)
    