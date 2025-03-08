from app.pipelines.pipeline import Pipeline


class PredictionPipeline(Pipeline):
    def execute(self) -> int:
        model_version = (
            self.payload.model_version
            or self.experiment_tracker.get_latest_version(self.payload.model_name)
        )
        model_uri = self.experiment_tracker.model_uri(
            model_name=self.payload.model_name, model_version=model_version
        )
        model = self.experiment_tracker.load_model(model_uri)

        X = self.payload.data.X
        y = model.predict(X)
        prediction = int(y[0])
        return prediction
