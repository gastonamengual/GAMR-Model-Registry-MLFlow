from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression


@dataclass
class LogisticRegressionModel:
    solver: str = "lbfgs"
    max_iter: int = 900
    multi_class: str = "auto"
    random_state: int = 8778

    def train(self, X, y):
        model = LogisticRegression(
            solver=self.solver, max_iter=self.max_iter,
            multi_class=self.multi_class, random_state=self.random_state
        )
        model.fit(X, y)
        return model