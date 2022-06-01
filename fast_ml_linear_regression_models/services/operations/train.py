import pickle
import sys
from logging import Logger

import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor, LinearRegression, Lars, HuberRegressor, \
    RANSACRegressor

from fast_ml_linear_regression_models.services.pipeline.settings import ModelSavingSettings
from fast_ml_linear_regression_models.services.utils.file_utils import get_filepath

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from fast_ml_linear_regression_models.services.utils.logging_utils import get_logger


class TrainingOperation:
    def train(self, algorithm: str, X: pd.DataFrame, y: pd.Series, directory: str):
        logger: Logger = get_logger(log_file_directory=directory)

        if algorithm == "LinearRegression":
            model = self._train_linear_regression(X, y)
            logger.info(f"[{algorithm}] {model.get_params()}")
            self._save_model(model=model, directory=directory, algorithm=algorithm)
            return model

        if algorithm == "Lars":
            model = self._train_lars(X, y)
            logger.info(f"[{algorithm}] {model.get_params()}")
            self._save_model(model=model, directory=directory, algorithm=algorithm)
            return model

        logger.info(f"[{algorithm}] Started optimizing hyperparameters")

        match algorithm:
            case "Ridge":
                model = self._train_ridge(X, y)
            case "Lasso":
                model = self._train_lasso(X, y)
            case "LassoLars":
                raise NotImplementedError
            case "ElasticNet":
                model = self._train_elastic_net(X, y)
            case "SGDRegressor":
                model = self._train_sgd_regressor(X, y)
            case "HuberRegressor":
                model = self._train_huber_regressor(X, y)
            case "RANSACRegressor":
                model = self._train_ransac_regressor(X, y)
            case _:
                raise Exception("Invalid algorithm name")

        logger.info(f"[{algorithm}] Ended optimizing hyperparameters")
        logger.info(f"[{algorithm}] {model.get_params()}")

        self._save_model(model=model, directory=directory, algorithm=algorithm)

        return model

    def _save_model(self, model, directory: str, algorithm: str):
        model_path: str = get_filepath(directory=directory, filename=f"{algorithm}.pickle")
        pickle.dump(model, open(model_path, "wb"))

    def _train_linear_regression(self, X: pd.DataFrame, y: pd.Series):
        model = LinearRegression()
        model.fit(X, y)
        return model

    def _train_ridge(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=Ridge(),
            search_spaces={
                "alpha": Real(1e-6, 1e+6, prior="log-uniform")
            },
            n_iter=32,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_lasso(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=Lasso(),
            search_spaces={
                "alpha": Real(1e-6, 1e+6, prior="log-uniform")
            },
            n_iter=32,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_elastic_net(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=ElasticNet(),
            search_spaces={
                "alpha": Real(1e-6, 1e+6, prior="log-uniform"),
                "l1_ratio": Real(sys.float_info.epsilon, 1, prior="log-uniform")
            },
            n_iter=32,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_lars(self, X: pd.DataFrame, y: pd.Series):
        model = Lars()
        model.fit(X, y)
        return model

    def _train_lasso_lars(self, X: pd.DataFrame, y: pd.Series):
        pass

    def _train_orthogonal_matching_pursuit(self, X: pd.DataFrame, y: pd.Series):
        pass

    def _train_bayesian_regression(self, X: pd.DataFrame, y: pd.Series):
        pass

    def _train_sgd_regressor(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=SGDRegressor(),
            search_spaces={
                "loss": Categorical(["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]),
                "alpha": Real(1e-6, 1e+6, prior="log-uniform"),
                "penalty": Categorical(["l2", "l1", "elasticnet"]),
                "l1_ratio": Real(sys.float_info.epsilon, 1, prior="log-uniform")
            },
            n_iter=32,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_passive_aggressive_regressor(self, X: pd.DataFrame, y: pd.Series):
        pass

    def _train_ransac_regressor(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=RANSACRegressor(),
            search_spaces={
                "min_samples": Integer(1, X.shape[1] - 1),
                "residual_threshold": Real(1e-6, 1e+6, prior="log-uniform"),
                "max_trials": Integer(1, 1e+6, prior="log-uniform")
            },
            n_iter=32,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_huber_regressor(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=HuberRegressor(),
            search_spaces={
                "alpha": Real(1e-6, 1e+6, prior="log-uniform")
            },
            n_iter=32,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_quantile_regressor(self, X: pd.DataFrame, y: pd.Series):
        pass
