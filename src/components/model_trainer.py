import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("models", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Replicates notebooks/modeltraining.ipynb: compare Linear/Lasso/Ridge/
        KNN/DecisionTree/RandomForest/AdaBoost, then run an extra
        RandomizedSearchCV tuning pass on Random Forest if it comes out on top
        (as it did in the notebook, with a test R2 around 0.81-0.85).
        """
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }

            params = {
                "Linear Regression": {},
                "Lasso": {"alpha": [0.1, 1.0, 10.0], "max_iter": [5000]},
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "K-Neighbors Regressor": {"n_neighbors": [3, 5, 7, 9]},
                "Decision Tree": {
                    # NOTE: 'friedman_mse' is not a valid criterion for
                    # DecisionTreeRegressor (only for the *Classifier* /
                    # gradient-boosting splitters) on this sklearn version.
                    "criterion": ["squared_error", "absolute_error", "poisson"],
                    "max_depth": [None, 5, 10, 20],
                },
                "Random Forest Regressor": {"n_estimators": [50, 100, 200]},
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1, 0.5],
                },
            }

            logging.info("Evaluating candidate models")
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(
                f"Best model from comparison: {best_model_name} (test R2 {best_model_score:.4f})"
            )

            if best_model_score < 0.6:
                raise CustomException("No best model found with an acceptable R2 score", sys)

            # Extra tuning pass mirroring the RandomizedSearchCV exploration in the notebook.
            if best_model_name == "Random Forest Regressor":
                logging.info("Running RandomizedSearchCV tuning pass on Random Forest")
                param_dist = {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [None, 5, 10, 15, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4, 6],
                    "max_features": [1.0, "sqrt", "log2", 0.3, 0.5],
                    "bootstrap": [True, False],
                }
                random_search = RandomizedSearchCV(
                    estimator=RandomForestRegressor(random_state=42),
                    param_distributions=param_dist,
                    n_iter=20,
                    cv=3,
                    scoring="r2",
                    random_state=42,
                    n_jobs=-1,
                )
                random_search.fit(X_train, y_train)
                tuned_model = random_search.best_estimator_
                tuned_score = r2_score(y_test, tuned_model.predict(X_test))

                logging.info(
                    f"Tuned Random Forest test R2: {tuned_score:.4f} "
                    f"(best params: {random_search.best_params_})"
                )

                if tuned_score >= best_model_score:
                    best_model = tuned_model
                    best_model_score = tuned_score

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)
