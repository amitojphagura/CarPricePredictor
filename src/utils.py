import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict) -> dict:
    """
    Tunes each candidate model with a small GridSearchCV (where a param grid is
    given) and returns a {model_name: test_r2} report.
    """
    try:
        report = {}

        for name, model in models.items():
            param_grid = params.get(name, {})

            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            report[name] = r2_score(y_test, y_test_pred)

        return report

    except Exception as e:
        raise CustomException(e, sys)
