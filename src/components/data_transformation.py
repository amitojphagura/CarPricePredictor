import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

TARGET_COLUMN = "Price"


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("models", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Builds the same ColumnTransformer worked out in notebooks/modeltraining.ipynb:
        OneHotEncoder on the categorical columns, StandardScaler on the numeric ones.
        """
        try:
            numerical_columns = X.select_dtypes(exclude="object").columns.tolist()
            categorical_columns = X.select_dtypes(include="object").columns.tolist()

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
                    ("StandardScaler", StandardScaler(), numerical_columns),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(input_feature_train_df)

            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # OneHotEncoder can return a sparse matrix; densify before stacking with numpy
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            logging.info("Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
