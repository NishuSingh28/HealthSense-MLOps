import os
import sys
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
)

from src.utils.logger import logger
from src.utils.exception import CustomException


class DataTransformation:

    def __init__(self):

        self.train_path = "data/processed/train.csv"

        self.test_path = "data/processed/test.csv"

        self.preprocessor_path = (
            "artifacts/preprocessor.pkl"
        )

        self.label_encoder_path = (
            "artifacts/label_encoder.pkl"
        )

    def initiate_data_transformation(self):

        try:

            logger.info("Reading train and test data")

            train_df = pd.read_csv(
                self.train_path
            )

            test_df = pd.read_csv(
                self.test_path
            )

            target_column = "NObeyesdad"

            X_train = train_df.drop(
                columns=[target_column]
            )

            X_test = test_df.drop(
                columns=[target_column]
            )

            label_encoder = LabelEncoder()

            y_train = label_encoder.fit_transform(
                train_df[target_column]
            )

            y_test = label_encoder.transform(
                test_df[target_column]
            )

            categorical_columns = (
                X_train.select_dtypes(
                    include="object"
                ).columns
            )

            numerical_columns = (
                X_train.select_dtypes(
                    exclude="object"
                ).columns
            )

            logger.info(
                "Creating preprocessing pipelines"
            )

            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                    ),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        numerical_pipeline,
                        numerical_columns,
                    ),
                    (
                        "cat_pipeline",
                        categorical_pipeline,
                        categorical_columns,
                    ),
                ]
            )

            logger.info(
                "Applying preprocessing"
            )

            X_train_transformed = (
                preprocessor.fit_transform(
                    X_train
                )
            )

            X_test_transformed = (
                preprocessor.transform(
                    X_test
                )
            )

            os.makedirs(
                "artifacts",
                exist_ok=True,
            )

            with open(
                self.preprocessor_path,
                "wb",
            ) as file_obj:

                pickle.dump(
                    preprocessor,
                    file_obj,
                )

            logger.info(
                "Preprocessor saved successfully"
            )

            with open(
                self.label_encoder_path,
                "wb",
            ) as file_obj:

                pickle.dump(
                    label_encoder,
                    file_obj,
                )

            logger.info(
                "Label encoder saved successfully"
            )

            logger.info(
                "Data transformation completed"
            )

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                preprocessor,
            )

        except Exception as e:

            raise CustomException(e, sys)