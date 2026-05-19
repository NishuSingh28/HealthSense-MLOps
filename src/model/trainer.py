import os
import sys
import pickle

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from src.utils.logger import logger
from src.utils.exception import CustomException


class ModelTrainer:

    def __init__(self):

        self.model_path = "artifacts/model.pkl"

    def initiate_model_training(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
    ):

        try:

            logger.info("Starting model training")

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000
                ),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(),
            }

            best_model = None
            best_accuracy = 0

            mlflow.set_experiment(
                "HealthSense_MLOps"
            )

            for model_name, model in models.items():

                with mlflow.start_run(
                    run_name=model_name
                ):

                    logger.info(
                        f"Training {model_name}"
                    )

                    model.fit(X_train, y_train)

                    predictions = model.predict(
                        X_test
                    )

                    accuracy = accuracy_score(
                        y_test,
                        predictions,
                    )

                    mlflow.log_metric(
                        "accuracy",
                        accuracy,
                    )

                    mlflow.sklearn.log_model(
                        model,
                        model_name,
                    )

                    logger.info(
                        f"{model_name} Accuracy: {accuracy}"
                    )

                    print(
                        f"{model_name}: {accuracy}"
                    )

                    if accuracy > best_accuracy:

                        best_accuracy = accuracy
                        best_model = model

            logger.info(
                f"Best model accuracy: {best_accuracy}"
            )

            os.makedirs(
                "artifacts",
                exist_ok=True,
            )

            with open(
                self.model_path,
                "wb",
            ) as file_obj:

                pickle.dump(
                    best_model,
                    file_obj,
                )

            logger.info("Best model saved")

            return best_accuracy

        except Exception as e:

            raise CustomException(e, sys)