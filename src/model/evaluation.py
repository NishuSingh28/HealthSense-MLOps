import os
import sys

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

from src.utils.logger import logger
from src.utils.exception import CustomException


class ModelEvaluation:

    def evaluate_model(
        self,
        y_test,
        predictions,
        model_name,
    ):

        try:

            logger.info(
                f"Evaluating {model_name}"
            )

            accuracy = accuracy_score(
                y_test,
                predictions,
            )

            report = classification_report(
                y_test,
                predictions,
            )

            os.makedirs(
                "reports",
                exist_ok=True,
            )

            report_path = (
                f"reports/{model_name}_report.txt"
            )

            with open(
                report_path,
                "w",
            ) as file:

                file.write(
                    f"Accuracy: {accuracy}\n\n"
                )

                file.write(report)

            logger.info(
                f"{model_name} evaluation report saved"
            )

            return accuracy

        except Exception as e:

            raise CustomException(e, sys)