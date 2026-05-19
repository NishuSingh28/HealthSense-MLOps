import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import logger
from src.utils.exception import CustomException


class DataIngestion:
    def __init__(self):

        self.raw_data_path = "data/raw/obesity.csv"

        self.train_data_path = "data/processed/train.csv"

        self.test_data_path = "data/processed/test.csv"

    def initiate_data_ingestion(self):

        logger.info("Starting data ingestion")

        try:

            df = pd.read_csv(self.raw_data_path)

            logger.info("Dataset loaded successfully")

            os.makedirs("data/processed", exist_ok=True)

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
            )

            train_set.to_csv(
                self.train_data_path,
                index=False,
            )

            test_set.to_csv(
                self.test_data_path,
                index=False,
            )

            logger.info("Train and test data saved")

            return (
                self.train_data_path,
                self.test_data_path,
            )

        except Exception as e:

            raise CustomException(e, sys)