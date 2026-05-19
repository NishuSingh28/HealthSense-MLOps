from src.data.ingestion import DataIngestion

from src.data.transformation import (
    DataTransformation,
)

from src.model.trainer import ModelTrainer


if __name__ == "__main__":

    ingestion = DataIngestion()

    ingestion.initiate_data_ingestion()

    transformation = DataTransformation()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = transformation.initiate_data_transformation()

    trainer = ModelTrainer()

    accuracy = trainer.initiate_model_training(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print(f"Best Model Accuracy: {accuracy}")