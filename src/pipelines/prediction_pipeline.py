import pickle


class PredictionPipeline:

    def __init__(self):

        self.model_path = "artifacts/model.pkl"

        self.preprocessor_path = (
            "artifacts/preprocessor.pkl"
        )

        self.label_encoder_path = (
            "artifacts/label_encoder.pkl"
        )

    def predict(self, data):

        with open(
            self.model_path,
            "rb",
        ) as model_file:

            model = pickle.load(model_file)

        with open(
            self.preprocessor_path,
            "rb",
        ) as preprocessor_file:

            preprocessor = pickle.load(
                preprocessor_file
            )

        with open(
            self.label_encoder_path,
            "rb",
        ) as encoder_file:

            label_encoder = pickle.load(
                encoder_file
            )

        data_transformed = (
            preprocessor.transform(data)
        )

        prediction = model.predict(
            data_transformed
        )

        decoded_prediction = (
            label_encoder.inverse_transform(
                prediction
            )
        )

        return decoded_prediction