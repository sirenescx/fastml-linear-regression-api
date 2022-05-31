import pandas as pd

from fast_ml_linear_regression_models.services.pipeline.settings import PredictionSettings, DataFrameSettings
from fast_ml_linear_regression_models.services.utils.file_utils import get_filepath


class PredictionOperation:
    def predict(self, X: pd.DataFrame, model, output_directory: str, algorithm: str):
        y = model.predict(X)
        X[DataFrameSettings.target_column_name] = y
        predictions_path: str = get_filepath(
            directory=output_directory,
            filename=f"{algorithm}_predictions.csv"
        )
        X.to_csv(predictions_path)
        return predictions_path
