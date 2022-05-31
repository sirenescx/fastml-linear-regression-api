class ModelSavingSettings:
    configuration_filename: str = "ridge.pickle"
    metrics_filename: str = "metrics.csv"


class PredictionSettings:
    predictions_filename: str = "ridge_predictions.csv"


class DataFrameSettings:
    target_column_name: str = "target"
