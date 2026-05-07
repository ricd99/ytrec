import mlflow
import numpy as np
from src.core.config import settings

def tune_model(df_train: np.ndarray, df_test):
    """
    Tune NearestNeighbors parameters (stub implementation).
    Since sentence embeddings are pre-trained, there are no CountVectorizer params
    to optimize. This function returns fixed values and logs with MLFlow
    Args:
        df_train: Embedding matrix from training data
        df_test: Embedding matrix from test data
    """

    params = {"n_neighbors": settings.n_neighbors_model, "metric": "cosine"}
    # mlflow.log_params(params)
    return params