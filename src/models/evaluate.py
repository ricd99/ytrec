import numpy as np
from sklearn.neighbors import NearestNeighbors

def evaluate_model(nn: NearestNeighbors, X: np.ndarray) -> tuple[float, float]:
    distances, _ = nn.kneighbors(X)
    mean_dist = float(distances.mean())
    median_dist = float(np.median(distances))

    return mean_dist, median_dist
    