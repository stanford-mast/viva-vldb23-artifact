import os
import sys

import numpy as np
import pandas as pd

from viva.udfs.inference import kmeans_model_udf

class KMeansInference:
    """
    Class for KMeans inference
    """

    def __init__(self, model: str):
        self.model_udf = self._prepare_model_udf(model)

    def _prepare_model_udf(self, model: str):
        def kmeans_fn():
            # Load serialized indexes
            serialized_indexes_path = 'data/tasti_index.bin'
            index_pd = pd.read_pickle(serialized_indexes_path)

            # Extract model's column if it exists, otherwise fail
            p_col = None
            if model in index_pd.columns:
                p_col = index_pd[model]
            else:
                print(model, 'has no associated TASTI index!')
                sys.exit(1)

            # Set up cluster centroids with dimension (n_clusters, n_features)
            # Also pull out labels
            centroids = []
            labels = []
            for row in p_col:
                centroids.append(row[0])
                labels.append(row[1])

            centroids = np.array(centroids)
            return centroids, labels

        return kmeans_model_udf(kmeans_fn)

    def _get_model_udf(self):
        return self.model_udf
