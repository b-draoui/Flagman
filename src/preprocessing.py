import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_COLS, NUMERIC_COLS


def build_preprocessor(scale=True, log=True, numeric_cols=NUMERIC_COLS, categorical_cols=CATEGORICAL_COLS):
    if scale:
        steps = [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))] if log else []
        numeric = Pipeline(steps + [("scale", StandardScaler())])
    else:
        numeric = "passthrough"
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric, list(numeric_cols)))
    if categorical_cols:
        one_hot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)
        transformers.append(("cat", one_hot, list(categorical_cols)))
    return ColumnTransformer(transformers)
