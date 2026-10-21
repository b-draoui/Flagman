import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_COLS, NUMERIC_COLS


def build_preprocessor(scale=True, log=True, numeric_cols=NUMERIC_COLS, categorical_cols=CATEGORICAL_COLS):
    one_hot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)

    if scale:
        steps = []
        if log:
            steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        steps.append(("scale", StandardScaler()))
        numeric = Pipeline(steps)
    else:
        numeric = "passthrough"

    return ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", one_hot, categorical_cols),
    ])