import time

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.data import CATEGORICAL_COLS, NUMERIC_COLS, SEED
from src.preprocessing import build_preprocessor

MODEL_NAMES = ["Dummy", "LogReg", "kNN", "NaiveBayes", "RandomForest", "XGBoost"]
SCALED_MODELS = {"LogReg", "kNN", "NaiveBayes"}
SAMPLE_WEIGHTED_MODELS = {"NaiveBayes", "XGBoost"}

PARAM_GRIDS = {
    "Dummy": [{}],
    "LogReg": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "kNN": [{"n_neighbors": k, "weights": w} for k in (1, 5, 15) for w in ("uniform", "distance")],
    "NaiveBayes": [{"var_smoothing": v} for v in (1e-9, 1e-6, 1e-3)],
    "RandomForest": [{"max_features": m, "min_samples_leaf": n} for m in ("sqrt", 0.3) for n in (1, 3)],
    "XGBoost": [{"max_depth": d, "learning_rate": lr} for d in (6, 10) for lr in (0.1, 0.3)],
}


def make_estimator(name, params, balanced=True):
    class_weight = "balanced" if balanced else None
    if name == "Dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "LogReg":
        return LogisticRegression(max_iter=2000, class_weight=class_weight, **params)
    if name == "kNN":
        return KNeighborsClassifier(n_jobs=-1, **params)
    if name == "NaiveBayes":
        return GaussianNB(**params)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, class_weight=class_weight, n_jobs=-1,
                                      random_state=SEED, **params)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, tree_method="hist", n_jobs=-1, random_state=SEED, **params)
    raise ValueError(name)


class Model:
    def __init__(self, name, params=None, balanced=True, log=True, smote=False,
                 numeric_cols=NUMERIC_COLS, categorical_cols=CATEGORICAL_COLS):
        self.name = name
        self.balanced = balanced
        steps = [("pre", build_preprocessor(scale=name in SCALED_MODELS, log=log,
                                            numeric_cols=numeric_cols, categorical_cols=categorical_cols))]
        if smote:
            steps.append(("smote", SMOTE(random_state=SEED)))
        steps.append(("model", make_estimator(name, params or {}, balanced and not smote)))
        self.pipeline = ImbPipeline(steps) if smote else Pipeline(steps)
        self.sample_weighted = balanced and not smote and name in SAMPLE_WEIGHTED_MODELS
        self.fit_seconds = None

    def fit(self, X, y):
        start = time.perf_counter()
        if self.sample_weighted:
            self.pipeline.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))
        else:
            self.pipeline.fit(X, y)
        self.fit_seconds = time.perf_counter() - start
        return self

    @property
    def classes_(self):
        return self.pipeline.classes_

    def predict_proba(self, X):
        start = time.perf_counter()
        proba = self.pipeline.predict_proba(X)
        self.predict_ms_per_1k = (time.perf_counter() - start) / len(X) * 1e6
        return proba
