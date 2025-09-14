import pandas as pd
from typing import Dict, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, classification_report
)
from xgboost import XGBClassifier


class MultiModelTrainer:
    """
    Train and compare multiple models on the same train/test split.
    - Handles one-hot encoding for sklearn models (LogReg, RF) on categorical columns.
    - Keeps XGBoost 'as-is' (works fine with numeric + pandas categorical if configured).
    """

    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series,
                 basket: Optional[pd.DataFrame] = None):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()
        self.basket = basket  # for exporting metadata with predictions

        self.models: Dict[str, object] = {}
        self.fitted: Dict[str, object] = {}
        self.probas: Dict[str, pd.Series] = {}
        self.preds: Dict[str, pd.Series] = {}
        self.metrics_: Optional[pd.DataFrame] = None

    # ---------- Public API ----------

    def build_models(self, names=("logreg", "rf", "xgb")) -> None:
        """Instantiate model objects based on short names."""
        built = {}
        for name in names:
            if name == "logreg":
                built[name] = Pipeline([
                    ("scaler", StandardScaler(with_mean=False)),  # with_mean=False dla sparse matrices (po one-hot)
                    ("clf", LogisticRegression(max_iter=5000, solver="lbfgs"))
                ])
            elif name == "rf":
                built[name] = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
            elif name == "xgb":
                built[name] = XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.07,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    enable_categorical=True,
                    eval_metric="logloss",
                    tree_method="hist",
                )
            else:
                raise ValueError(f"Unknown model name: {name}")
        self.models = built

    def fit_all(self) -> None:
        """Train all models; handle encoding for sklearn models automatically."""
        for name, model in self.models.items():
            Xt_train, Xt_test = self._prepare_for_model(name, self.X_train, self.X_test)
            model.fit(Xt_train, self.y_train)
            self.fitted[name] = model

            # predictions
            y_pred = model.predict(Xt_test)
            self.preds[name] = pd.Series(y_pred, index=self.y_test.index, name=f"{name}_pred")

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(Xt_test)[:, 1]
            elif hasattr(model, "decision_function"):
                # fallback: scale decision scores to [0,1] if needed
                scores = model.decision_function(Xt_test)
                y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            else:
                y_proba = pd.Series([0.0] * len(Xt_test), index=self.y_test.index)

            self.probas[name] = pd.Series(y_proba, index=self.y_test.index, name=f"{name}_proba")

    def evaluate(self) -> pd.DataFrame:
        """Compute a metrics table for all fitted models."""
        rows = []
        for name in self.fitted.keys():
            y_true = self.y_test
            y_pred = self.preds[name]
            y_proba = self.probas[name]

            row = {
                "model": name,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else None,
            }
            rows.append(row)
        self.metrics_ = pd.DataFrame(rows).sort_values(by=["f1", "roc_auc"], ascending=False).reset_index(drop=True)
        return self.metrics_

    def print_reports(self) -> None:
        """Print sklearn classification_report per model (nice for console)."""
        for name in self.fitted.keys():
            print(f"\n===== {name} report =====")
            print(classification_report(self.y_test, self.preds[name], digits=4))

    def export_metrics(self, path: str) -> pd.DataFrame:
        """Save metrics table to CSV for Power BI."""
        if self.metrics_ is None:
            self.evaluate()
        self.metrics_.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved metrics CSV → {path}")
        return self.metrics_

    def export_predictions(self, path: str, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Export predictions for a chosen model (or the best by F1 if None).
        Includes metadata (InvoiceNo, CustomerID, Country) if 'basket' provided.
        """
        if model_name is None:
            if self.metrics_ is None:
                self.evaluate()
            model_name = self.metrics_.iloc[0]["model"]  # best by sort order

        if model_name not in self.fitted:
            raise ValueError(f"Model '{model_name}' not trained. Available: {list(self.fitted.keys())}")

        y_pred = self.preds[model_name]
        y_proba = self.probas[model_name]

        if self.basket is not None:
            meta_cols = ["InvoiceNo", "CustomerID", "Country"]
            meta = self.basket.loc[self.X_test.index, [c for c in meta_cols if c in self.basket.columns]].copy()
            df_out = meta.assign(TrueReturn=self.y_test.values,
                                 PredictedReturn=y_pred.values,
                                 ProbReturn=y_proba.values)
        else:
            df_out = pd.DataFrame({
                "TrueReturn": self.y_test.values,
                "PredictedReturn": y_pred.values,
                "ProbReturn": y_proba.values
            }, index=self.y_test.index)

        df_out.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved predictions CSV → {path} (model={model_name}, rows={len(df_out)})")
        return df_out

    def get_eval_payload(self):
        """
        Convenience bundle for plotting:
        returns dict with y_test, probas (per model), preds (per model), and fitted models.
        """
        return {
            "y_test": self.y_test,
            "probas": self.probas,   # Dict[str, pd.Series]
            "preds": self.preds,     # Dict[str, pd.Series]
            "models": self.fitted,   # Dict[str, estimator]
        }

    # ---------- Internals ----------

    def _prepare_for_model(self, name: str, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        For sklearn models, one-hot encode categorical columns.
        For XGBoost, pass through as-is.
        """
        if name in ("logreg", "rf"):
            cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) in ("category", "object", "string")]
            if cat_cols:
                Xtr = pd.get_dummies(X_train, columns=cat_cols, dummy_na=False)
                Xte = pd.get_dummies(X_test,  columns=cat_cols, dummy_na=False)
                Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
            else:
                Xtr, Xte = X_train, X_test
            return Xtr, Xte


        return X_train, X_test
