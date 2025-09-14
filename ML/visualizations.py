# src/visualizations.py
import math
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

try:
    import xgboost as xgb
except Exception:
    xgb = None


class Visualizer:
    """
    Plot multi-model ROC/PR on a single chart + per-model confusion matrices,
    and (optionally) feature importance for XGBoost.
    """

    def __init__(self,
                 y_test: pd.Series,
                 probas: Dict[str, pd.Series],
                 preds: Dict[str, pd.Series],
                 models: Optional[Dict[str, object]] = None):
        self.y_test = y_test
        self.probas = probas
        self.preds = preds
        self.models = models or {}

    # ---------- Multi-line ROC ----------
    def plot_roc(self, save_path: Optional[str] = None):
        plt.figure(figsize=(7, 6))
        for name, proba in self.probas.items():
            fpr, tpr, _ = roc_curve(self.y_test, proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — model comparison")
        plt.legend(loc="lower right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    # ---------- Multi-line PR ----------
    def plot_pr(self, save_path: Optional[str] = None):
        plt.figure(figsize=(7, 6))
        for name, proba in self.probas.items():
            precision, recall, _ = precision_recall_curve(self.y_test, proba)
            plt.plot(recall, precision, lw=2, label=name)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve — model comparison")
        plt.legend(loc="lower left")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    # ---------- Confusion matrices (one per model) ----------
    def plot_confusions(self, save_path: Optional[str] = None, per_model_files: bool = False):
        names = list(self.preds.keys())
        n = len(names)
        cols = 3 if n >= 3 else n
        rows = math.ceil(n / cols)

        if per_model_files:
            # Save each confusion matrix as a separate PNG
            for name in names:
                cm = confusion_matrix(self.y_test, self.preds[name])
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(cm, interpolation="nearest")
                ax.set_title(f"Confusion Matrix — {name}")
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(["No Return", "Return"])
                ax.set_yticklabels(["No Return", "Return"])
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                for (i, j), v in np.ndenumerate(cm):
                    ax.text(j, i, f"{v:d}", ha="center", va="center")
                plt.tight_layout()
                out = save_path.replace(".png", f"_{name}.png") if save_path else None
                if out:
                    plt.savefig(out, bbox_inches="tight")
                plt.show()
            return

        # Grid with all models together
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows))
        axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

        for idx, name in enumerate(names):
            ax = axes[idx]
            cm = confusion_matrix(self.y_test, self.preds[name])
            im = ax.imshow(cm, interpolation="nearest")
            ax.set_title(f"{name}")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["No Return", "Return"])
            ax.set_yticklabels(["No Return", "Return"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f"{v:d}", ha="center", va="center")
        # hide unused axes
        for k in range(len(names), len(axes)):
            axes[k].axis("off")

        fig.suptitle("Confusion Matrices — model comparison", y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    # ---------- Optional: feature importance for XGBoost ----------
    def plot_xgb_importance(self, model_name: str = "xgb", top_n: int = 15, save_path: Optional[str] = None):
        if xgb is None:
            print("xgboost not available.")
            return
        model = self.models.get(model_name)
        if model is None:
            print(f"Model '{model_name}' not found among fitted models.")
            return
        try:
            xgb.plot_importance(model, max_num_features=top_n, importance_type="weight")
            plt.title(f"Feature Importance (Top {top_n}) — {model_name}")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
            plt.show()
        except Exception as e:
            print(f"Could not plot XGB importance: {e}")
