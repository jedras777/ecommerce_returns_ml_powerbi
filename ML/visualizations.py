import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
)

# Import trained artifacts from XG_boost.py
from XG_boost import y_test, y_proba, y_pred, model


# =========================
# 1) ROC Curve
# =========================
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# =========================
# 2) Precision–Recall Curve
# =========================
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color="purple", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.tight_layout()
plt.show()


# =========================
# 3) Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Return", "Return"],
    yticklabels=["No Return", "Return"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# =========================
# 4) Feature Importance
# =========================
xgb.plot_importance(model, max_num_features=15, importance_type="weight")
plt.title("Feature Importance (Top 15)")
plt.tight_layout()
plt.show()
