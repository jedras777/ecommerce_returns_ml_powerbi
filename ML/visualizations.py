import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from ML.XG_boost import y_pred_chrono
from XG_boost import y_test_chrono,y_proba_chrono, model

fpr, tpr, thresholds = roc_curve(y_test_chrono, y_proba_chrono)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()




from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test_chrono, y_proba_chrono)

plt.figure(figsize=(6,6))
plt.plot(recall, precision, color="purple", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test_chrono, y_pred_chrono)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Return","Return"], yticklabels=["No Return","Return"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

import xgboost as xgb

xgb.plot_importance(model, max_num_features=15, importance_type="weight")
plt.title("Feature Importance (Top 15)")
plt.show()
