from ML.agregation import X_train, X_test, y_train, y_test, basket
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

# =========================
# 1) Model definition
# =========================
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    enable_categorical=True,
)

# =========================
# 2) Train
# =========================
model.fit(X_train, y_train)

# =========================
# 3) Predict
# =========================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# =========================
# 4) Metrics
# =========================
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# =========================
# 5) Export predictions (test set only)
# =========================
# Align metadata to the exact test rows (same index as X_test)
test_meta = basket.loc[X_test.index, ["InvoiceNo", "CustomerID", "Country"]].copy()

# Merge into a single DataFrame
df_out = test_meta.assign(
    TrueReturn=y_test.values,
    PredictedReturn=y_pred,
    ProbReturn=y_proba,
)

df_out.to_csv(r"C:\Users\jendr\PycharmProjects\UCI_online_retail_ml\outputs\predictions.csv", index=False, encoding="utf-8-sig")
print("Saved predictions.csv:", df_out.shape)
