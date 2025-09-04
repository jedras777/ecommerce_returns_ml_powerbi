from ML.agregation import X_train_chrono,X_test_chrono,y_test_chrono,y_train_chrono
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score



model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    enable_categorical=True
)


model.fit(X_train_chrono, y_train_chrono)

# Predykcje
y_pred_chrono = model.predict(X_test_chrono)
y_proba_chrono = model.predict_proba(X_test_chrono)[:,1]

print("ROC-AUC:", roc_auc_score(y_test_chrono, y_proba_chrono))
print(classification_report(y_test_chrono, y_pred_chrono))
