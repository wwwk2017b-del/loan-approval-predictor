import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/loan_data.csv")

le = LabelEncoder()
for col in ["education", "self_employed", "property_area"]:
    df[col] = le.fit_transform(df[col])

features = ["age", "income", "coapplicant_income", "loan_amount", "loan_term",
            "credit_score", "employment_years", "existing_loans", "dependents",
            "education", "self_employed", "property_area"]

X = df[features]
y = df["loan_approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    X_tr = X_train_sc if name == "Logistic Regression" else X_train
    X_te = X_test_sc if name == "Logistic Regression" else X_test
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    acc = accuracy_score(y_test, preds)
    results[name] = {"acc": acc, "preds": preds, "model": model}
    print(f"\n📊 {name}")
    print(f"   Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, preds, target_names=["Rejected","Approved"]))

# Plot 9: Confusion Matrix (Best model)
best_preds = results["Gradient Boosting"]["preds"]
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected","Approved"],
            yticklabels=["Rejected","Approved"])
plt.title("Confusion Matrix — Gradient Boosting")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs/09_confusion_matrix.png")
plt.close()
print("✅ Saved: 09_confusion_matrix.png")

# Plot 10: ROC Curve
plt.figure(figsize=(8, 6))
for name, res in results.items():
    m = res["model"]
    X_te = X_test_sc if name == "Logistic Regression" else X_test
    proba = m.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
plt.plot([0,1],[0,1],"k--")
plt.title("ROC Curve — All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/10_roc_curve.png")
plt.close()
print("✅ Saved: 10_roc_curve.png")

# Plot 11: Feature Importance
rf = results["Random Forest"]["model"]
importances = rf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color="steelblue")
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/11_feature_importance.png")
plt.close()
print("✅ Saved: 11_feature_importance.png")

# Plot 12: Model Accuracy Comparison
plt.figure(figsize=(8, 5))
names = list(results.keys())
accs = [results[n]["acc"] * 100 for n in names]
bars = plt.bar(names, accs, color=["#3498db", "#2ecc71", "#e67e22"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{acc:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/12_model_comparison.png")
plt.close()
print("✅ Saved: 12_model_comparison.png")
