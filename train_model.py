import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os

# Create results folder if missing
os.makedirs("results", exist_ok=True)

# Load dataset
data = pd.read_csv("data/patient_data.csv")

# Display basic info
print("Dataset Loaded:")
print(data.head())

# Handle missing values
data = data.dropna()

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split data
X = data.drop('readmitted', axis=1)
y = data['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC-AUC: {roc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('results/confusion_matrix.png')
plt.close()

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
importances.head(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.savefig('results/feature_importance.png')
plt.close()

# Save model
joblib.dump(model, "results/readmission_model.pkl")

print("Model training complete. Results saved in /results folder.")
