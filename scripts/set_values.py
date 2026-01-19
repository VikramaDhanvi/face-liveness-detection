# train_xgboost_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib

# Load your collected data
df = pd.read_csv('../collected/collected_results.csv')

# Only use tilt_change, ear_change, mar_change, avg_movement as features
X = df[['tilt_change', 'ear_change', 'mar_change', 'avg_movement']]
y = df['real_label']  # Target (0: fake, 1: real)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Classifier
model = xgb.XGBClassifier(
    max_depth=4,
    n_estimators=100,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Train
model.fit(X_train, y_train)

# Predict on test
y_pred = model.predict(X_test)

# Print Evaluation
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🧠 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\n🎯 Overall Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Save trained model
joblib.dump(model, '../models/xgboost_liveness_model.pkl')
print("\n✅ Trained XGBoost model saved at '../models/xgboost_liveness_model.pkl'.")

# Bonus: Feature Importance
import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.show()
