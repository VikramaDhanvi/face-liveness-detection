# train_landmark_model_fixed.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

CSV_PATH = '../data/landmark_deltas.csv'
MODEL_PATH = '../models'
os.makedirs(MODEL_PATH, exist_ok=True)

def train_landmark_model_fixed():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if df.empty or len(df) < 10:
        raise ValueError("CSV is empty or too small!")

    print(f"✅ CSV loaded with {len(df)} rows.")

    df = df.sample(n=min(1500, len(df)), random_state=42)

    X = df.drop(columns=['label'])
    y = df['label']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded)

    xgb = XGBClassifier(eval_metric='mlogloss', n_estimators=100, max_depth=4)
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300)
    rf = RandomForestClassifier(n_estimators=80, max_depth=5)
    meta = LogisticRegression()

    model = StackingClassifier(estimators=[('xgb', xgb), ('mlp', mlp), ('rf', rf)], final_estimator=meta)

    print("🚀 Training fixed landmark model...")
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"🎯 Landmark Model Accuracy: {score:.4f}")

    # Save model, scaler, and encoder+mapping
    joblib.dump(model, os.path.join(MODEL_PATH, 'landmark_model_fixed.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_PATH, 'scaler_fixed.pkl'))
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    joblib.dump({'encoder': label_encoder, 'mapping': label_mapping}, os.path.join(MODEL_PATH, 'label_encoder_fixed.pkl'))

if __name__ == "__main__":
    train_landmark_model_fixed()
