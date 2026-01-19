# Import basic libraries for file handling, machine learning, image processing, etc.
import os  # For folder and path handling
import joblib  # To save and load machine learning models
import glob  # To easily find files inside folders
import cv2  # OpenCV - for reading and processing images
import pandas as pd  # For handling data in table format
import numpy as np  # For numerical operations
import tensorflow as tf  # TensorFlow - Deep learning library
import mediapipe as mp  # Mediapipe - used for face landmark detection
from sklearn.model_selection import train_test_split  # To split data into train and test sets
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For data normalization and label encoding
from sklearn.ensemble import StackingClassifier, RandomForestClassifier  # Stacking and Random Forest Classifiers
from sklearn.linear_model import LogisticRegression  # Logistic Regression for final meta learner
from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron (Neural Network) for ML
from xgboost import XGBClassifier  # XGBoost - powerful gradient boosting trees
from tensorflow.keras import layers, models  # For building the CNN model
from tensorflow.keras.utils import to_categorical  # To convert labels into one-hot encoded format

# ----------------------------------------
# CONFIGURATION
# ----------------------------------------

IMG_SIZE = 96  # Size to which every image will be resized
IMG_PATH = '../video_frames'  # Path where all face images are stored
CSV_PATH = '../data/landmark_deltas.csv'  # Path where landmark features CSV is stored
MODEL_PATH = '../models'  # Path where trained models will be saved

# Create the model folder if it does not exist already
os.makedirs(MODEL_PATH, exist_ok=True)

# ----------------------------------------
# LANDMARK MODEL (Classical Machine Learning Model)
# ----------------------------------------

# Function to train model based on landmarks (CSV)
def train_landmark_model():
    # Check if the CSV file exists
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(CSV_PATH)

    # Check if the CSV has enough data
    if df.empty or len(df) < 10:
        raise ValueError("CSV is empty or too small!")

    print(f"✅ CSV loaded with {len(df)} rows.")

    # For speed: take only a sample of maximum 1500 rows
    df = df.sample(n=min(1500, len(df)), random_state=42)

    # Split features and labels
    X = df.drop(columns=['label'])  # Features (input variables)
    y = df['label']  # Labels (target variable)

    # Encode labels (convert words like 'real' into numbers)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Normalize (standardize) the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded)

    # Initialize individual base models
    xgb = XGBClassifier(eval_metric='mlogloss', n_estimators=100, max_depth=4)  # XGBoost model
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300)  # Neural network model
    rf = RandomForestClassifier(n_estimators=80, max_depth=5)  # Random Forest model
    meta = LogisticRegression()  # Logistic regression as the meta-model

    # Create a Stacking model combining XGBoost, MLP, and Random Forest
    model = StackingClassifier(estimators=[('xgb', xgb), ('mlp', mlp), ('rf', rf)], final_estimator=meta)

    print("🚀 Training landmark model...")

    # Train the Stacking model
    model.fit(X_train, y_train)

    # Evaluate model accuracy
    score = model.score(X_test, y_test)
    print(f"🎯 Landmark Model Accuracy: {score:.4f}")

    # Return the model, scaler, and label encoder
    return model, scaler, label_encoder

# ----------------------------------------
# CNN MODEL (Deep Learning Model)
# ----------------------------------------

# Function to load all images and their labels
def load_image_data():
    X, y = [], []  # X: images, y: labels

    # Get the list of classes (folders) inside IMG_PATH
    classes = sorted(os.listdir(IMG_PATH))
    label_map = {c: i for i, c in enumerate(classes)}  # Mapping from class name to numeric label

    # Loop through each class folder
    for label in classes:
        class_dir = os.path.join(IMG_PATH, label)
        image_files = glob.glob(os.path.join(class_dir, '**', '*.jpg'), recursive=True)

        # Loop through each image
        for img_path in image_files:
            img = cv2.imread(img_path)  # Read the image
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
                X.append(img / 255.0)  # Normalize image pixels between 0 and 1
                y.append(label_map[label])  # Append label

    return np.array(X), np.array(y), label_map

# Function to build and train the CNN model
def train_cnn_model():
    # Load images and labels
    X, y, label_map = load_image_data()

    # Convert labels to one-hot encoding (needed for classification)
    y_cat = to_categorical(y, num_classes=len(label_map))

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y)

    # Define the CNN architecture
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))  # Input image size
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)  # First convolution layer
    x = layers.MaxPooling2D()(x)  # Down-sampling
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)  # Second convolution layer
    x = layers.MaxPooling2D()(x)  # Down-sampling
    x = layers.Flatten()(x)  # Flatten to 1D
    x = layers.Dense(128, activation='relu', name='feature_dense')(x)  # Fully connected layer
    outputs = layers.Dense(len(label_map), activation='softmax')(x)  # Output layer

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model (optimizer, loss, metric)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("🚀 Training CNN model...")

    # Train the model
    model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate accuracy
    acc = model.evaluate(X_test, y_test)[1]
    print(f"🎯 CNN Model Accuracy: {acc:.4f}")

    # Return the trained model and label map
    return model, label_map

# ----------------------------------------
# HYBRID MODEL (Combination of Landmarks + CNN Features)
# ----------------------------------------

# Function to train the final hybrid model
def train_combined_model(landmark_model, scaler, cnn_model, label_encoder):
    df = pd.read_csv(CSV_PATH)  # Read landmark CSV

    if df.empty:
        raise ValueError("CSV is empty!")

    X = df.drop(columns=['label'])  # Features
    y = label_encoder.transform(df['label'])  # Labels encoded

    X_scaled = scaler.transform(X)  # Standardize landmarks

    # Prepare to load images
    image_paths = []
    for label in os.listdir(IMG_PATH):
        class_dir = os.path.join(IMG_PATH, label)
        frames = glob.glob(os.path.join(class_dir, '**', '*.jpg'), recursive=True)
        for img_path in frames:
            image_paths.append(img_path)

    images = []
    matching_indices = []

    # Find matching image for each CSV row
    for i, row in df.iterrows():
        expected_name = f'frame_{i:04}.jpg'  # Example: frame_0005.jpg
        match_found = False
        for img_path in image_paths:
            if img_path.endswith(expected_name):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                    images.append(img)
                    matching_indices.append(i)
                    match_found = True
                    break
        if not match_found:
            continue

    if len(images) == 0:
        raise ValueError("No matching images found for CSV entries!")

    images = np.array(images)

    # Extract deep features from CNN (before final output layer)
    feature_model = models.Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('feature_dense').output)
    img_features = feature_model.predict(images, batch_size=32, verbose=1)

    X_matched = X_scaled[matching_indices]
    y_matched = y[matching_indices]

    # Combine landmark features + CNN features
    combined = np.hstack([X_matched, img_features])

    # Split combined data
    X_train, X_test, y_train, y_test = train_test_split(combined, y_matched, test_size=0.2, stratify=y_matched)

    # Train XGBoost model
    model = XGBClassifier(
        n_estimators=150,
        eval_metric='mlogloss',
        objective='multi:softmax',
        num_class=len(label_encoder.classes_)
    )

    print("🚀 Training hybrid model...")
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"🎯 Hybrid Model Accuracy: {acc:.4f}")

    return model

# ----------------------------------------
# MAIN EXECUTION
# ----------------------------------------

if __name__ == "__main__":
    print("🚀 Starting FULL Fast Training...")

    # Train landmark-based model
    landmark_model, scaler, label_encoder = train_landmark_model()
    joblib.dump(landmark_model, os.path.join(MODEL_PATH, 'landmark_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_PATH, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(MODEL_PATH, 'label_encoder.pkl'))

    # Train CNN-based model
    cnn_model, label_map = train_cnn_model()
    cnn_model.save(os.path.join(MODEL_PATH, 'cnn_model.keras'))

    # Train Hybrid model (landmarks + CNN features)
    hybrid_model = train_combined_model(landmark_model, scaler, cnn_model, label_encoder)
    joblib.dump(hybrid_model, os.path.join(MODEL_PATH, 'hybrid_face_model.pkl'))

    print("✅ All models trained and saved successfully!")
