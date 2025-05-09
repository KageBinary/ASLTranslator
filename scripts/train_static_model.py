# needed libraries
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_static_letter_model():
# Load processed training data (features and labels)
    df = pd.read_csv('data/processed/training_data_letters_MASTER.csv')

# Separate features (X) and target labels (y) 
    # ✅ Correct: Drop BOTH 'label' and 'session_id' for features 
    X = df.drop(columns=['label', 'session_id']).values
    y = df['label'].values

    print(f"✅ Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features.")

# Check expected feature size (based on known dataset specification)
    if X.shape[1] != 91:
        raise ValueError(f"❌ Error: Expected 91 features, but found {X.shape[1]} features. Check your CSV!")

# Stratify ensures class distribution is preserved in train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features to zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    print("✅ Scaler saved to models/feature_scaler.pkl")

    # Build the model
    model = Sequential([
        Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(len(np.unique(y)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Set up callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

    # Train the model
    print("\n🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Save the model
    model.save('models/letter_model.h5')
    print("\n✅ Model saved to models/letter_model.h5")

    import pickle
    with open('models/history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("✅ Training history saved to models/history.pkl")

if __name__ == "__main__":
    train_static_letter_model()

# Visually inspected. No bugs to cause any crash!
