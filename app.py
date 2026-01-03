
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load data
data = pd.read_csv("creditcard.csv")

# Features and target
X = data.drop(['Class', 'Time'], axis=1)
y = data['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale Amount
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE for imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Build Keras model
model = Sequential([
    Dense(32, input_dim=X_train_res.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train_res, y_train_res, epochs=10, batch_size=256, verbose=2)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", acc)

# Save model and scaler for deployment
model.save("fraud_model.h5")
import joblib
joblib.dump(scaler, "scaler.save")
