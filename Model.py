
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LSTM
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

# Step 1: Data Preprocessing
# Load CCD dataset
ccd_data = pd.read_excel('CCD.xls', skiprows=1)

# Handle missing values
ccd_data.fillna(method='ffill', inplace=True)

# Split features and target variable
X = ccd_data.drop('default payment next month', axis=1)
y = ccd_data['default payment next month']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#showing bias before undersampling
plt.subplot(1, 2, 2)
y.value_counts().plot(kind="bar", color=["red", "blue"])
plt.title("Bias before undersampling")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.xticks(rotation=0)
for i, count in enumerate(y.value_counts()):
  plt.text(i, count, str(count), ha='center', va='bottom')
plt.show()

rus = RandomUnderSampler(sampling_strategy= 'majority',random_state = 42)
X_res, y_res = rus.fit_resample(X_scaled, y)

#showing bias after undersampling
plt.subplot(1, 2, 2)
y_res.value_counts().plot(kind="bar", color=["red", "blue"])
plt.title("Bias after undersampling")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.xticks(rotation=0)
for i, count in enumerate(y_res.value_counts()):
  plt.text(i, count, str(count), ha='center', va='bottom')
plt.show()
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Step 2: Model Architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),  # Add Batch Normalization layer for improved stability
    Dropout(0.4),
    Dense(16, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Step 3: Training
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=40, batch_size=128, verbose=1, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)

# Step 4: Validation
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Step 5: Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Step 6: Show results
# Accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


print(f'Accuracy: {accuracy}, Loss score: {score[0]}, Precision: {precision}, Recall: {recall}, F1-score: {f1}')
