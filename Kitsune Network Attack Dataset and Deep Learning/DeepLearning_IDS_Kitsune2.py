import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pl
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tf_keras.models import Sequential, load_model, Sequential
from tf_keras.layers import Dense, Dropout
from tf_keras.utils import to_categorical


# Load the dataset
df = pd.read_csv("dataset\\merged_all.csv")
df = df.dropna()
#Drop unnecessary columns (e.g., Timestamp if not needed)
df = df.drop(columns=['Timestamp'])

#Encode categorical labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

#Split features (X) and target (y)
X = df.drop(columns=['Label']).values
y = df['Label'].values

#Normalize feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Define Deep Learning Model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(np.unique(y)), activation='softmax') #Multi-class classification
])

#Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Train the model
history = model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test))

#Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")



#Plot training history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('IDS Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Ava McCarthy Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#Confusion Matrix and Classification Report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes))

cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("AI Kitsune Dataset Confusion Matrix and Classification Report")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()