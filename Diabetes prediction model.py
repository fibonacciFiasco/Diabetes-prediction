# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:33:32 2025

@author: diya
"""

# Diabetes Prediction using Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Dataset
data = pd.read_csv("diabetes.csv")

# Step 2: Explore Dataset
print("First 5 rows:\n", data.head())
print("\nDataset Info:")
print(data.info())
print("\nSummary statistics:")
print(data.describe())

# Step 3: Split features and target
X = data.drop("Outcome", axis=1)  # Features
y = data["Outcome"]               # Target (1: diabetic, 0: non-diabetic)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build Logistic Regression Model
model = LogisticRegression(max_iter=1000)  # increase max_iter if warning appears
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate Model
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Predict New Sample
sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  # replace with your values
prediction = model.predict(sample)
print("\nPrediction for sample:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")

# Step 9: Optional - Plotting outcome distribution
plt.hist(data['Outcome'], bins=2, edgecolor='black')
plt.title('Distribution of Outcome (0: No Diabetes, 1: Diabetes)')
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.xticks([0, 1])
plt.show()