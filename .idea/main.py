import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import matplotlib

data = pd.read_csv("SeoulBikeData.csv", index_col = "Date", parse_dates = True)
df_bike = pd.DataFrame(data)
print(df_bike.info())
df = df_bike.drop(columns=['Seasons','Holiday','Functioning Day'])

correlation_matrix = df.corr()
sorted_corr = correlation_matrix["Rented Bike Count"].sort_values(ascending=False)
print(sorted_corr)

from sklearn.model_selection import train_test_split

# Step 1: Convert categorical columns into dummy/one-hot encoded columns
df_encoded = pd.get_dummies(df_bike, columns=["Seasons", "Holiday", "Functioning Day"], drop_first=True)

# Step 2: Split into training and testing sets
train_set, test_set = train_test_split(df_encoded, test_size=0.2, random_state=42)

# Separate features (X) and target variable (y)
X_train = train_set.drop(columns=["Rented Bike Count"])  # Features
y_train = train_set["Rented Bike Count"]  # Target
X_test = test_set.drop(columns=["Rented Bike Count"])
y_test = test_set["Rented Bike Count"]

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=42)
# from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor(random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

print(predictions)
print(y_test.values)

y_actual = np.array(y_test)
y_predicted = np.array(predictions)

# Avoid division by zero
non_zero_indices = y_actual != 0
y_actual = y_actual[non_zero_indices]
y_predicted = y_predicted[non_zero_indices]

# Calculate MPAE
mpae = np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100
print(f"MPAE: {mpae:.2f}%")

# predictions = model.predict(test_set)
# predicted_vs_actual = pd.concat([test_set, pd.DataFrame(predictions, index=test_set.index)], axis=1)
# predicted_vs_actual.columns = ["Actual", "Predicted"]
# print(predicted_vs_actual.head())

# # Calculate the absolute errors
# errors = abs(predicted_vs_actual['Predicted'] - predicted_vs_actual['Actual'])
# # Calculate mean absolute percentage error (MAPE) and add to list
# MAPE = 100 * np.mean((errors / predicted_vs_actual['Actual']))
#
# print(MAPE)