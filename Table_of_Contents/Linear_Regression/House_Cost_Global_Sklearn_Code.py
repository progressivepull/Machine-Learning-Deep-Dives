# Databricks notebook source
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1: Load dataset

# COMMAND ----------

data = pd.DataFrame({
    "size": [
        1000, 1200, 1400, 1600, 1800,
        2000, 2200, 2400, 2600, 2800,
        3000, 3200, 3400, 3600, 3800
    ],
    "bedrooms": [
        2, 2, 3, 3, 3,
        4, 4, 4, 5, 5,
        5, 5, 6, 6, 6
    ],
    "price": [
        200000, 240000, 280000, 320000, 360000,
        400000, 440000, 480000, 520000, 560000,
        600000, 640000, 680000, 720000, 760000
    ]
})

print("Training Dataset:")
print(data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2: Define input features and target

# COMMAND ----------


X = data[["size", "bedrooms"]]
y = data["price"]

# Display X
print("\n========== X: INPUT FEATURES ==========")
print(X)

# Display y
print("\n========== y: TARGET VALUES ==========")
print(y)

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 3: SPLIT TRAINING AND TEST DATA

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# COMMAND ----------

print("\n========== TRAINING FEATURES ==========")
print(X_train)

# COMMAND ----------

print("\n========== TEST FEATURES ==========")
print(X_test)

# COMMAND ----------

print("\n========== TRAINING TARGET ==========")
print(y_train)

# COMMAND ----------

print("\n========== TEST TARGET ==========")
print(y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 4: CREATE LINEAR REGRESSION MODEL
# MAGIC

# COMMAND ----------

model = LinearRegression()

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 5: TRAIN MODEL

# COMMAND ----------

model.fit(X_train, y_train)

print("\n========== MODEL TRAINED ==========")

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 6: MAKE PREDICTIONS

# COMMAND ----------

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\n========== PREDICTION RESULTS ==========")
print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 7: EVALUATE MODEL

# COMMAND ----------

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 8: SAVE MODEL

# COMMAND ----------

joblib.dump(model, "house_price_model.pkl")

print("\n========== MODEL SAVED ==========")
print("File: house_price_model.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 9: LOAD SAVED MODEL

# COMMAND ----------

loaded_model = joblib.load("house_price_model.pkl")

print("\n========== MODEL LOADED ==========")

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 10: PREDICT NEW HOUSE PRICE

# COMMAND ----------

new_house = pd.DataFrame({
    "size": [2500],
    "bedrooms": [4]
})

prediction = loaded_model.predict(new_house)

print("\n========== NEW HOUSE PREDICTION ==========")
print("House Size:", new_house["size"].iloc[0])
print("Bedrooms:", new_house["bedrooms"].iloc[0])
print("Predicted Price: ${:,.2f}".format(prediction[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC # STEP 11: PREDICT MULTIPLE HOUSES

# COMMAND ----------

new_houses = pd.DataFrame({
    "size": [1500, 2500, 3500],
    "bedrooms": [3, 4, 6]
})

new_houses["predicted_price"] = loaded_model.predict(
    new_houses
)

print("\n========== MULTIPLE HOUSE PREDICTIONS ==========")
print(new_houses)

print("\n========== PROGRAM COMPLETED ==========")