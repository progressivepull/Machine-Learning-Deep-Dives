# Databricks notebook source
# MAGIC %md
# MAGIC In Machine Learning, **shuffling means randomly changing the order of rows in a dataset before dividing them into training and testing groups.**
# MAGIC
# MAGIC ## Why do we shuffle?
# MAGIC Imagine you have 15 houses arranged from smallest to largest.
# MAGIC
# MAGIC Without shuffling:
# MAGIC - The first 3 houses are selected for testing.
# MAGIC - The remaining 12 houses are used for training.
# MAGIC - The testing group contains only small houses.
# MAGIC - The training group contains larger houses.
# MAGIC This creates a problem because the model is tested on a different range of house sizes than it learned from.
# MAGIC
# MAGIC With shuffling:
# MAGIC - All 15 houses are randomly mixed.
# MAGIC - Three houses are selected for testing.
# MAGIC - The remaining 12 houses are used for training.
# MAGIC - Both groups are more likely to contain a mixture of small, medium, and large houses.
# MAGIC
# MAGIC ## Why is this important?
# MAGIC Shuffling helps the model in three ways:
# MAGIC 1. Reduces bias: Prevents the original order of the dataset from unfairly influencing which records are selected.
# MAGIC 2. Improves representation: Increases the chance that training and testing datasets contain different types of examples.
# MAGIC 3. Makes evaluation more reliable: Helps measure how well the model predicts data it has never seen.
# MAGIC ## Simple analogy
# MAGIC Imagine a teacher has 15 students arranged from lowest to highest exam score.
# MAGIC
# MAGIC If the teacher selects the first three students for testing, the group contains only students with low scores.
# MAGIC
# MAGIC But if the teacher randomly mixes the students before selecting three, the testing group has a better chance of representing different performance levels.
# MAGIC ### Key takeaway
# MAGIC We shuffle data so that training and testing datasets are selected randomly rather than according to their original order.
# MAGIC
# MAGIC This helps reduce selection bias and gives a more realistic evaluation of the machine learning model.
# MAGIC
# MAGIC Shuffling does not guarantee perfectly representative groups, especially with small datasets, but it usually helps when observations are independent.

# COMMAND ----------

import pandas as pd
import random

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

print(data)

# COMMAND ----------

# Define input features (X)
X = data[["size", "bedrooms"]]

# Define target variable (y)
y = data["price"]

# COMMAND ----------

# Display X
print("\n========== X: INPUT FEATURES ==========")
print(X)

# COMMAND ----------

# Display y
print("\n========== y: TARGET VALUES ==========")
print(y)

# COMMAND ----------


def manual_train_test_split(X, y, test_size=0.2, random_state=42):

    # Get total number of records
    total_rows = len(X)

    print("Total Rows " + str(total_rows))

    # Calculate test records
    test_count = int(total_rows * test_size)

    print( "Test Count " + str(test_count) ) 

    # Create list of row positions
    indices = list(range(total_rows))

    print("Indices: " + str(indices))

    # Shuffle row positions
    rng = random.Random(random_state)
    rng.shuffle(indices)

    print("Shuffle Indices: " + str(indices))

    # Divide row positions
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    # Select training data
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]

    # Select testing data
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test



X_train, X_test, y_train, y_test = manual_train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)




# COMMAND ----------

print("\n========== X TRAIN ==========")
print(X_train)

# COMMAND ----------

print("\n========== X TEST ==========")
print(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # Explain Code

# COMMAND ----------


total_rows = 15
indices = list(range(total_rows))
print(indices)

# COMMAND ----------

# MAGIC %md
# MAGIC | Code | Explanation |
# MAGIC |---|---|
# MAGIC | `total_rows` | Total number of rows in your dataset |
# MAGIC | `range(total_rows)` | Generates numbers from `0` to `total_rows - 1` |
# MAGIC | `list()` | Converts the range into a Python list |
# MAGIC | `indices =` | Stores the resulting list in a variable named `indices` |

# COMMAND ----------

test_count = 3
print(indices)
test_indices = indices[:test_count]
print(test_indices)

# COMMAND ----------

# MAGIC %md
# MAGIC | Code | Explanation |
# MAGIC |---|---|
# MAGIC | `test_count = 3` | Specifies that 3 rows will be selected for testing. |
# MAGIC | `print(indices)` | Displays the complete list of row indices. |
# MAGIC | `indices[:test_count]` | Selects the first 3 elements from the list. |
# MAGIC | `test_indices =` | Stores the selected elements in `test_indices`. |
# MAGIC | `print(test_indices)` | Displays the selected testing indices. |