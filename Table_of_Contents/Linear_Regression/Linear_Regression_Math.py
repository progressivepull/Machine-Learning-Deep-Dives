# Databricks notebook source
from sklearn.linear_model import LinearRegression

# COMMAND ----------

# MAGIC %md
# MAGIC # Linear Regression in Machine Learning
# MAGIC
# MAGIC ## 1. What Is Linear Regression?
# MAGIC
# MAGIC **Linear Regression is a supervised machine learning algorithm used to predict a numerical value by learning the relationship between input features and an output value.**
# MAGIC
# MAGIC For example, Linear Regression can predict a house price based on:
# MAGIC
# MAGIC - House size
# MAGIC - Number of bedrooms
# MAGIC
# MAGIC The algorithm analyzes historical data, identifies relationships, and uses those relationships to predict new values.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. How Linear Regression Works
# MAGIC
# MAGIC Imagine you have a dataset containing information about houses.
# MAGIC
# MAGIC | House Size (sq ft) | Bedrooms | Price |
# MAGIC |---|---|---|
# MAGIC | 1,000 | 2 | $200,000 |
# MAGIC | 1,500 | 3 | $300,000 |
# MAGIC | 2,000 | 4 | $400,000 |
# MAGIC | 2,500 | 4 | $500,000 |
# MAGIC | 3,000 | 5 | $600,000 |
# MAGIC
# MAGIC The machine learning model examines these examples and learns how house characteristics relate to prices.
# MAGIC
# MAGIC In this example:
# MAGIC
# MAGIC - Every additional square foot increases the price by $200.
# MAGIC - Larger houses have higher prices.
# MAGIC - House size has a direct linear relationship with price.
# MAGIC
# MAGIC The model can then predict the price of a house it has never seen.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Understanding the Line of Best Fit
# MAGIC
# MAGIC Linear Regression finds a straight line that best represents the relationship between input values and output values.
# MAGIC
# MAGIC The line is called the **Line of Best Fit**.
# MAGIC
# MAGIC ### Important Concepts
# MAGIC
# MAGIC | Concept | Explanation |
# MAGIC |---|---|
# MAGIC | Data Points | Actual observations from the dataset |
# MAGIC | Regression Line | The line representing predicted values |
# MAGIC | Prediction | The numerical value calculated by the model |
# MAGIC | Residual | Difference between actual and predicted values |
# MAGIC | Line of Best Fit | The line that minimizes the sum of squared residuals |
# MAGIC
# MAGIC ### How It Works
# MAGIC
# MAGIC 1. The algorithm examines the training data.
# MAGIC 2. It estimates a relationship between input features and the target.
# MAGIC 3. It calculates predicted values.
# MAGIC 4. It measures the difference between predictions and actual values.
# MAGIC 5. It adjusts the coefficients to minimize squared errors.
# MAGIC
# MAGIC The objective is to find the line that produces the smallest possible sum of squared prediction errors.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4. Linear Regression Formula
# MAGIC
# MAGIC For Simple Linear Regression:
# MAGIC
# MAGIC **Predicted Value = Intercept + Coefficient × Input**
# MAGIC
# MAGIC Mathematically:
# MAGIC
# MAGIC `ŷ = b₀ + b₁x`
# MAGIC
# MAGIC | Symbol | Meaning |
# MAGIC |---|---|
# MAGIC | `ŷ` | Predicted value |
# MAGIC | `b₀` | Intercept |
# MAGIC | `b₁` | Coefficient or slope |
# MAGIC | `x` | Input feature |
# MAGIC
# MAGIC ### Example: Predicting House Prices
# MAGIC
# MAGIC Suppose the model learns:
# MAGIC
# MAGIC `Price = 200 × Size`
# MAGIC
# MAGIC For a house measuring 2,500 square feet:
# MAGIC
# MAGIC `Price = 200 × 2500`
# MAGIC
# MAGIC **Predicted Price = $500,000**
# MAGIC
# MAGIC This means the model predicts that a 2,500-square-foot house will cost $500,000.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 5. Simple vs. Multiple Linear Regression
# MAGIC
# MAGIC There are two common types of Linear Regression.
# MAGIC
# MAGIC | Type | Input Features | Example |
# MAGIC |---|---|---|
# MAGIC | Simple Linear Regression | One feature | Predict price using house size |
# MAGIC | Multiple Linear Regression | Two or more features | Predict price using house size and bedrooms |
# MAGIC
# MAGIC ### Simple Linear Regression
# MAGIC
# MAGIC Uses one input feature.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC `Price = b₀ + b₁(Size)`
# MAGIC
# MAGIC ### Multiple Linear Regression
# MAGIC
# MAGIC Uses multiple input features.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC `Price = b₀ + b₁(Size) + b₂(Bedrooms)`
# MAGIC
# MAGIC Where:
# MAGIC
# MAGIC | Variable | Description |
# MAGIC |---|---|
# MAGIC | Price	| Predicted house price|
# MAGIC | b₀	| Intercept (starting value when all input features are zero) |
# MAGIC | b₁	| Coefficient representing the effect of house size on price |
# MAGIC | Size	| House size in square feet |
# MAGIC | b₂	| Coefficient representing the effect of the number of bedrooms on price |
# MAGIC | Bedrooms	| Number of bedrooms in the house |
# MAGIC
# MAGIC
# MAGIC Understanding the Formula
# MAGIC In this example, the dataset follows a simple relationship:
# MAGIC \[
# MAGIC \text{Price}=200\times\text{Size}
# MAGIC \]

# COMMAND ----------

# MAGIC %md
# MAGIC Therefore, one valid set of regression coefficients is:
# MAGIC
# MAGIC | Coefficient	| Value | 	Explanation |
# MAGIC |---|------|-----|
# MAGIC | b₀	| 0	| No intercept |
# MAGIC | b₁	| 200 |	Each additional square foot increases the predicted price by $200 |
# MAGIC | b₂	| 0 |	Bedrooms do not independently affect price in this sample dataset|
# MAGIC
# MAGIC # Example: Predict a 2,500-square-foot house with 4 bedrooms.
# MAGIC \[
# MAGIC \text{Price}=0+200(2500)+0(4)
# MAGIC \]
# MAGIC
# MAGIC \[
# MAGIC \boxed{\text{Price}=\$500,000}
# MAGIC \]
# MAGIC
# MAGIC Key takeaway: Multiple Linear Regression can use several input features to predict a numerical value. In this particular dataset, house size alone perfectly explains the price, so the bedroom coefficient can be zero.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2D Scatter Plot: House size vs. price, with bedroom counts represented by colors.

# COMMAND ----------


import pandas as pd
import matplotlib.pyplot as plt

# Create housing dataset
data = pd.DataFrame({
    "size": [1000, 1500, 2000, 2500, 3000],
    "bedrooms": [2, 3, 4, 4, 5],
    "price": [200000, 300000, 400000, 500000, 600000]
})

# Create scatter plot
plt.figure(figsize=(10, 6))

plt.scatter(
    data["size"],
    data["price"],
    color="blue",
    s=100,
    label="Actual House Prices"
)

# Add regression line
plt.plot(
    data["size"],
    200 * data["size"],
    color="red",
    label="Linear Regression Line"
)

plt.title("House Size vs House Price")
plt.xlabel("House Size (Square Feet)")
plt.ylabel("House Price ($)")
plt.legend()
plt.grid(True)

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # 3D Scatter Plot: House size, bedrooms, and price on three separate axes.

# COMMAND ----------


import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    "size": [1000, 1500, 2000, 2500, 3000],
    "bedrooms": [2, 3, 4, 4, 5],
    "price": [200000, 300000, 400000, 500000, 600000]
})

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    data["size"],
    data["bedrooms"],
    data["price"],
    color="blue",
    s=100
)

ax.set_xlabel("House Size (sq ft)")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("House Price ($)")
ax.set_title("3D Housing Dataset")

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # How to Solve for b₀, b₁, and b₂ in Multiple Linear Regression
# MAGIC
# MAGIC ## 1. Multiple Linear Regression Formula
# MAGIC
# MAGIC The Multiple Linear Regression equation is:
# MAGIC
# MAGIC **Price = b₀ + b₁(Size) + b₂(Bedrooms)**
# MAGIC
# MAGIC The goal is to calculate the three unknown coefficients: **b₀, b₁, and b₂**.
# MAGIC
# MAGIC ### Understanding the Coefficients
# MAGIC
# MAGIC | Coefficient | Description |
# MAGIC |---|---|
# MAGIC | `b₀` | Intercept — predicted price when size and bedrooms are both zero |
# MAGIC | `b₁` | Change in predicted price for each additional square foot, holding bedrooms constant |
# MAGIC | `b₂` | Change in predicted price for each additional bedroom, holding size constant |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2. Housing Dataset
# MAGIC
# MAGIC We will use the following dataset to calculate the coefficients.
# MAGIC
# MAGIC | House Size (sq ft) | Bedrooms | Price |
# MAGIC |---|---|---|
# MAGIC | 1,000 | 2 | $200,000 |
# MAGIC | 1,500 | 3 | $300,000 |
# MAGIC | 2,000 | 4 | $400,000 |
# MAGIC | 2,500 | 4 | $500,000 |
# MAGIC | 3,000 | 5 | $600,000 |
# MAGIC
# MAGIC Each row represents one equation containing the three unknown coefficients.
# MAGIC
# MAGIC ### Convert the Dataset into Equations
# MAGIC
# MAGIC **House 1:**
# MAGIC
# MAGIC `200000 = b₀ + 1000b₁ + 2b₂`
# MAGIC
# MAGIC **House 2:**
# MAGIC
# MAGIC `300000 = b₀ + 1500b₁ + 3b₂`
# MAGIC
# MAGIC **House 3:**
# MAGIC
# MAGIC `400000 = b₀ + 2000b₁ + 4b₂`
# MAGIC
# MAGIC **House 4:**
# MAGIC
# MAGIC `500000 = b₀ + 2500b₁ + 4b₂`
# MAGIC
# MAGIC **House 5:**
# MAGIC
# MAGIC `600000 = b₀ + 3000b₁ + 5b₂`
# MAGIC
# MAGIC We can solve these equations to determine the coefficients.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3. Step 1: Solve for b₁ (House Size Coefficient)
# MAGIC
# MAGIC Choose two houses with the same number of bedrooms.
# MAGIC
# MAGIC From the dataset:
# MAGIC
# MAGIC | House Size | Bedrooms | Price |
# MAGIC |---|---|---|
# MAGIC | 2,000 | 4 | $400,000 |
# MAGIC | 2,500 | 4 | $500,000 |
# MAGIC
# MAGIC ### Write the Equations
# MAGIC
# MAGIC **House 3:**
# MAGIC
# MAGIC `400000 = b₀ + 2000b₁ + 4b₂`
# MAGIC
# MAGIC **House 4:**
# MAGIC
# MAGIC `500000 = b₀ + 2500b₁ + 4b₂`
# MAGIC
# MAGIC ### Subtract the First Equation from the Second
# MAGIC
# MAGIC `500000 - 400000 = (2500 - 2000)b₁`
# MAGIC
# MAGIC Simplify:
# MAGIC
# MAGIC `100000 = 500b₁`
# MAGIC
# MAGIC ### Divide Both Sides by 500
# MAGIC
# MAGIC `b₁ = 100000 / 500`
# MAGIC
# MAGIC **Answer:**
# MAGIC
# MAGIC `b₁ = 200`
# MAGIC
# MAGIC ### Explanation
# MAGIC
# MAGIC The house size coefficient is **200**.
# MAGIC
# MAGIC This means that increasing house size by one square foot increases the predicted price by $200 when the bedroom count remains unchanged.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4. Step 2: Solve for b₂ (Bedroom Coefficient)
# MAGIC
# MAGIC Now that we know:
# MAGIC
# MAGIC `b₁ = 200`
# MAGIC
# MAGIC We can substitute this value into the equations for Houses 1 and 2.
# MAGIC
# MAGIC ### House 1
# MAGIC
# MAGIC Original equation:
# MAGIC
# MAGIC `200000 = b₀ + 1000b₁ + 2b₂`
# MAGIC
# MAGIC Substitute `b₁ = 200`:
# MAGIC
# MAGIC `200000 = b₀ + 1000(200) + 2b₂`
# MAGIC
# MAGIC Multiply:
# MAGIC
# MAGIC `200000 = b₀ + 200000 + 2b₂`
# MAGIC
# MAGIC Subtract 200000 from both sides:
# MAGIC
# MAGIC `0 = b₀ + 2b₂`
# MAGIC
# MAGIC ### House 2
# MAGIC
# MAGIC Original equation:
# MAGIC
# MAGIC `300000 = b₀ + 1500b₁ + 3b₂`
# MAGIC
# MAGIC Substitute `b₁ = 200`:
# MAGIC
# MAGIC `300000 = b₀ + 1500(200) + 3b₂`
# MAGIC
# MAGIC Multiply:
# MAGIC
# MAGIC `300000 = b₀ + 300000 + 3b₂`
# MAGIC
# MAGIC Subtract 300000 from both sides:
# MAGIC
# MAGIC `0 = b₀ + 3b₂`
# MAGIC
# MAGIC ### Subtract the Two Simplified Equations
# MAGIC
# MAGIC We now have:
# MAGIC
# MAGIC **Equation 1:**
# MAGIC
# MAGIC `0 = b₀ + 2b₂`
# MAGIC
# MAGIC **Equation 2:**
# MAGIC
# MAGIC `0 = b₀ + 3b₂`
# MAGIC
# MAGIC Subtract Equation 1 from Equation 2:
# MAGIC
# MAGIC `0 = (b₀ + 3b₂) - (b₀ + 2b₂)`
# MAGIC
# MAGIC Simplify:
# MAGIC
# MAGIC `0 = b₂`
# MAGIC
# MAGIC **Answer:**
# MAGIC
# MAGIC `b₂ = 0`
# MAGIC
# MAGIC ### Explanation
# MAGIC
# MAGIC The bedroom coefficient is **0**.
# MAGIC
# MAGIC This means that the number of bedrooms does not contribute additional predictive information to house prices in this particular dataset.
# MAGIC
# MAGIC House size alone perfectly explains the prices.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 5. Step 3: Solve for b₀ (Intercept)
# MAGIC
# MAGIC We now know:
# MAGIC
# MAGIC - `b₁ = 200`
# MAGIC - `b₂ = 0`
# MAGIC
# MAGIC Use the equation for House 1.
# MAGIC
# MAGIC ### Original Equation
# MAGIC
# MAGIC `200000 = b₀ + 1000b₁ + 2b₂`
# MAGIC
# MAGIC ### Substitute the Known Coefficients
# MAGIC
# MAGIC `200000 = b₀ + 1000(200) + 2(0)`
# MAGIC
# MAGIC ### Multiply
# MAGIC
# MAGIC `200000 = b₀ + 200000 + 0`
# MAGIC
# MAGIC ### Subtract 200000 from Both Sides
# MAGIC
# MAGIC `200000 - 200000 = b₀`
# MAGIC
# MAGIC **Answer:**
# MAGIC
# MAGIC `b₀ = 0`
# MAGIC
# MAGIC ### Explanation
# MAGIC
# MAGIC The intercept is **0**.
# MAGIC
# MAGIC In this mathematical model, a house with zero square feet and zero bedrooms would have a predicted price of zero.
# MAGIC
# MAGIC This is only a mathematical interpretation of the intercept, not a realistic housing valuation.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 6. Final Regression Coefficients
# MAGIC
# MAGIC We have successfully calculated all three coefficients.
# MAGIC
# MAGIC | Coefficient | Value | Explanation |
# MAGIC |---|---|---|
# MAGIC | `b₀` | 0 | Intercept |
# MAGIC | `b₁` | 200 | Price increases by $200 per additional square foot |
# MAGIC | `b₂` | 0 | Bedrooms have no additional effect in this dataset |
# MAGIC
# MAGIC ### Original Formula
# MAGIC
# MAGIC `Price = b₀ + b₁(Size) + b₂(Bedrooms)`
# MAGIC
# MAGIC ### Substitute the Coefficients
# MAGIC
# MAGIC `Price = 0 + 200(Size) + 0(Bedrooms)`
# MAGIC
# MAGIC ### Simplified Formula
# MAGIC
# MAGIC **Price = 200 × Size**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 7. Step 4: Test the Regression Equation
# MAGIC
# MAGIC Suppose we want to predict the price of a house with:
# MAGIC
# MAGIC - House Size: 2,500 square feet
# MAGIC - Bedrooms: 4
# MAGIC
# MAGIC ### Original Equation
# MAGIC
# MAGIC `Price = b₀ + b₁(Size) + b₂(Bedrooms)`
# MAGIC
# MAGIC ### Substitute the Values
# MAGIC
# MAGIC `Price = 0 + 200(2500) + 0(4)`
# MAGIC
# MAGIC ### Perform Multiplication
# MAGIC
# MAGIC `Price = 0 + 500000 + 0`
# MAGIC
# MAGIC ### Add the Results
# MAGIC
# MAGIC `Price = 500000`
# MAGIC
# MAGIC **Final Predicted House Price: $500,000**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 8. How Python Calculates the Coefficients Automatically
# MAGIC
# MAGIC Instead of manually solving the equations, Python can calculate the coefficients using Scikit-learn.
# MAGIC
# MAGIC ### Python Example
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

from sklearn.linear_model import LinearRegression

X = data[["size", "bedrooms"]]
y = data["price"]

model = LinearRegression()
model.fit(X, y)

print("b0:", model.intercept_)
print("b1:", model.coef_[0])
print("b2:", model.coef_[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Expected Output
# MAGIC
# MAGIC ```text
# MAGIC b0: 0
# MAGIC b1: 200
# MAGIC b2: 0
# MAGIC ```
# MAGIC
# MAGIC Small floating-point differences may occur.
# MAGIC
# MAGIC ### Explanation of Python Commands
# MAGIC
# MAGIC | Python Code | Explanation |
# MAGIC |---|---|
# MAGIC | `LinearRegression()` | Creates a Linear Regression model |
# MAGIC | `model.fit(X, y)` | Trains the model using input features and target values |
# MAGIC | `model.intercept_` | Returns the intercept, b₀ |
# MAGIC | `model.coef_[0]` | Returns the coefficient for house size, b₁ |
# MAGIC | `model.coef_[1]` | Returns the coefficient for bedrooms, b₂ |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 9. Why Does Linear Regression Calculate Coefficients?
# MAGIC
# MAGIC Linear Regression calculates coefficients to find the mathematical relationship that best explains the training data.
# MAGIC
# MAGIC For real-world datasets, the model usually cannot predict every value perfectly.
# MAGIC
# MAGIC Therefore, Linear Regression uses a mathematical method called **Ordinary Least Squares (OLS)**.
# MAGIC
# MAGIC OLS finds the coefficients that minimize the sum of squared differences between actual and predicted values.
# MAGIC
# MAGIC These differences are called **residuals**.
# MAGIC
# MAGIC ### Example
# MAGIC
# MAGIC | Actual Price | Predicted Price | Residual |
# MAGIC |---|---|---|
# MAGIC | $200,000 | $195,000 | $5,000 |
# MAGIC | $300,000 | $310,000 | -$10,000 |
# MAGIC | $400,000 | $390,000 | $10,000 |
# MAGIC
# MAGIC The model attempts to minimize the sum of squared residuals.
# MAGIC
# MAGIC In our sample housing dataset, the relationship is perfectly linear, so the coefficients can reproduce all five prices exactly.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 10. Key Takeaways
# MAGIC
# MAGIC 1. **b₀ (Intercept):** Represents the predicted starting value when all features equal zero.
# MAGIC 2. **b₁ (Size Coefficient):** Represents the change in predicted price for each additional square foot, holding bedrooms constant.
# MAGIC 3. **b₂ (Bedroom Coefficient):** Represents the change in predicted price for each additional bedroom, holding size constant.
# MAGIC 4. **Linear Regression:** Finds coefficients that minimize squared prediction errors.
# MAGIC 5. **Scikit-learn:** Automatically calculates regression coefficients using `LinearRegression()` and `model.fit()`.
# MAGIC
# MAGIC ### Final Equation
# MAGIC
# MAGIC **Price = 0 + 200(Size) + 0(Bedrooms)**
# MAGIC
# MAGIC Simplified:
# MAGIC
# MAGIC **Price = 200 × Size**
# MAGIC
# MAGIC For a 2,500-square-foot house with 4 bedrooms:
# MAGIC
# MAGIC **Predicted Price = $500,000**