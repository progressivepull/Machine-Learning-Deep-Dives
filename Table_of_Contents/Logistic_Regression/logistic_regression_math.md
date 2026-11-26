# Logistic Regression — Mathematical Explanation Step by Step

## 1. What Is Logistic Regression?

**Logistic Regression** is a supervised machine learning algorithm primarily used for **binary classification**.

- `0` = No, False, Fail, Not Spam
- `1` = Yes, True, Pass, Spam

Although the name includes *Regression*, Logistic Regression is primarily used for classification.

### Example: Predicting Student Exam Results

Suppose we want to predict whether a student will pass an exam based on hours studied.

| Hours Studied ($x$) | Passed Exam ($y$) |
|---:|---:|
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 1 |
| 5 | 1 |
| 6 | 1 |

Our objective is to estimate:

$$
P(y=1\mid x)
$$

This means: **What is the probability of passing, given the number of hours studied?**

---

## 2. Step 1 — Start With a Linear Equation

Logistic Regression begins by calculating a linear combination of the input features:

$$
\boxed{z=w_0+w_1x}
$$

| Symbol | Meaning |
|---|---|
| $z$ | Linear score |
| $w_0$ | Bias or intercept |
| $w_1$ | Weight or coefficient |
| $x$ | Input feature |

### Example

Suppose $w_0=-4$ and $w_1=1$. Then:

$$
z=-4+x
$$

If the student studies 6 hours:

$$
z=-4+6=\boxed{2}
$$

The linear score can be negative, zero, or greater than one. A probability must lie between zero and one:

$$
0\le P\le1
$$

Therefore, Logistic Regression applies a sigmoid function.

---

## 3. Step 2 — Apply the Sigmoid Function

The **sigmoid function** converts a real-valued score into a probability:

$$
\boxed{\sigma(z)=\frac{1}{1+e^{-z}}}
$$

Here, $e\approx2.71828$.

| Linear Score ($z$) | Sigmoid Output | Interpretation |
|---:|---:|---|
| -5 | 0.0067 | Very low probability |
| -2 | 0.1192 | Low probability |
| -1 | 0.2689 | Below 50% |
| 0 | 0.5000 | Exactly 50% |
| 1 | 0.7311 | Above 50% |
| 2 | 0.8808 | High probability |
| 5 | 0.9933 | Very high probability |

**Important properties:**

- As $z\to-\infty$, $\sigma(z)\to0$.
- When $z=0$, $\sigma(z)=0.5$.
- As $z\to+\infty$, $\sigma(z)\to1$.

### Worked Example

For $z=2$:

$$
P(y=1\mid x)=\frac{1}{1+e^{-2}}
=\frac{1}{1+0.1353}
\approx\boxed{0.8808}
$$

This is an **88.08% predicted probability** of passing under our illustrative model.

---

## 4. Step 3 — The Complete Logistic Regression Equation

Combining the linear score and sigmoid gives:

$$
\boxed{P(y=1\mid x)=\frac{1}{1+e^{-(w_0+w_1x)}}}
$$

For multiple features:

$$
\boxed{P(y=1\mid\mathbf{x})=\frac{1}{1+e^{-(w_0+w_1x_1+\cdots+w_nx_n)}}}
$$

For example:

- $x_1$ = Hours studied
- $x_2$ = Attendance percentage
- $x_3$ = Number of completed assignments

Then:

$$
z=w_0+w_1x_1+w_2x_2+w_3x_3
$$

Each coefficient describes the effect of its feature on the **log-odds**, holding other features constant.

---

## 5. Step 4 — Convert Probability Into a Classification

A common decision threshold for binary classification is **0.5**:

$$
\boxed{\hat y=\begin{cases}1 & \text{if }P(y=1\mid x)\ge0.5\\0 & \text{if }P(y=1\mid x)<0.5\end{cases}}
$$

Using $z=-4+x$:

| Study Hours | Linear Score ($z$) | Probability | Prediction |
|---:|---:|---:|---|
| 1 | -3 | 0.0474 | Fail (0) |
| 2 | -2 | 0.1192 | Fail (0) |
| 3 | -1 | 0.2689 | Fail (0) |
| 4 | 0 | 0.5000 | Pass (1) |
| 5 | 1 | 0.7311 | Pass (1) |
| 6 | 2 | 0.8808 | Pass (1) |

**Note:** A threshold of 0.5 is common but not mandatory. The appropriate threshold depends on the costs of false positives and false negatives.

---

## 6. Step 5 — Understand Odds and Log-Odds

**Odds** compare the probability of an event with the probability of it not occurring:

$$
\boxed{\text{Odds}=\frac{p}{1-p}}
$$

For $p=0.8$:

$$
\text{Odds}=\frac{0.8}{0.2}=\boxed{4}
$$

This represents odds of 4 to 1.

**Log-odds** are the natural logarithm of odds:

$$
\boxed{\ln\left(\frac{p}{1-p}\right)=w_0+w_1x}
$$

This is the **logit equation**.

### Interpreting a Coefficient

Suppose $w_1=0.7$. A one-unit increase in $x$, holding other features constant, multiplies the odds by:

$$
e^{0.7}\approx\boxed{2.014}
$$

The odds roughly double. **This does not mean the probability increases by 70%.**

---

## 7. Step 6 — Binary Cross-Entropy Loss

Training requires a loss function that measures the difference between predicted probabilities and actual outcomes.

Logistic Regression commonly uses **Binary Cross-Entropy**, or **Log Loss**:

$$
\boxed{L=-[y\ln(p)+(1-y)\ln(1-p)]}
$$

| Symbol | Meaning |
|---|---|
| $L$ | Loss |
| $y$ | Actual class (0 or 1) |
| $p$ | Predicted probability of class 1 |
| $\ln$ | Natural logarithm |

### Example A — A Good Prediction

Actual class $y=1$, predicted probability $p=0.9$:

$$
L=-\ln(0.9)\approx\boxed{0.1054}
$$

The loss is small.

### Example B — A Poor Prediction

Actual class $y=1$, predicted probability $p=0.1$:

$$
L=-\ln(0.1)\approx\boxed{2.3026}
$$

The loss is much larger.

| Actual Class | Predicted $p$ | Loss |
|---:|---:|---:|
| 1 | 0.9 | 0.1054 |
| 1 | 0.7 | 0.3567 |
| 1 | 0.5 | 0.6931 |
| 1 | 0.3 | 1.2040 |
| 1 | 0.1 | 2.3026 |

For $m$ training examples, average loss is:

$$
\boxed{J(w)=-\frac{1}{m}\sum_{i=1}^{m}[y_i\ln(p_i)+(1-y_i)\ln(1-p_i)]}
$$

The training objective is to minimize $J(w)$.

---

## 8. Step 7 — Gradient Descent

**Gradient Descent** updates the model coefficients to reduce loss:

$$
\boxed{w_j^{\mathrm{new}}=w_j^{\mathrm{old}}-\alpha\frac{\partial J}{\partial w_j}}
$$

Where:

- $\alpha$ is the **learning rate**.
- $\frac{\partial J}{\partial w_j}$ is the **gradient** with respect to weight $w_j$.

For unregularized Logistic Regression:

$$
\boxed{\frac{\partial J}{\partial w_j}=\frac{1}{m}\sum_{i=1}^{m}(p_i-y_i)x_{ij}}
$$

For the intercept, define $x_{i0}=1$.

### Deriving the Gradient

The sigmoid derivative is:

$$
\frac{dp}{dz}=p(1-p)
$$

The derivative of binary cross-entropy with respect to the predicted probability is:

$$
\frac{\partial L}{\partial p}=-\frac{y}{p}+\frac{1-y}{1-p}
$$

Using the chain rule:

$$
\frac{\partial L}{\partial z}
=\frac{\partial L}{\partial p}\frac{dp}{dz}
=\left(-\frac{y}{p}+\frac{1-y}{1-p}\right)p(1-p)
=\boxed{p-y}
$$

Since $\frac{\partial z}{\partial w_j}=x_j$:

$$
\boxed{\frac{\partial L}{\partial w_j}=(p-y)x_j}
$$

Averaging across examples produces the full-dataset gradient.

---

## 9. Step 8 — Complete Gradient Descent Example

We will perform **one training update** on one observation.

| Variable | Value |
|---|---:|
| Input $x$ | 2 |
| Actual label $y$ | 1 |
| Initial intercept $w_0$ | 0 |
| Initial weight $w_1$ | 0 |
| Learning rate $\alpha$ | 0.1 |

### Step 1 — Linear Score

$$
z=0+(0)(2)=\boxed{0}
$$

### Step 2 — Sigmoid Probability

$$
p=\frac{1}{1+e^0}=\boxed{0.5}
$$

### Step 3 — Error Term

$$
p-y=0.5-1=\boxed{-0.5}
$$

### Step 4 — Gradients

Weight gradient:

$$
\frac{\partial L}{\partial w_1}=(-0.5)(2)=\boxed{-1}
$$

Intercept gradient:

$$
\frac{\partial L}{\partial w_0}=\boxed{-0.5}
$$

### Step 5 — Update the Weight

$$
w_1^{\mathrm{new}}=0-0.1(-1)=\boxed{0.1}
$$

### Step 6 — Update the Intercept

$$
w_0^{\mathrm{new}}=0-0.1(-0.5)=\boxed{0.05}
$$

### Step 7 — New Prediction

$$
z=0.05+(0.1)(2)=0.25
$$

$$
p=\frac{1}{1+e^{-0.25}}\approx\boxed{0.5622}
$$

| Measurement | Before Update | After Update |
|---|---:|---:|
| Intercept | 0 | 0.05 |
| Weight | 0 | 0.1 |
| Linear score | 0 | 0.25 |
| Probability | 50.00% | 56.22% |

Because the actual class is 1, this update increased the predicted probability in the appropriate direction. One step does not guarantee global training convergence; optimization repeats across the data.

---

## 10. Logistic Regression vs. Linear Regression

| Feature | Linear Regression | Logistic Regression |
|---|---|---|
| Main purpose | Predict continuous numbers | Classify observations |
| Example | Predict house prices | Predict pass or fail |
| Output | Continuous value | Probability from 0 to 1 |
| Mathematical model | Linear equation | Linear score plus sigmoid |
| Typical loss | Mean Squared Error | Binary Cross-Entropy |
| Optimization | Analytical solution or iterative optimization | Numerical optimization |
| Coefficient meaning | Change in expected output | Change in log-odds |

---

## 11. Important Formulas to Remember

| Concept | Formula | Purpose |
|---|---|---|
| Linear score | $z=w_0+\sum_{j=1}^n w_jx_j$ | Combine inputs and weights |
| Sigmoid | $\sigma(z)=\frac{1}{1+e^{-z}}$ | Convert score to probability |
| Probability | $p=\sigma(z)$ | Predict class-1 probability |
| Log-odds | $\ln(\frac{p}{1-p})=z$ | Interpret coefficients |
| Binary Cross-Entropy | $L=-[y\ln(p)+(1-y)\ln(1-p)]$ | Measure individual error |
| Gradient | $\frac{1}{m}\sum_i(p_i-y_i)x_{ij}$ | Determine update direction |
| Gradient Descent | $w_j\leftarrow w_j-\alpha\frac{\partial J}{\partial w_j}$ | Update weights |

---

## 12. Complete Logistic Regression Workflow

```text
                START
                  |
                  v
          Input Features (X)
                  |
                  v
      Calculate Linear Equation
          z = w0 + w1*x
                  |
                  v
       Apply Sigmoid Function
        p = 1 / (1 + e^-z)
                  |
                  v
        Predicted Probability
                  |
                  v
       Compare With Actual (y)
                  |
                  v
       Calculate Cross-Entropy
                  |
                  v
          Calculate Gradient
                  |
                  v
        Update Model Weights
                  |
                  v
     Repeat Until Stopping Rule
                  |
                  v
           Trained Model
                  |
                  v
       Predict New Probability
                  |
                  v
      Apply Decision Threshold
                  |
                  v
       Final Classification
              0 or 1
```

---

## 13. Summary

| Step | Mathematical Concept | Purpose |
|---:|---|---|
| 1 | Linear equation | Combine features and weights |
| 2 | Sigmoid function | Convert score to probability |
| 3 | Decision threshold | Choose class 0 or 1 |
| 4 | Log-odds | Interpret the coefficients |
| 5 | Binary Cross-Entropy | Measure prediction error |
| 6 | Gradient | Determine how weights should change |
| 7 | Gradient Descent | Update model coefficients |
| 8 | Repeat training | Reduce average loss |
| 9 | Prediction | Classify new observations |

### Three Fundamental Equations

**1. Compute the linear score:**

$$
\boxed{z=w_0+\sum_{j=1}^nw_jx_j}
$$

**2. Convert the score into a probability:**

$$
\boxed{p=\frac{1}{1+e^{-z}}}
$$

**3. Learn weights by minimizing the average loss:**

$$
\boxed{\min_w J(w)}
$$

**Key takeaway:** Logistic Regression learns a linear relationship between features and log-odds, converts that score into a probability using the sigmoid function, and classifies observations using a decision threshold.
