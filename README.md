# 🌳 CART Algorithm (Classification and Regression Trees)

This repository provides a comprehensive overview of the **CART algorithm**, one of the most popular and foundational decision tree algorithms in machine learning. CART is used for both **classification** and **regression** tasks and builds binary trees using the **Gini Index** or **Mean Squared Error** as the splitting criterion.

---

## 📘 What is CART?

**CART** stands for **Classification and Regression Trees**. It is a **supervised learning algorithm** that creates binary decision trees for:

- **Classification tasks** (predicting categories)
- **Regression tasks** (predicting continuous values)

CART forms the basis for many advanced ensemble methods such as **Random Forests** and **Gradient Boosting Trees**.

---

## 🔍 Key Features

| Feature                | Description                                               |
|------------------------|-----------------------------------------------------------|
| 🧠 Learning Type        | Supervised (Classification and Regression)               |
| 🌲 Tree Type            | Binary Tree (each node splits into two branches)         |
| 📈 Splitting Criteria   | - Gini Index (for classification) <br> - MSE (for regression) |
| 🧮 Handles Continuous?  | ✅ Yes                                                    |
| ❓ Handles Missing Data | ⚠️ Not natively (preprocessing required)                  |
| ✂️ Pruning              | ✅ Yes – Supports pre-pruning and post-pruning            |

---

## ⚙️ How CART Works

### For Classification:
1. At each node, CART evaluates all possible splits.
2. Uses the **Gini Index** to determine the best feature and threshold.
3. Builds a binary tree recursively.
4. Optionally prunes the tree to avoid overfitting.

### For Regression:
1. Uses **Mean Squared Error (MSE)** to split data.
2. Each leaf node predicts the average value of the target in that group.

---

## 🧪 Gini Index Formula (for Classification)

\[
Gini = 1 - \sum_{i=1}^{C} p_i^2
\]

Where \( p_i \) is the probability of class \( i \) at a node. A lower Gini score indicates a better split.

---

## 🛠 Real-World Applications

- 🏥 Disease diagnosis (classification)
- 📉 Stock price prediction (regression)
- 💳 Credit scoring (classification)
- 🏠 House price estimation (regression)
- 🔍 Fraud detection (classification)

---

## 🆚 CART vs Other Tree Algorithms

| Feature               | ID3           | C4.5                | CART                     |
|-----------------------|---------------|---------------------|--------------------------|
| Splitting Criterion   | Info Gain     | Gain Ratio          | Gini (Classification) / MSE (Regression) |
| Handles Continuous?   | ❌ No         | ✅ Yes              | ✅ Yes                   |
| Handles Missing?      | ❌ No         | ✅ Yes              | ⚠️ Not natively          |
| Pruning               | ❌ No         | ✅ Post-pruning     | ✅ Pre/Post-pruning      |
| Tree Structure        | Multi-way     | Multi-way           | **Binary Tree**          |

---

## 🧰 Tools & Implementations

- **Scikit-learn**: Use `DecisionTreeClassifier` or `DecisionTreeRegressor`.
- **R**: Use `rpart` package.
- **XGBoost / LightGBM**: Built on principles from CART.
- **Weka**: Provides a tree builder similar to CART.

---

## 🐍 Example (Python with Scikit-learn)

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='gini', max_depth=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
