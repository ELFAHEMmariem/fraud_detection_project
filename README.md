Here is the English translation of your project description, perfectly formatted for your GitHub **README.md**.

---

#  Credit Card Fraud Detection

This project utilizes Machine Learning to identify fraudulent transactions from a highly imbalanced dataset. The primary objective is to maximize fraud detection (**Recall**) while minimizing false alarms.

##  Data Analysis (EDA)
* **Dataset:** Credit Card Fraud Detection (Kaggle).
* **Imbalance:** Fraudulent transactions represent only **0.17%** of all transactions.
* **Anonymization:** Features $V_1$ to $V_{28}$ are the result of a **Principal Component Analysis (PCA)** transformation, implemented to protect banking confidentiality.

## 🛠️ Technical Pipeline
To address the challenges of this specific dataset, I implemented the following pipeline:
1. **Preprocessing:** Feature scaling of transaction amounts using `StandardScaler`.
2. **Balancing:** Application of **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic fraud examples and balance the training set.
3. **Modeling:** - **XGBoost:** For its speed and high accuracy on structured/tabular data.
   - **MLP (Multi-Layer Perceptron):** A neural network approach to capture complex non-linear relationships.

##  Model Explainability (SHAP)
The use of **SHAP (SHapley Additive exPlanations)** allows us to open the "black box" of the model. We can identify which specific variables ($V_{17}$, $V_{14}$, etc.) have the most significant influence on classifying a transaction as fraudulent.

![SHAP Analysis](resultat_shap.png)

