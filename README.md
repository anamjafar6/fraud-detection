## Project Overview

This project focuses on building an end-to-end machine learning pipeline for detecting fraudulent credit card transactions. Fraud detection is a critical problem due to its financial and reputational impact.
The dataset is highly imbalanced, with fraud cases accounting for only a very small fraction of the transactions. The primary goal is to identify frauds effectively while minimizing false alarms.

## Dataset

* **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Transactions**: 284,807
* **Features**:

  * `Time`: Seconds elapsed between each transaction and the first transaction
  * `Amount`: Transaction amount
  * `V1`–`V28`: Anonymized features generated using PCA
  * `Class`: Target variable (0 = Non-Fraud, 1 = Fraud)

## Key Challenges

* **Class imbalance**: Only 0.17% of transactions are fraudulent.
* **Anonymized features**: Most features (V1–V28) are PCA components, limiting interpretability.
* **Evaluation metrics**: Accuracy is misleading, so we focused on precision, recall, F1-score, and ROC-AUC.

## Project Pipeline

1. **Exploratory Data Analysis (EDA)**

   * Summary statistics of features
   * Fraud vs non-fraud distribution
   * Visualizations of feature distributions and correlations
   * Identification of skewness and outliers

2. **Data Preprocessing**

   * Checked and confirmed no significant missing values
   * Scaled `Amount` and `Time` using StandardScaler
   * Feature engineering: derived `hour` from `Time`
   * Train-test split

3. **Handling Class Imbalance**

   * Chose **class weights** approach instead of oversampling/undersampling to retain all data

4. **Model Training**

   * Logistic Regression (baseline)
   * Random Forest
   * XGBoost
   * LightGBM

5. **Model Evaluation**

   * Metrics: Precision, Recall, F1-score, ROC-AUC
   * Confusion matrices and classification reports
   * Threshold tuning using precision-recall trade-offs

6. **Interpretability**

   * Feature importance (XGBoost and LightGBM)
   * SHAP values for global and local explanations

7. **Final Model**

   * XGBoost selected as best model
   * Threshold optimized at 0.84 for best F1 balance
   * Final results:

     * Precision: 92.9%
     * Recall: 79.1%
     * F1: 85.4%
     * ROC-AUC: 0.97

8. **Deployment (Work in Progress)**

   * FastAPI service built to expose the trained model as an API
   * Temporary tests performed in Colab with ngrok
   * Next step: permanent deployment on a cloud platform such as Render or Hugging Face Spaces

## Results

| Model               | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.06      | 0.87   | 0.12     | 0.97    |
| Random Forest       | 0.97      | 0.70   | 0.81     | 0.92    |
| XGBoost             | 0.90      | 0.79   | 0.84     | 0.97    |
| LightGBM            | 0.92      | 0.77   | 0.84     | 0.97    |

## How to Run

1. Clone this repository:

   ```
   git clone https://github.com/<your-username>/fraud-detection.git
   cd fraud-detection
   ```
2. Install required libraries:

   ```
   pip install -r requirements.txt
   ```
3. Open the notebook and run all cells to reproduce analysis and results.

## Requirements

* Python 3.9+
* Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost, lightgbm, shap, joblib, fastapi, uvicorn

## Future Work

* Full deployment of FastAPI service on cloud (Render, Railway, or Hugging Face Spaces)
* Integration with a Streamlit dashboard for demonstration
* Experiment with unsupervised anomaly detection techniques
* Continuous monitoring and retraining to address data drift
