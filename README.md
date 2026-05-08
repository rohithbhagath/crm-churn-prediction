# 📊 CRM Customer Churn Prediction System

A machine learning-based churn prediction module integrated into a CRM platform to identify at-risk customers and enable proactive retention strategies.

---

## 🧩 Problem Statement

Many CRM clients face high customer churn, directly impacting long-term revenue. This system identifies customers likely to leave and enables data-driven engagement strategies to improve retention.

---

## 🎯 Business Objectives

- Identify customers with high churn probability
- Improve retention through targeted engagement
- Reduce revenue loss from customer drop-offs
- Analyze behavioral patterns leading to churn
- Enable data-driven marketing strategies

---

## 📁 Project Structure

```
crm-churn-prediction/
│
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_model_building.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_risk_segmentation.ipynb
│
├── churn-dashboard/
│   ├── app.py
│   ├── segmented_churn_data.csv
│   ├── best_churn_model.pkl
│   ├── scaler.pkl
│   ├── kmeans_model.pkl
│   ├── pca_model.pkl
│   └── requirements.txt
│
├── reports/
│   └── model_evaluation_report.csv
│
└── README.md
```

---

## 📦 Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Data Type | Description |
|---|---|
| Customer Profile | Age, tenure, senior citizen, dependents |
| Transaction Data | Monthly charges, total charges |
| Subscription Details | Contract type, payment method, internet service |
| Interaction Logs | Tech support, online security, streaming services |

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalance Handling | Imbalanced-learn (SMOTE) |
| Clustering | K-Means, PCA |
| Dashboard | Streamlit |
| Model Saving | Joblib |

---

## 🔬 ML Pipeline

```
Raw Data
   │
   ▼
Data Cleaning & Preprocessing
   │
   ▼
Feature Engineering (CLV, ServiceCount, ContractRiskScore...)
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
SMOTE (Handle Class Imbalance)
   │
   ▼
Model Training (Logistic Regression, Random Forest, XGBoost, SVM)
   │
   ▼
Model Evaluation (Accuracy, F1, ROC-AUC, Cross Validation)
   │
   ▼
Risk Segmentation (K-Means Clustering)
   │
   ▼
Streamlit Dashboard
```

---

## 🏗️ Models Developed

### 1. Customer Churn Classifier
Trained and compared four classification models:

| Model | Description |
|---|---|
| Logistic Regression | Baseline model |
| Random Forest | Feature importance & ensemble learning |
| XGBoost | Best performance, gradient boosting |
| SVM | Support vector boundary detection |

### 2. Retention Risk Segmentation Model
K-Means clustering segments customers into:

| Segment | Description |
|---|---|
| 🔴 High Risk | Immediate intervention required |
| 🟡 Medium Risk | Proactive engagement needed |
| 🟢 Low Risk | Maintain & upsell |

---

## 🧪 Feature Engineering

| Feature | Formula |
|---|---|
| CLV | Monthly Charges × Tenure |
| AvgMonthlySpend | Total Charges / Tenure |
| ServiceCount | Count of subscribed services |
| ContractRiskScore | Month-to-month=3, One year=2, Two year=1 |
| ElectronicCheckFlag | 1 if payment is Electronic check |
| TenureGroup | New / Growing / Mature / Loyal |
| HighSpendFlag | 1 if charges above average |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/crm-churn-prediction.git
cd crm-churn-prediction
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn \
            streamlit plotly joblib matplotlib seaborn
```

### 3. Run the Notebooks (Google Colab or Jupyter)

Run notebooks in order:

```
01 → 02 → 03 → 04 → 05 → 06
```

### 4. Launch the Dashboard

```bash
cd churn-dashboard
streamlit run app.py
```

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| 📊 Overview | KPIs, churn distribution, CLV, tenure charts |
| 🔍 Churn Analysis | Contract, internet, scatter & box plots, at-risk table |
| 🧩 Risk Segments | Segment profiles, pie charts, retention strategies |
| 🔮 Predict Churn | Live churn prediction with gauge chart & recommendations |

---

## 📈 Model Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correctly identified churners |
| Recall | Churners caught out of all actual churners |
| F1 Score | Balance of precision and recall |
| ROC-AUC | Model's ability to distinguish churn vs non-churn |
| Cross Validation | 5-Fold stratified CV for robustness |

---

## 💡 Retention Strategies by Segment

**🔴 High Risk**
- Immediate personal outreach
- Offer loyalty discounts or plan upgrades
- Assign dedicated account manager
- Priority complaint resolution

**🟡 Medium Risk**
- Re-engagement email campaigns
- Contract upgrade incentives
- Product usage tips & tutorials
- Activity monitoring & alerts

**🟢 Low Risk**
- Regular satisfaction surveys
- Referral bonus programs
- Upsell premium features
- Consistent communication

---

## 📋 Deliverables

- [x] Python Notebooks (6 notebooks)
- [x] Clean & Engineered Dataset
- [x] Trained ML Models (.pkl)
- [x] Model Evaluation Report (CSV)
- [x] Interactive Streamlit Dashboard

---

## 👤 Author

**Your Name**
- GitHub: [@rohithbhagath](https://github.com/rohithbhagath)
- LinkedIn: [Rohith Bhagath A s](https://www.linkedin.com/in/rohith-bhagath/)

---

## 📄 License

This project is licensed under the MIT License.