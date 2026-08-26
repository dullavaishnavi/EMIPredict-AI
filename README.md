# 🏦 EMIPredict AI — Intelligent Financial Risk Assessment Platform

**EMIPredict AI** is a machine-learning-based FinTech platform that evaluates a loan applicant's **EMI eligibility** and predicts their **maximum affordable monthly EMI**.

The project combines classification, regression, feature engineering, exploratory data analysis, MLflow experiment tracking, and Streamlit deployment into an end-to-end financial risk assessment system.

---

## 🎯 Project Objectives

The main objectives of EMIPredict AI are:

* Predict whether a loan applicant is **Eligible** for EMI financing.
* Identify applicants who may be **Not Eligible / High Risk**.
* Estimate the applicant's **maximum affordable monthly EMI**.
* Analyze important financial factors such as income, expenses, credit score, existing EMI and savings.
* Compare multiple machine-learning algorithms.
* Track experiments using **MLflow**.
* Export the best-performing models for deployment.
* Provide an interactive **Streamlit web application**.

---

## 📊 Dataset

The project uses an EMI/financial applicant dataset containing information such as:

* Age
* Gender
* Marital status
* Education
* Monthly salary
* Employment type
* Years of employment
* Company type
* House type
* Monthly rent
* Family size
* Dependents
* School fees
* College fees
* Travel expenses
* Grocery and utility expenses
* Other monthly expenses
* Existing loans
* Current EMI amount
* Credit score
* Bank balance
* Emergency fund
* EMI scenario
* Requested loan amount
* Requested tenure

### Target Variables

**Classification target:**

```text
emi_eligibility
```

Possible outcomes include:

```text
Eligible
Not_Eligible
High_Risk
```

For the binary ML classification pipeline:

```text
Eligible       → 1
Not_Eligible   → 0
High_Risk      → 0
```

**Regression target:**

```text
max_monthly_emi
```

---

# 🧠 Machine Learning Approach

## 1. Data Loading

The notebook supports:

* Loading the real EMI dataset
* Generating a synthetic dataset when the real dataset is unavailable

---

## 2. Data Cleaning

The preprocessing pipeline performs:

* Missing-value handling
* Numeric type conversion
* Categorical value cleaning
* Duplicate removal
* Target validation
* Unknown-category handling

---

## 3. Exploratory Data Analysis

EDA includes:

* Target distribution
* Maximum EMI distribution
* Financial variable statistics
* Correlation analysis
* Credit score vs DTI analysis
* Comparison of financial characteristics across eligibility groups

---

# ⚙️ Feature Engineering

The project creates additional financial indicators, including:

### Debt-to-Income Ratio

```text
DTI = Current EMI / Monthly Salary
```

### Expense-to-Income Ratio

```text
Expense Ratio = Total Monthly Expenses / Monthly Salary
```

### EMI-to-Income Ratio

```text
EMI Ratio = Current EMI / Monthly Salary
```

### Requested Amount-to-Income Ratio

```text
Requested Amount / Monthly Salary
```

### Disposable Income

```text
Disposable Income =
Monthly Salary - Total Monthly Expenses
```

### Emergency Fund Coverage

```text
Emergency Fund / Monthly Salary
```

### Financial Risk Score

A combined indicator based on:

* Credit score
* Existing EMI burden
* Expense burden
* Emergency fund coverage

A credit-score band is also created:

```text
Poor
Fair
Good
Excellent
```

---

# 🤖 Classification Models

The following models are trained to predict EMI eligibility:

### Logistic Regression

A baseline linear classification model.

### Random Forest

An ensemble of decision trees capable of capturing nonlinear relationships.

### XGBoost

A gradient-boosted tree algorithm designed for strong predictive performance.

### Gradient Boosting

An ensemble method that sequentially improves weak learners.

---

# 📈 Classification Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

The **best classification model is selected using F1 Score**.

The final model is exported as:

```text
models/best_classification_model.joblib
```

---

# 💰 Regression Models

The regression task predicts the applicant's:

```text
Maximum Monthly EMI
```

Models used:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* Gradient Boosting Regressor

---

# 📏 Regression Evaluation

The models are evaluated using:

### MAE

Mean Absolute Error.

### RMSE

Root Mean Squared Error.

### R²

Coefficient of determination.

The **best regression model is selected using R²**.

The final model is exported as:

```text
models/best_regression_model.joblib
```

---

# 📊 MLflow Experiment Tracking

MLflow is used to track:

* Model name
* Training configuration
* Training/test dataset sizes
* Evaluation metrics
* Trained model artifacts

Separate registered model names are used for:

```text
EMIPredict_Eligibility
EMIPredict_MaxEMI
```

This makes it easier to compare experiments and manage trained models.

---

# 🚀 Model Deployment

The trained models are exported using **Joblib**.

```text
models/
├── best_classification_model.joblib
└── best_regression_model.joblib
```

These models are then loaded by the Streamlit application.

The Streamlit application provides an interactive interface where users can enter applicant information and receive:

### Output 1 — EMI Eligibility

```text
Eligible
or
Not Eligible
```

### Output 2 — Maximum EMI

```text
Predicted Maximum Monthly EMI
```

---

# 🖥️ Streamlit Application

The main Streamlit file is:

```text
app.py
```

The application loads:

```python
classification_model = joblib.load(
    "models/best_classification_model.joblib"
)

regression_model = joblib.load(
    "models/best_regression_model.joblib"
)
```

### Streamlit Main File Path

```text
app.py
```

---

# 📁 Project Structure

```text
EMIPredict-AI/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── emi_prediction_dataset.csv
│
├── models/
│   ├── best_classification_model.joblib
│   └── best_regression_model.joblib
│
├── notebooks/
│   └── EMIPredict_AI_Colab_Notebook_FIXED.ipynb
│
├── reports/
│   ├── classification_results.csv
│   └── regression_results.csv
│
└── screenshots/
    ├── eda.png
    ├── classification_results.png
    ├── regression_results.png
    └── streamlit_app.png
```

---

# 🛠️ Technology Stack

| Technology   | Purpose               |
| ------------ | --------------------- |
| Python       | Programming           |
| Pandas       | Data processing       |
| NumPy        | Numerical computation |
| Scikit-learn | Machine learning      |
| XGBoost      | Gradient boosting     |
| Matplotlib   | Visualization         |
| Seaborn      | EDA                   |
| MLflow       | Experiment tracking   |
| Joblib       | Model serialization   |
| Google Colab | Model development     |
| GitHub       | Version control       |
| Streamlit    | Deployment            |

---

# 🔄 End-to-End Workflow

```text
                 Applicant Dataset
                       │
                       ▼
                Data Loading
                       │
                       ▼
               Data Cleaning
                       │
                       ▼
                     EDA
                       │
                       ▼
              Feature Engineering
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      Classification          Regression
             │                   │
             ▼                   ▼
      4 ML Algorithms      4 ML Algorithms
             │                   │
             ▼                   ▼
        Evaluation           Evaluation
             │                   │
             └─────────┬─────────┘
                       ▼
                Best Models
                       │
                       ▼
                    Joblib
                       │
                       ▼
                  Streamlit
                       │
                       ▼
              Financial Prediction
```

---

# ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/EMIPredict-AI.git
cd EMIPredict-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

### 4. Open the application

Streamlit will provide a local URL such as:

```text
http://localhost:8501
```

---

# 📓 Google Colab

The complete ML pipeline is available in:

```text
notebooks/EMIPredict_AI_Colab_Notebook_FIXED.ipynb
```

The notebook covers:

```text
Data Loading
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Engineering
      ↓
Classification
      ↓
Regression
      ↓
MLflow
      ↓
Best Model Selection
      ↓
Joblib Export
```

---

# 📌 Future Improvements

Potential future enhancements include:

* Multi-class prediction for `Eligible`, `Not_Eligible`, and `High_Risk`
* Hyperparameter optimization
* Explainable AI using SHAP
* More advanced credit-risk features
* Real-time model monitoring
* User authentication
* Cloud deployment
* Automated ML pipelines
* Model drift monitoring
* Improved financial-risk visualization

---

# 👩‍💻 Author

**D Vaishnavi**

B.Tech — Computer Science Engineering

**Project:** EMIPredict AI
**Domain:** FinTech / Banking
**Focus:** Machine Learning, Financial Risk Assessment & AI

---

## ⭐ Project Highlights

> **EMIPredict AI transforms applicant financial information into actionable EMI eligibility and affordability predictions using machine learning.**

**Data → Feature Engineering → ML → MLflow → Best Model → Streamlit → Financial Risk Assessment**
