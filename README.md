# 💳 Loan Approval Prediction (Machine Learning Pipeline)

This project predicts whether a loan application will be **approved** or **not approved** using a machine learning model.  
It uses a clean Scikit-learn **Pipeline + ColumnTransformer** setup to handle preprocessing and modeling in one place.

---

## 📌 Project Overview

- **Goal**: Predict `Loan_Status` for each application  
- **Model**: Logistic Regression  
- **Pipeline**:
  - Automatic preprocessing for **numeric** and **categorical** features
  - Train/test split with **stratification**
  - Evaluation using multiple classification metrics

---

## 🗂 Dataset

File: `loan_data_set.csv` (or similar)

Typical columns:

- Applicant information: `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`
- Financial info: `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`
- Property info: `Property_Area`
- Target:
  - `Loan_Status` (`Y` = approved, `N` = not approved)

In the code:

- `Loan_ID` and `Loan_Status` are dropped from features (`X`)
- `Loan_Status` is used as the target (`y`)

---

## 🧠 Approach

1. **Load Data**
   - Read the CSV file with `pandas`
   - Inspect shape, columns, missing values

2. **Split Features & Target**
   - `X` = all columns except `Loan_ID`, `Loan_Status`
   - `y` = `Loan_Status`

3. **Detect Feature Types**
   - **Numeric features**: columns with `int64` / `float64`
   - **Categorical features**: columns with `object` dtype

4. **Preprocessing Pipelines**

   - Numeric pipeline:
     - `SimpleImputer(strategy="median")`

   - Categorical pipeline:
     - `SimpleImputer(strategy="most_frequent")`
     - `OneHotEncoder(handle_unknown="ignore")`

   - Combined using `ColumnTransformer`:
     - Applies numeric pipeline to numeric columns
     - Applies categorical pipeline to categorical columns

5. **Train/Test Split**
   - 80% train / 20% test
   - `stratify=y` to keep the class balance

6. **Model & Pipeline**
   - Model: `LogisticRegression(max_iter=500, random_state=42)`
   - Wrap preprocessing + model in a single `Pipeline`:
     - `("preprocess", preprocessor)`
     - `("model", log_reg)`

7. **Training & Evaluation**
   - Fit the pipeline on training data
   - Predict on test data
   - Compute:
     - Confusion matrix
     - Precision
     - Recall
     - F1-score
     - `classification_report`

---

## 🛠 Tech Stack

- **Language**: Python
- **Libraries**:
  - `pandas`
  - `numpy`
  - `scikit-learn`
    - `Pipeline`, `ColumnTransformer`
    - `SimpleImputer`, `OneHotEncoder`
    - `LogisticRegression`
    - `train_test_split`, `classification_report`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`

---

## 📊 Metrics

Typical metrics printed:

- **Confusion Matrix**
- **Precision** (how many predicted approvals are actually approved)
- **Recall** (how many actual approvals are correctly found)
- **F1-Score**
- Full **classification report** per class

You can tune the threshold or try alternative models if needed.

---

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/loan-approval-ml.git
   cd loan-approval-ml
