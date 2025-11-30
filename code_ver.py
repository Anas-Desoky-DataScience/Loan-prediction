import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    recall_score,
    f1_score,
    precision_score
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('loan_data_set.csv')

print(df.info())
print(df.isna().sum())
print(df.duplicated().sum())

# Split into features and target
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# IMPORTANT: infer numeric / categorical columns from X (features only)
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Preprocessing
numeric_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Model
log_reg = LogisticRegression(max_iter=500)

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', log_reg)
])

# Fit
pipeline.fit(X_train, y_train)

# Predict
pred = pipeline.predict(X_test)

# Metrics
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("Precision:", precision_score(y_test, pred, pos_label='Y'))  # adjust label if needed
print("Recall:", recall_score(y_test, pred, pos_label='Y'))
print("F1 score:", f1_score(y_test, pred, pos_label='Y'))
print("\nClassification report:\n", classification_report(y_test, pred))