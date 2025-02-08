# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Set style for plots
sns.set(style="whitegrid")

# Load Dataset (Update file path accordingly)
df = pd.read_csv("Women_cancer_statistics_in_Iran_2022.csv")

# Display first few rows
print("Dataset Preview:\n", df.head())

# Rename columns for readability
df.columns = ['Cancer', 'Incidents_Number', 'ASR_Incidents', 'Crude_Rate_Incidents', 
              'Cum_Risk_Incidents', 'Mortality_Number', 'ASR_Mortality', 
              'Crude_Rate_Mortality', 'Cum_Risk_Mortality']

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Convert necessary columns to numeric (handling errors)
numeric_cols = ['Incidents_Number', 'ASR_Incidents', 'Crude_Rate_Incidents', 'Cum_Risk_Incidents',
                'Mortality_Number', 'ASR_Mortality', 'Crude_Rate_Mortality', 'Cum_Risk_Mortality']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Cancer', y='Incidents_Number', palette='viridis')
plt.title('Cancer Incidents by Type in Women (2022)')
plt.xticks(rotation=90)
plt.xlabel('Cancer Type')
plt.ylabel('Number of Incidents')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Cancer', y='Mortality_Number', palette='magma')
plt.title('Cancer Mortality by Type in Women (2022)')
plt.xticks(rotation=90)
plt.xlabel('Cancer Type')
plt.ylabel('Number of Mortality')
plt.show()

# Correlation Matrix
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Cancer Statistics')
plt.show()

# Feature Selection for Prediction
predictor_vars = ['ASR_Incidents', 'Crude_Rate_Incidents', 'Cum_Risk_Incidents', 'ASR_Mortality', 
                  'Crude_Rate_Mortality', 'Cum_Risk_Mortality']
outcome_var = 'Incidents_Number'

X = df[predictor_vars]
y = df[outcome_var]

# Splitting Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nTrain-Test Split:", X_train.shape, X_test.shape)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Model Performance
r2_score_rf = metrics.r2_score(y_test, y_pred_rf)
print(f"\nRandom Forest Model R² Score: {r2_score_rf:.3f}")

# Logistic Regression Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(class_weight='balanced', max_iter=200)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluate Logistic Regression Model
accuracy_lr = metrics.accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr:.3f}")

# Save Random Forest Model
joblib.dump(rf_model, 'random_forest_model.pkl')
print("\nModel saved as 'random_forest_model.pkl'")
