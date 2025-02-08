# Women's Cancer Statistics in Iran - 2022: Data Analysis and Prediction
Women's Cancer Statistics in Iran - 2022

## Overview
This project analyzes cancer statistics for women in Iran in 2022 using data from Iran Open Data. The goal is to explore cancer trends, perform exploratory data analysis (EDA), and develop predictive models for cancer incidents and mortality rates.

## Dataset
The dataset contains information about various types of cancer, including:
- Cancer type
- Number of incidents
- Age-standardized rate (ASR) for incidents
- Crude rate for incidents
- Cumulative risk up to age 74 for incidents
- Number of mortalities
- ASR for mortality
- Crude rate for mortality
- Cumulative risk up to age 74 for mortality

## Features
The dataset has been cleaned by:
- Renaming columns for readability
- Handling missing values
- Ensuring numerical data types

## Exploratory Data Analysis (EDA)
EDA includes:
- Visualization of cancer incidents and mortality rates
- Correlation matrix between different statistical factors

## Predictive Modeling
### Models Used:
1. **Random Forest Classifier** - Used to predict cancer incidents based on other features.
2. **Logistic Regression** - Applied as an alternative method for classification.

### Steps in Modeling:
- Feature selection
- Splitting data into training and testing sets
- Training models
- Evaluating performance using accuracy scores and R² scores

### Model Performance
- **Random Forest Model**: Demonstrated moderate accuracy with an R² score of 0.054.
- **Logistic Regression Model**: Did not perform well, achieving an accuracy score of 0.000.

## Future Work
- Hyperparameter tuning to improve model performance
- Experimenting with additional machine learning models such as Gradient Boosting and XGBoost
- Incorporating more features or external data sources

## How to Use This Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/womens-cancer-iran.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis notebook:
   ```bash
   jupyter notebook analysis.ipynb
   ```

## Saving the Model
The trained Random Forest model is saved for future use:
```python
import joblib
joblib.dump(rf_model, 'random_forest_model.pkl')
```

## Contributions
Contributions are welcome! Feel free to fork the repository and submit a pull request with any improvements.

## License
This project is licensed under the MIT License.

