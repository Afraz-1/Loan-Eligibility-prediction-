# Loan Approval Prediction

Machine learning model to predict loan approval decisions based on applicant financial and demographic data.

## Dataset
- 614 loan applications with 12 features
- Target: Binary classification (Approved/Rejected)
- Features: Income, education, credit history, property area, dependents

## Methodology
1. **Data Preprocessing**: Missing value imputation, outlier removal, feature scaling
2. **Feature Engineering**: Created total income, EMI, and income-to-EMI ratio features
3. **Model Comparison**: Tested Random Forest, SVM, SGD, Gradient Boosting, XGBoost, and Voting Classifier
4. **Hyperparameter Tuning**: GridSearchCV optimization for best performing model

## Results
- Best Model: SVM with linear kernel
- Test Accuracy: 81.2%
- F1 Score: 87.9%
- ROC AUC: 70.5%

## Key Features
- Complete ML pipeline with proper validation
- Multiple algorithm comparison
- Feature importance analysis
- ROC curve visualization

## Technologies
Python, Scikit-learn, XGBoost, Pandas, Matplotlib, Seaborn

## Installation
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

