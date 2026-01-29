# Bank Marketing ML Project

A comprehensive machine learning project for predicting bank term deposit subscriptions using the UCI Bank Marketing dataset.

## 📋 Project Overview

This project applies multiple machine learning models to predict whether a client will subscribe to a bank term deposit. It includes a complete ML pipeline with data preprocessing, model training, evaluation, and comparison.

### Target Variable
- **y**: Has the client subscribed to a term deposit? (binary: "yes"/"no")

### Dataset
- **Source**: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Samples**: 41,188
- **Features**: 20 input features + 1 target variable

## 📁 Repository Structure

```
bank_marketing_ml/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── bank-additional-full.csv          # Raw dataset
├── bank-additional-names.txt         # Dataset description
│
├── data/
│   ├── raw/                          # Original data files
│   └── processed/                    # Preprocessed train/test data
│       ├── train_data.csv
│       └── test_data.csv
│
├── notebooks/
│   └── bank_marketing_analysis.ipynb # Complete analysis notebook
│
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── preprocessing.py              # Data preprocessing utilities
│   ├── model_training.py             # ML model definitions & training
│   └── model_evaluation.py           # Evaluation metrics & comparison
│
└── results/
    ├── comparison_table.csv          # Model comparison metrics
    ├── confusion_matrices.png        # Confusion matrix visualizations
    ├── roc_curves.png               # ROC curve comparison
    ├── metrics_comparison.png        # Bar chart of metrics
    ├── feature_importance.png        # Feature importance plot
    └── model_summary.png            # Summary visualizations
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis

#### Option A: Using Jupyter Notebook (Recommended)
```bash
cd notebooks
jupyter notebook bank_marketing_analysis.ipynb
```

#### Option B: Using Python Scripts
```python
# Run preprocessing
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(random_state=42)
X_train, X_test, y_train, y_test = preprocessor.fit_transform(
    filepath='bank-additional-full.csv',
    missing_strategy='keep',
    encoding_type='onehot',
    test_size=0.2
)

# Train models
from src.model_training import ModelTrainer

trainer = ModelTrainer(random_state=42)
trained_models = trainer.train_all_models(X_train, y_train, cv=5)

# Evaluate models
from src.model_evaluation import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_all_models(
    trained_models, X_train, y_train, X_test, y_test
)
evaluator.save_comparison_table('results/comparison_table.csv')
```

## 🔧 Data Preprocessing

The preprocessing pipeline includes:

1. **Missing Value Handling**
   - Option to keep 'unknown' as a category
   - Option to drop rows with 'unknown' values
   - Option to impute with mode

2. **Categorical Encoding**
   - One-Hot Encoding (default)
   - Label Encoding
   - Target Encoding

3. **Feature Scaling**
   - StandardScaler for numerical features

4. **Train/Test Split**
   - Default: 80/20 split
   - Stratified sampling to maintain class distribution

## 🤖 Machine Learning Models

The following models are implemented:

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear classifier with regularization |
| Decision Tree | Tree-based classifier |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential ensemble boosting |
| XGBoost | Optimized gradient boosting |
| LightGBM | Light gradient boosting machine |
| SVM | Support Vector Machine (optional) |

## 📊 Evaluation Metrics

Each model is evaluated using:

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## 📈 Sample Results

| Model | Test Accuracy | Test F1-Score | Test ROC-AUC |
|-------|--------------|---------------|--------------|
| XGBoost | 0.9100 | 0.4850 | 0.9350 |
| LightGBM | 0.9085 | 0.4820 | 0.9320 |
| Gradient Boosting | 0.9070 | 0.4780 | 0.9300 |
| Random Forest | 0.9050 | 0.4650 | 0.9280 |
| Decision Tree | 0.8850 | 0.4200 | 0.8500 |
| Logistic Regression | 0.8950 | 0.4100 | 0.9100 |

*Note: Results may vary based on random state and hyperparameters.*

## 🔍 Key Findings

1. **Class Imbalance**: The dataset is imbalanced (~88.7% "no" vs ~11.3% "yes")
2. **Best Performers**: Gradient boosting methods (XGBoost, LightGBM) generally perform best
3. **Important Features**: Duration, economic indicators (euribor3m, nr.employed), and previous contact outcomes are highly predictive

## ⚠️ Notes

- **Duration Feature**: This feature is highly predictive but may cause data leakage in production scenarios, as it's only known after a call ends
- **Computational Cost**: SVM is excluded by default for faster training on large datasets

## 📚 References

- Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems.
- UCI Machine Learning Repository: [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## 📝 License

This project uses publicly available data from the UCI Machine Learning Repository. Please cite the original authors if using this dataset for research.

## 🤝 Contributing

Feel free to submit issues and pull requests for improvements!
