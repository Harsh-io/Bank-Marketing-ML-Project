"""
model_training.py
=================
Model training module for the Bank Marketing ML project.

This module provides:
- Definitions of multiple ML models suitable for categorical data
- Training functions with hyperparameter tuning options
- Model persistence (save/load functionality)

Implemented Models:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. XGBoost Classifier
6. LightGBM Classifier
7. Support Vector Machine (SVM)

Usage:
    from src.model_training import ModelTrainer
    
    trainer = ModelTrainer()
    trained_models = trainer.train_all_models(X_train, y_train)

Author: Data Science Team
Date: January 2026
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced boosting models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Use 'pip install xgboost' to enable.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Use 'pip install lightgbm' to enable.")


class ModelTrainer:
    """
    A comprehensive model training class that implements multiple ML models.
    
    Attributes:
        models (dict): Dictionary of model names and their instances
        trained_models (dict): Dictionary of trained model instances
        training_times (dict): Dictionary of training times for each model
        cv_scores (dict): Dictionary of cross-validation scores
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the ModelTrainer with default models.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.training_times = {}
        self.cv_scores = {}
        
        # Initialize default models
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialize all available models with default hyperparameters.
        """
        # 1. Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            solver='lbfgs',
            n_jobs=-1
        )
        
        # 2. Decision Tree
        self.models['Decision Tree'] = DecisionTreeClassifier(
            random_state=self.random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        # 3. Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        )
        
        # 4. Gradient Boosting (sklearn)
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            random_state=self.random_state,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8
        )
        
        # 5. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
        
        # 6. LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = LGBMClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                verbose=-1
            )
        
        # 7. Support Vector Machine
        self.models['SVM'] = SVC(
            random_state=self.random_state,
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True  # Enable probability estimates for ROC-AUC
        )
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def get_model(self, model_name):
        """
        Get a specific model by name.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            sklearn estimator: The model instance
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        return self.models[model_name]
    
    def add_custom_model(self, name, model):
        """
        Add a custom model to the trainer.
        
        Args:
            name (str): Name of the model
            model: sklearn-compatible model instance
        """
        self.models[name] = model
        print(f"Added custom model: {name}")
    
    def train_model(self, model_name, X_train, y_train, cv=5):
        """
        Train a single model with cross-validation.
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame or np.array): Training features
            y_train (pd.Series or np.array): Training labels
            cv (int): Number of cross-validation folds
            
        Returns:
            sklearn estimator: Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.models[model_name]
        
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        self.cv_scores[model_name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_times[model_name] = training_time
        self.trained_models[model_name] = model
        
        print(f"  ✓ Training completed in {training_time:.2f}s")
        print(f"  ✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return model
    
    def train_all_models(self, X_train, y_train, cv=5, exclude=None):
        """
        Train all available models.
        
        Args:
            X_train (pd.DataFrame or np.array): Training features
            y_train (pd.Series or np.array): Training labels
            cv (int): Number of cross-validation folds
            exclude (list): List of model names to exclude from training
            
        Returns:
            dict: Dictionary of trained models
        """
        if exclude is None:
            exclude = []
        
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        total_start = time.time()
        
        for model_name in self.models.keys():
            if model_name not in exclude:
                self.train_model(model_name, X_train, y_train, cv=cv)
        
        total_time = time.time() - total_start
        
        print("\n" + "="*60)
        print(f"ALL MODELS TRAINED IN {total_time:.2f}s")
        print("="*60)
        
        return self.trained_models
    
    def get_training_summary(self):
        """
        Get a summary of training results.
        
        Returns:
            pd.DataFrame: Summary of training times and CV scores
        """
        summary_data = []
        
        for model_name in self.trained_models.keys():
            summary_data.append({
                'Model': model_name,
                'Training Time (s)': self.training_times.get(model_name, 0),
                'CV Accuracy Mean': self.cv_scores.get(model_name, {}).get('mean', 0),
                'CV Accuracy Std': self.cv_scores.get(model_name, {}).get('std', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('CV Accuracy Mean', ascending=False)
        
        return summary_df
    
    def tune_hyperparameters(self, model_name, X_train, y_train, param_grid, cv=5):
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            model_name (str): Name of the model to tune
            X_train: Training features
            y_train: Training labels
            param_grid (dict): Parameter grid for GridSearchCV
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Best parameters and best score
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        print(f"\nTuning hyperparameters for {model_name}...")
        
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        print(f"  ✓ Best Parameters: {grid_search.best_params_}")
        print(f"  ✓ Best CV Score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_models(self, output_dir):
        """
        Save all trained models to disk.
        
        Args:
            output_dir (str): Directory to save the models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving models to {output_dir}...")
        
        for model_name, model in self.trained_models.items():
            # Create a safe filename
            safe_name = model_name.replace(' ', '_').lower()
            filepath = os.path.join(output_dir, f'{safe_name}.joblib')
            joblib.dump(model, filepath)
            print(f"  ✓ Saved: {filepath}")
    
    def load_models(self, input_dir):
        """
        Load trained models from disk.
        
        Args:
            input_dir (str): Directory containing saved models
        """
        print(f"\nLoading models from {input_dir}...")
        
        for filename in os.listdir(input_dir):
            if filename.endswith('.joblib'):
                filepath = os.path.join(input_dir, filename)
                model_name = filename.replace('.joblib', '').replace('_', ' ').title()
                self.trained_models[model_name] = joblib.load(filepath)
                print(f"  ✓ Loaded: {model_name}")


def get_hyperparameter_grids():
    """
    Get predefined hyperparameter grids for tuning.
    
    Returns:
        dict: Dictionary of parameter grids for each model
    """
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga']
        },
        'Decision Tree': {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    }
    
    return param_grids


if __name__ == "__main__":
    # Example usage
    print("Model Training Module")
    print("="*60)
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Show available models
    print("\nAvailable models for training:")
    for name in trainer.models.keys():
        print(f"  - {name}")
    
    print("\n✓ Model training module loaded successfully!")
    print("\nTo train models, use:")
    print("  trainer.train_all_models(X_train, y_train)")
