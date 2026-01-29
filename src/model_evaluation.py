"""
model_evaluation.py
===================
Model evaluation module for the Bank Marketing ML project.

This module provides:
- Comprehensive evaluation metrics for classification models
- Comparison tables and visualizations
- ROC curves and confusion matrices
- Feature importance analysis

Metrics Calculated:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Usage:
    from src.model_evaluation import ModelEvaluator
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(trained_models, X_test, y_test)
    evaluator.create_comparison_table()

Author: Data Science Team
Date: January 2026
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    A comprehensive model evaluation class for classification tasks.
    
    Attributes:
        results (dict): Dictionary storing evaluation results for each model
        comparison_df (pd.DataFrame): DataFrame with model comparison metrics
    """
    
    def __init__(self):
        """
        Initialize the ModelEvaluator.
        """
        self.results = {}
        self.comparison_df = None
        self.best_model = None
        self.best_model_name = None
    
    def evaluate_model(self, model, model_name, X_train, y_train, X_test, y_test):
        """
        Evaluate a single model on training and test data.
        
        Args:
            model: Trained sklearn-compatible model
            model_name (str): Name of the model
            X_train: Training features
            y_train: Training labels
            X_test: Testing features
            y_test: Testing labels
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Probability predictions (for ROC-AUC)
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_train_proba = y_train_pred
            y_test_proba = y_test_pred
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            
            # Training Metrics
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
            'train_roc_auc': roc_auc_score(y_train, y_train_proba),
            
            # Testing Metrics
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, y_test_proba),
            
            # Additional data for plotting
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }
        
        # Store results
        self.results[model_name] = metrics
        
        # Print summary
        print(f"  ✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  ✓ Test F1-Score: {metrics['test_f1']:.4f}")
        print(f"  ✓ Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, trained_models, X_train, y_train, X_test, y_test):
        """
        Evaluate all trained models.
        
        Args:
            trained_models (dict): Dictionary of trained model instances
            X_train, y_train: Training data
            X_test, y_test: Testing data
            
        Returns:
            dict: Dictionary of evaluation results for all models
        """
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)
        
        for model_name, model in trained_models.items():
            self.evaluate_model(model, model_name, X_train, y_train, X_test, y_test)
        
        # Create comparison table
        self.create_comparison_table()
        
        return self.results
    
    def create_comparison_table(self):
        """
        Create a structured comparison table of all model results.
        
        Returns:
            pd.DataFrame: Comparison table with all metrics
        """
        if not self.results:
            raise ValueError("No results available. Evaluate models first.")
        
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train Accuracy': metrics['train_accuracy'],
                'Test Accuracy': metrics['test_accuracy'],
                'Train Precision': metrics['train_precision'],
                'Test Precision': metrics['test_precision'],
                'Train Recall': metrics['train_recall'],
                'Test Recall': metrics['test_recall'],
                'Train F1-Score': metrics['train_f1'],
                'Test F1-Score': metrics['test_f1'],
                'Train ROC-AUC': metrics['train_roc_auc'],
                'Test ROC-AUC': metrics['test_roc_auc']
            })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by Test F1-Score (or your preferred metric)
        self.comparison_df = self.comparison_df.sort_values(
            'Test F1-Score', ascending=False
        ).reset_index(drop=True)
        
        # Identify best model
        best_idx = self.comparison_df['Test F1-Score'].idxmax()
        self.best_model_name = self.comparison_df.loc[best_idx, 'Model']
        
        return self.comparison_df
    
    def print_comparison_table(self):
        """
        Print a formatted comparison table to the console.
        """
        if self.comparison_df is None:
            self.create_comparison_table()
        
        print("\n" + "="*100)
        print("MODEL COMPARISON TABLE")
        print("="*100)
        
        # Create a formatted display version
        display_df = self.comparison_df.copy()
        
        # Format numeric columns
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        print(display_df.to_string(index=False))
        
        print("\n" + "="*100)
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print("="*100)
    
    def save_comparison_table(self, output_path):
        """
        Save the comparison table to a CSV file.
        
        Args:
            output_path (str): Path to save the CSV file
        """
        if self.comparison_df is None:
            self.create_comparison_table()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.comparison_df.to_csv(output_path, index=False)
        print(f"\n✓ Comparison table saved to: {output_path}")
    
    def get_classification_report(self, model_name, y_test):
        """
        Get a detailed classification report for a specific model.
        
        Args:
            model_name (str): Name of the model
            y_test: True labels
            
        Returns:
            str: Classification report
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found in results.")
        
        y_pred = self.results[model_name]['y_test_pred']
        report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])
        
        return report
    
    def plot_confusion_matrices(self, figsize=(15, 10), save_path=None):
        """
        Plot confusion matrices for all models.
        
        Args:
            figsize (tuple): Figure size
            save_path (str): Path to save the figure (optional)
        """
        if not self.results:
            raise ValueError("No results available. Evaluate models first.")
        
        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                ax=axes[idx],
                xticklabels=['No', 'Yes'],
                yticklabels=['No', 'Yes']
            )
            axes[idx].set_title(f'{model_name}\nAccuracy: {metrics["test_accuracy"]:.4f}')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Confusion Matrices - All Models', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrices saved to: {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, y_test, figsize=(10, 8), save_path=None):
        """
        Plot ROC curves for all models.
        
        Args:
            y_test: True labels
            figsize (tuple): Figure size
            save_path (str): Path to save the figure (optional)
        """
        if not self.results:
            raise ValueError("No results available. Evaluate models first.")
        
        plt.figure(figsize=figsize)
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        # Plot ROC curve for each model
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for (model_name, metrics), color in zip(self.results.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, metrics['y_test_proba'])
            auc = metrics['test_roc_auc']
            plt.plot(fpr, tpr, color=color, label=f'{model_name} (AUC = {auc:.4f})')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curves saved to: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, metrics=['Test Accuracy', 'Test Precision', 
                                                'Test Recall', 'Test F1-Score'],
                                figsize=(12, 6), save_path=None):
        """
        Plot a bar chart comparing selected metrics across models.
        
        Args:
            metrics (list): List of metrics to compare
            figsize (tuple): Figure size
            save_path (str): Path to save the figure (optional)
        """
        if self.comparison_df is None:
            self.create_comparison_table()
        
        plot_df = self.comparison_df[['Model'] + metrics].melt(
            id_vars=['Model'], 
            var_name='Metric', 
            value_name='Score'
        )
        
        plt.figure(figsize=figsize)
        
        ax = sns.barplot(data=plot_df, x='Model', y='Score', hue='Metric')
        
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Metrics comparison saved to: {save_path}")
        
        plt.show()
    
    def get_feature_importance(self, model, model_name, feature_names, top_n=20):
        """
        Get feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            model_name (str): Name of the model
            feature_names (list): List of feature names
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"  ⚠ {model_name} does not have feature_importances_ attribute")
            return None
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_feature_importance(self, model, model_name, feature_names, 
                                top_n=20, figsize=(10, 8), save_path=None):
        """
        Plot feature importance for a model.
        
        Args:
            model: Trained model
            model_name (str): Name of the model
            feature_names (list): List of feature names
            top_n (int): Number of top features to plot
            figsize (tuple): Figure size
            save_path (str): Path to save the figure (optional)
        """
        importance_df = self.get_feature_importance(model, model_name, feature_names, top_n)
        
        if importance_df is None:
            return
        
        plt.figure(figsize=figsize)
        
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to: {save_path}")
        
        plt.show()
        
        return importance_df


def generate_full_report(evaluator, y_test, output_dir='results'):
    """
    Generate a comprehensive evaluation report with all visualizations.
    
    Args:
        evaluator (ModelEvaluator): Fitted ModelEvaluator instance
        y_test: True labels
        output_dir (str): Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING FULL EVALUATION REPORT")
    print("="*60)
    
    # Save comparison table
    evaluator.save_comparison_table(os.path.join(output_dir, 'comparison_table.csv'))
    
    # Print comparison table
    evaluator.print_comparison_table()
    
    # Print classification reports
    print("\n" + "-"*60)
    print("DETAILED CLASSIFICATION REPORTS")
    print("-"*60)
    
    for model_name in evaluator.results.keys():
        print(f"\n{model_name}:")
        print(evaluator.get_classification_report(model_name, y_test))
    
    print("\n✓ Full report generation complete!")


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("="*60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    print("\nModel Evaluator initialized successfully!")
    print("\nTo evaluate models, use:")
    print("  evaluator.evaluate_all_models(trained_models, X_train, y_train, X_test, y_test)")
    print("\nTo create comparison table:")
    print("  evaluator.create_comparison_table()")
    print("  evaluator.save_comparison_table('results/comparison_table.csv')")
