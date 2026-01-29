"""
main.py
=======
Main execution script for the Bank Marketing ML project.

This script runs the complete ML pipeline:
1. Load and preprocess data
2. Train multiple models
3. Evaluate and compare results
4. Save results to files

Usage:
    python main.py

Author: Data Science Team
Date: January 2026
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator, generate_full_report


def main():
    """
    Main function to run the complete ML pipeline.
    """
    print("\n" + "="*80)
    print("BANK MARKETING ML PROJECT - COMPLETE ANALYSIS")
    print("="*80)
    
    # Configuration
    DATA_PATH = 'bank-additional-full.csv'
    RESULTS_DIR = 'results'
    PROCESSED_DATA_DIR = 'data/processed'
    RANDOM_STATE = 42
    
    # Ensure directories exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # ========================================
    # Step 1: Data Preprocessing
    # ========================================
    print("\n" + "-"*80)
    print("STEP 1: DATA PREPROCESSING")
    print("-"*80)
    
    preprocessor = DataPreprocessor(random_state=RANDOM_STATE)
    
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        filepath=DATA_PATH,
        missing_strategy='keep',  # Keep 'unknown' as a category
        encoding_type='onehot',   # Use One-Hot Encoding
        test_size=0.2             # 80/20 train-test split
    )
    
    # Save processed data
    preprocessor.save_processed_data(
        X_train, X_test, y_train, y_test,
        output_dir=PROCESSED_DATA_DIR
    )
    
    # ========================================
    # Step 2: Model Training
    # ========================================
    print("\n" + "-"*80)
    print("STEP 2: MODEL TRAINING")
    print("-"*80)
    
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    
    # Train all models (exclude SVM for faster execution)
    trained_models = trainer.train_all_models(
        X_train, y_train,
        cv=5,
        exclude=['SVM']  # Exclude SVM due to computational cost
    )
    
    # Display training summary
    training_summary = trainer.get_training_summary()
    print("\nTraining Summary:")
    print(training_summary.to_string(index=False))
    
    # ========================================
    # Step 3: Model Evaluation
    # ========================================
    print("\n" + "-"*80)
    print("STEP 3: MODEL EVALUATION")
    print("-"*80)
    
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    results = evaluator.evaluate_all_models(
        trained_models,
        X_train, y_train,
        X_test, y_test
    )
    
    # Print comparison table
    evaluator.print_comparison_table()
    
    # ========================================
    # Step 4: Save Results
    # ========================================
    print("\n" + "-"*80)
    print("STEP 4: SAVING RESULTS")
    print("-"*80)
    
    # Save comparison table
    evaluator.save_comparison_table(
        os.path.join(RESULTS_DIR, 'comparison_table.csv')
    )
    
    # Generate visualizations (if matplotlib display is available)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Save confusion matrices
        evaluator.plot_confusion_matrices(
            figsize=(16, 12),
            save_path=os.path.join(RESULTS_DIR, 'confusion_matrices.png')
        )
        
        # Save ROC curves
        evaluator.plot_roc_curves(
            y_test,
            figsize=(10, 8),
            save_path=os.path.join(RESULTS_DIR, 'roc_curves.png')
        )
        
        # Save metrics comparison
        evaluator.plot_metrics_comparison(
            figsize=(14, 6),
            save_path=os.path.join(RESULTS_DIR, 'metrics_comparison.png')
        )
        
        # Save feature importance (for Random Forest)
        if 'Random Forest' in trained_models:
            evaluator.plot_feature_importance(
                trained_models['Random Forest'],
                'Random Forest',
                preprocessor.feature_names,
                top_n=20,
                figsize=(10, 8),
                save_path=os.path.join(RESULTS_DIR, 'feature_importance.png')
            )
    except Exception as e:
        print(f"  ⚠ Could not generate visualizations: {e}")
    
    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    best_model = evaluator.best_model_name
    best_metrics = evaluator.results[best_model]
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"\nBest Model Performance:")
    print(f"  • Test Accuracy:  {best_metrics['test_accuracy']:.4f}")
    print(f"  • Test Precision: {best_metrics['test_precision']:.4f}")
    print(f"  • Test Recall:    {best_metrics['test_recall']:.4f}")
    print(f"  • Test F1-Score:  {best_metrics['test_f1']:.4f}")
    print(f"  • Test ROC-AUC:   {best_metrics['test_roc_auc']:.4f}")
    
    print(f"\nResults saved to '{RESULTS_DIR}/' directory:")
    print(f"  • comparison_table.csv")
    print(f"  • confusion_matrices.png")
    print(f"  • roc_curves.png")
    print(f"  • metrics_comparison.png")
    print(f"  • feature_importance.png")
    
    print("\n" + "="*80)
    
    return evaluator.comparison_df


if __name__ == "__main__":
    comparison_df = main()
