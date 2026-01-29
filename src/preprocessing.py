"""
preprocessing.py
================
Data preprocessing module for the Bank Marketing ML project.

This module handles:
- Loading and initial exploration of the data
- Missing value detection and handling
- Categorical variable encoding (One-Hot, Label, Target encoding)
- Feature scaling and normalization
- Train/test splitting

Usage:
    from src.preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        filepath='data/raw/bank-additional-full.csv'
    )

Author: Data Science Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for the Bank Marketing dataset.
    
    Attributes:
        df (pd.DataFrame): The loaded dataset
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        scaler (StandardScaler): Scaler for numerical features
        label_encoders (dict): Dictionary of label encoders for categorical features
        column_transformer (ColumnTransformer): Transformer for feature encoding
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the DataPreprocessor.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.df = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.column_transformer = None
        self.feature_names = None
        
        # Define column types based on the Bank Marketing dataset
        self.categorical_columns = [
            'job', 'marital', 'education', 'default', 
            'housing', 'loan', 'contact', 'month', 
            'day_of_week', 'poutcome'
        ]
        
        self.numerical_columns = [
            'age', 'duration', 'campaign', 'pdays', 
            'previous', 'emp.var.rate', 'cons.price.idx',
            'cons.conf.idx', 'euribor3m', 'nr.employed'
        ]
        
        self.target_column = 'y'
    
    def load_data(self, filepath):
        """
        Load the dataset from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print(f"Loading data from {filepath}...")
        self.df = pd.read_csv(filepath, sep=';')
        print(f"Dataset loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def explore_data(self):
        """
        Perform initial data exploration.
        
        Returns:
            dict: Dictionary containing exploration results
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        exploration = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'target_distribution': self.df[self.target_column].value_counts().to_dict()
        }
        
        print("\n" + "="*60)
        print("DATA EXPLORATION SUMMARY")
        print("="*60)
        print(f"\nDataset Shape: {exploration['shape']}")
        print(f"\nTarget Variable Distribution:")
        for key, value in exploration['target_distribution'].items():
            percentage = (value / self.df.shape[0]) * 100
            print(f"  {key}: {value} ({percentage:.2f}%)")
        
        # Check for 'unknown' values in categorical columns
        print("\n'Unknown' values in categorical columns:")
        for col in self.categorical_columns:
            if col in self.df.columns:
                unknown_count = (self.df[col] == 'unknown').sum()
                if unknown_count > 0:
                    print(f"  {col}: {unknown_count} ({(unknown_count/len(self.df))*100:.2f}%)")
        
        return exploration
    
    def handle_missing_values(self, strategy='keep'):
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy for handling missing values
                - 'keep': Keep 'unknown' as a separate category
                - 'drop': Drop rows with 'unknown' values
                - 'mode': Replace 'unknown' with mode of the column
                
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"\nHandling missing values with strategy: '{strategy}'")
        
        if strategy == 'keep':
            # Keep 'unknown' as a valid category
            print("  - Keeping 'unknown' as a separate category")
            
        elif strategy == 'drop':
            # Drop rows with 'unknown' values
            initial_rows = len(self.df)
            for col in self.categorical_columns:
                if col in self.df.columns:
                    self.df = self.df[self.df[col] != 'unknown']
            dropped_rows = initial_rows - len(self.df)
            print(f"  - Dropped {dropped_rows} rows with 'unknown' values")
            
        elif strategy == 'mode':
            # Replace 'unknown' with mode
            for col in self.categorical_columns:
                if col in self.df.columns:
                    # Calculate mode excluding 'unknown'
                    mode_val = self.df[self.df[col] != 'unknown'][col].mode()
                    if len(mode_val) > 0:
                        unknown_count = (self.df[col] == 'unknown').sum()
                        if unknown_count > 0:
                            self.df[col] = self.df[col].replace('unknown', mode_val[0])
                            print(f"  - Replaced {unknown_count} 'unknown' values in '{col}' with '{mode_val[0]}'")
        
        return self.df
    
    def encode_categorical_features(self, encoding_type='onehot'):
        """
        Encode categorical features using the specified method.
        
        Args:
            encoding_type (str): Type of encoding to use
                - 'onehot': One-Hot Encoding (creates binary columns)
                - 'label': Label Encoding (converts to integers)
                - 'target': Target Encoding (replaces with target mean)
                
        Returns:
            tuple: (X, y) encoded features and target
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print(f"\nEncoding categorical features using: '{encoding_type}'")
        
        # Separate features and target
        self.y = (self.df[self.target_column] == 'yes').astype(int)
        self.X = self.df.drop(columns=[self.target_column])
        
        if encoding_type == 'onehot':
            # One-Hot Encoding
            self.X = pd.get_dummies(self.X, columns=self.categorical_columns, drop_first=True)
            self.feature_names = list(self.X.columns)
            print(f"  - Created {len(self.feature_names)} features after One-Hot Encoding")
            
        elif encoding_type == 'label':
            # Label Encoding
            for col in self.categorical_columns:
                if col in self.X.columns:
                    self.label_encoders[col] = LabelEncoder()
                    self.X[col] = self.label_encoders[col].fit_transform(self.X[col])
            self.feature_names = list(self.X.columns)
            print(f"  - Label encoded {len(self.categorical_columns)} categorical columns")
            
        elif encoding_type == 'target':
            # Target Encoding (mean encoding)
            for col in self.categorical_columns:
                if col in self.X.columns:
                    # Calculate mean target value for each category
                    target_means = self.df.groupby(col)[self.target_column].apply(
                        lambda x: (x == 'yes').mean()
                    )
                    self.X[col] = self.X[col].map(target_means)
            self.feature_names = list(self.X.columns)
            print(f"  - Target encoded {len(self.categorical_columns)} categorical columns")
        
        return self.X, self.y
    
    def scale_features(self, columns_to_scale=None):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            columns_to_scale (list): List of columns to scale. 
                                    If None, scales all numerical columns.
                                    
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        if self.X is None:
            raise ValueError("Features not prepared. Call encode_categorical_features() first.")
        
        if columns_to_scale is None:
            columns_to_scale = [col for col in self.numerical_columns if col in self.X.columns]
        
        print(f"\nScaling {len(columns_to_scale)} numerical features...")
        
        # Scale the numerical columns
        self.X[columns_to_scale] = self.scaler.fit_transform(self.X[columns_to_scale])
        
        return self.X
    
    def split_data(self, test_size=0.2, stratify=True):
        """
        Split the data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data for testing
            stratify (bool): Whether to stratify by target variable
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.X is None or self.y is None:
            raise ValueError("Features not prepared. Call encode_categorical_features() first.")
        
        print(f"\nSplitting data: {1-test_size:.0%} train, {test_size:.0%} test")
        
        stratify_param = self.y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"  - Training set: {X_train.shape[0]} samples")
        print(f"  - Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def fit_transform(self, filepath, missing_strategy='keep', 
                      encoding_type='onehot', test_size=0.2):
        """
        Complete preprocessing pipeline: load, handle missing values,
        encode, scale, and split the data.
        
        Args:
            filepath (str): Path to the CSV file
            missing_strategy (str): Strategy for missing values
            encoding_type (str): Type of categorical encoding
            test_size (float): Proportion of test data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        self.load_data(filepath)
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Handle missing values
        self.handle_missing_values(strategy=missing_strategy)
        
        # Step 4: Encode categorical features
        self.encode_categorical_features(encoding_type=encoding_type)
        
        # Step 5: Scale numerical features
        self.scale_features()
        
        # Step 6: Split data
        X_train, X_test, y_train, y_test = self.split_data(test_size=test_size)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, output_dir):
        """
        Save processed data to CSV files.
        
        Args:
            X_train, X_test, y_train, y_test: Train/test splits
            output_dir (str): Directory to save the files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine features and target for saving
        train_data = X_train.copy()
        train_data['target'] = y_train.values
        
        test_data = X_test.copy()
        test_data['target'] = y_test.values
        
        train_path = os.path.join(output_dir, 'train_data.csv')
        test_path = os.path.join(output_dir, 'test_data.csv')
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        print(f"\nProcessed data saved:")
        print(f"  - Training data: {train_path}")
        print(f"  - Testing data: {test_path}")


def get_feature_importance_df(feature_names, importances):
    """
    Create a DataFrame of feature importances.
    
    Args:
        feature_names (list): List of feature names
        importances (array): Array of importance values
        
    Returns:
        pd.DataFrame: DataFrame with features and their importances
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(random_state=42)
    
    # Run the complete preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        filepath='../bank-additional-full.csv',
        missing_strategy='keep',
        encoding_type='onehot',
        test_size=0.2
    )
    
    # Save processed data
    preprocessor.save_processed_data(
        X_train, X_test, y_train, y_test,
        output_dir='../data/processed'
    )
    
    print("\n✓ Preprocessing script executed successfully!")
