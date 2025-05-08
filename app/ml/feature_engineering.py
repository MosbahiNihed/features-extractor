from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.numeric_features = ['age', 'tenure', 'monthly_charges', 'total_charges']
        self.categorical_features = ['gender', 'contract_type', 'payment_method']
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to the input dataframe
        """
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Create new features
        df = self._create_derived_features(df)
        
        # Transform features
        X = self.preprocessor.fit_transform(df)
        
        return X
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        """
        # Fill numeric missing values with median
        for col in self.numeric_features:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical missing values with mode
        for col in self.categorical_features:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones
        """
        # Create tenure groups
        df['tenure_group'] = pd.qcut(df['tenure'], q=4, labels=['low', 'medium', 'high', 'very_high'])
        
        # Create monthly charges per tenure
        df['charges_per_tenure'] = df['monthly_charges'] / (df['tenure'] + 1)
        
        # Create total charges per month
        df['total_charges_per_month'] = df['total_charges'] / (df['tenure'] + 1)
        
        return df 

class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=1)  # Default to 1 component

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and transform features from customer data
        """
        try:
            # Create a copy to avoid modifying the original
            features_df = df.copy()
            
            # Convert categorical variables to numeric if not already done
            categorical_cols = ['gender', 'contract_type', 'payment_method']
            for col in categorical_cols:
                if col in features_df.columns and features_df[col].dtype == 'object':
                    features_df[col] = pd.Categorical(features_df[col]).codes
            
            # Ensure numeric columns are float
            numeric_cols = ['age', 'tenure', 'monthly_charges', 'total_charges']
            for col in numeric_cols:
                if col in features_df.columns:
                    features_df[col] = features_df[col].astype(float)
            
            # Calculate additional features
            if 'tenure' in features_df.columns and 'monthly_charges' in features_df.columns:
                features_df['avg_monthly_charge'] = features_df['total_charges'] / features_df['tenure'].replace(0, 1)
            
            if 'age' in features_df.columns:
                features_df['age_group'] = pd.cut(
                    features_df['age'],
                    bins=[0, 25, 35, 45, 55, 100],
                    labels=['18-25', '26-35', '36-45', '46-55', '55+']
                ).cat.codes
            
            # Fill any missing values
            features_df = features_df.fillna(0)
            
            return features_df
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            raise

    def analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Analyze feature importance using Random Forest
        """
        try:
            # Ensure all data is numeric
            X = X.astype(float)
            y = y.astype(int)
            
            # Train a Random Forest model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Get feature importance
            importance = dict(zip(X.columns, model.feature_importances_))
            return {k: float(v) for k, v in importance.items()}
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {str(e)}")
            raise

    def get_feature_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlations between features
        """
        try:
            # Ensure all data is numeric
            numeric_df = df.select_dtypes(include=[np.number])
            return numeric_df.corr()
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            raise

    def perform_pca(self, X: pd.DataFrame) -> dict:
        """
        Perform Principal Component Analysis
        """
        try:
            # Ensure all data is numeric
            X = X.astype(float)
            
            # Determine number of components
            n_features = X.shape[1]
            n_samples = X.shape[0]
            n_components = min(n_features, n_samples, 2)  # Use at most 2 components
            
            if n_components < 1:
                return {
                    "error": "Not enough features for PCA",
                    "n_features": n_features,
                    "n_samples": n_samples
                }
            
            # Update PCA components
            self.pca.n_components = n_components
            
            # Scale the data
            X_scaled = self.scaler.fit_transform(X)
            
            # Perform PCA
            pca_result = self.pca.fit_transform(X_scaled)
            
            # Get explained variance
            explained_variance = self.pca.explained_variance_ratio_
            
            return {
                "components": pca_result.tolist(),
                "explained_variance": explained_variance.tolist(),
                "feature_names": X.columns.tolist(),
                "n_components": n_components
            }
        except Exception as e:
            logger.error(f"Error in PCA: {str(e)}")
            raise 