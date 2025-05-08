import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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