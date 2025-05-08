from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
from typing import Dict, Any
import pandas as pd
import numpy as np

class ChurnPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.metrics = {}
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train the model and return metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions
        """
        return self.model.predict_proba(X)
    
    def save_model(self, path: str):
        """
        Save the model to disk
        """
        joblib.dump(self.model, path)
        
        # Save metrics
        metrics_path = path.replace('.joblib', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f)
    
    def load_model(self, path: str):
        """
        Load the model from disk
        """
        self.model = joblib.load(path)
        
        # Load metrics
        metrics_path = path.replace('.joblib', '_metrics.json')
        with open(metrics_path, 'r') as f:
            self.metrics = json.load(f)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        """
        feature_importance = self.model.feature_importances_
        return dict(zip(self.model.feature_names_in_, feature_importance)) 