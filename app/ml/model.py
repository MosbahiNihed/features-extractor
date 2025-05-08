from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import NotFittedError
import joblib
import json
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime
from pathlib import Path

model_dir = "models/random_forest_20250508_224500"
Path(model_dir).mkdir(parents=True, exist_ok=True)  # Creates directory if missing
class ChurnPredictor:
    def __init__(self, n_splits=5):  # Add configurable splits
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.n_splits = n_splits  # Store split count
        self.metrics = {}
        self._is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        # Input validation
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            raise ValueError("Need at least 2 classes in y")
            
        # Auto-adjust splits if classes are too small
        safe_splits = min(self.n_splits, min(counts))
        
        # Use cross-validation
        cv = StratifiedKFold(n_splits=safe_splits)
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        
        results = cross_validate(
            self.model,
            X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )
        
        # Store averaged metrics
        self.metrics = {
            'accuracy': np.mean(results['test_accuracy']),
            'precision': np.mean(results['test_precision']),
            'recall': np.mean(results['test_recall']),
            'f1': np.mean(results['test_f1']),
            'n_splits_used': safe_splits,
            'class_distribution': dict(zip(unique_classes, counts))
        }
        
        # Final training on full data
        self.model.fit(X, y)
        self._is_trained = True
        return self.metrics
    
    
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model with error handling
        """
        if not self._is_trained:
            raise NotFittedError("Model not trained yet. Call .train() first.")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.model.predict(X)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}") from e
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions with error handling
        """
        if not self._is_trained:
            raise NotFittedError("Model not trained yet. Call .train() first.")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.model.predict_proba(X)
        except Exception as e:
            raise RuntimeError(f"Probability prediction failed: {str(e)}") from e
    
def save_model(self, path: str):
    """Save model with directory creation"""
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'metrics': self.metrics,
            'is_trained': self._is_trained
        }, path)
        print(f"Model saved to {path}")
    except Exception as e:
        raise RuntimeError(f"Save failed: {str(e)}") from e

def load_model(self, path: str):
    """Load model with error handling""" 
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file missing: {path}")
            
        data = joblib.load(path)
        self.model = data['model']
        self.metrics = data['metrics']
        self._is_trained = data['is_trained']
    except Exception as e:
        raise RuntimeError(f"Load failed: {str(e)}") from e
    
def get_model_path(base_dir="models"):
    """Generate timestamped model path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"{base_dir}/random_forest_{timestamp}"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    return f"{model_dir}/model.joblib"

def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores with error handling
        """
        if not self._is_trained:
            raise NotFittedError("Model not trained yet. Call .train() first.")
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                if hasattr(self.model, 'feature_names_in_'):
                    features = self.model.feature_names_in_
                else:
                    features = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
                
                return dict(zip(features, self.model.feature_importances_))
            else:
                raise AttributeError("Model doesn't support feature importances")
        except Exception as e:
            raise RuntimeError(f"Failed to get feature importance: {str(e)}") from e