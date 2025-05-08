from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ModelTraining:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None

    def prepare_data(self, 
                    data: pd.DataFrame, 
                    target_column: str,
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training
        """
        try:
            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def train_models(self, 
                    X_train: np.ndarray, 
                    y_train: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Train multiple models and return their cross-validation scores
        """
        try:
            results = {}
            best_score = 0

            for name, model in self.models.items():
                # Perform cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                # Train model on full training set
                model.fit(X_train, y_train)
                
                # Store results
                results[name] = {
                    'mean_cv_score': cv_scores.mean(),
                    'std_cv_score': cv_scores.std()
                }
                
                # Update best model
                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    self.best_model = model
                    self.best_model_name = name

            return results
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def evaluate_model(self, 
                      model: object, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }

            return metrics
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def save_model(self, 
                  model: object, 
                  model_name: str, 
                  metrics: Dict[str, float],
                  version: str = None) -> str:
        """
        Save model and its metadata
        """
        try:
            # Generate version if not provided
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create model directory
            model_dir = f"models/{model_name}_{version}"
            
            # Save model
            joblib.dump(model, f"{model_dir}/model.joblib")
            
            # Save scaler
            joblib.dump(self.scaler, f"{model_dir}/scaler.joblib")
            
            # Save metrics
            with open(f"{model_dir}/metrics.json", 'w') as f:
                json.dump(metrics, f, indent=4)
            
            return model_dir
        except Exception as e:
            logger.error(f"Error in model saving: {str(e)}")
            raise

    def load_model(self, model_path: str) -> Tuple[object, object]:
        """
        Load saved model and scaler
        """
        try:
            model = joblib.load(f"{model_path}/model.joblib")
            scaler = joblib.load(f"{model_path}/scaler.joblib")
            return model, scaler
        except Exception as e:
            logger.error(f"Error in model loading: {str(e)}")
            raise 