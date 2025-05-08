from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from app.db.database import get_db
from app.ml.feature_engineering import FeatureEngineer
from app.ml.model import ChurnPredictor
from app.db.models import Customer, FeatureSet, Model

router = APIRouter()

# Initialize feature engineer and model
feature_engineer = FeatureEngineer()
model = ChurnPredictor()

@router.post("/predict")
async def predict_churn(
    customer_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Predict customer churn probability
    """
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Engineer features
        X = feature_engineer.engineer_features(df)
        
        # Make prediction
        prediction = model.predict(X)
        probability = model.predict_proba(X)
        
        return {
            "churn_prediction": bool(prediction[0]),
            "churn_probability": float(probability[0][1])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/feature-importance")
async def get_feature_importance():
    """
    Get feature importance scores
    """
    try:
        importance = model.get_feature_importance()
        return importance
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/train")
async def train_model(
    db: Session = Depends(get_db)
):
    """
    Train the model with current data
    """
    try:
        # Get all customers
        customers = db.query(Customer).all()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'age': c.age,
            'gender': c.gender,
            'tenure': c.tenure,
            'monthly_charges': c.monthly_charges,
            'total_charges': c.total_charges,
            'contract_type': c.contract_type,
            'payment_method': c.payment_method
        } for c in customers])
        
        # Engineer features
        X = feature_engineer.engineer_features(df)
        y = np.array([c.churn for c in customers])
        
        # Train model
        metrics = model.train(X, y)
        
        # Save model
        model.save_model("models/churn_model.joblib")
        
        return {
            "message": "Model trained successfully",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 