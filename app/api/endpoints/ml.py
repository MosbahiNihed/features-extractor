from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List
import pandas as pd
from app.ml.feature_engineering import FeatureEngineering
from app.ml.model_training import ModelTraining
from app.ml.llm_service import LLMService
from app.db.database import get_db
from sqlalchemy.orm import Session
from app.models import Customer
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
feature_engineering = FeatureEngineering()
model_training = ModelTraining()

# Initialize LLM service with API key from environment
llm_service = LLMService()

@router.post("/features/analyze")
async def analyze_features(db: Session = Depends(get_db)):
    """
    Analyze customer features and return insights
    """
    try:
        # Get all customers
        customers = db.query(Customer).all()
        
        if not customers:
            raise HTTPException(status_code=404, detail="No customers found in the database")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'age': c.age,
            'gender': c.gender,
            'tenure': c.tenure,
            'monthly_charges': float(c.monthly_charges),
            'total_charges': float(c.total_charges),
            'contract_type': c.contract_type,
            'payment_method': c.payment_method,
            'churn': bool(c.churn)
        } for c in customers])
        
        # Extract features
        features_df = feature_engineering.extract_features(df)
        
        # Get feature importance
        X = features_df.drop(columns=['churn'])
        y = features_df['churn']
        importance = feature_engineering.analyze_feature_importance(X, y)
        
        # Get correlations and convert to dict
        correlations = feature_engineering.get_feature_correlations(features_df)
        correlations_dict = {
            col: {idx: float(val) for idx, val in row.items()}
            for col, row in correlations.to_dict().items()
        }
        
        # Perform PCA
        pca_results = feature_engineering.perform_pca(X)
        
        # Ensure PCA results are JSON-serializable
        if isinstance(pca_results, dict):
            if "components" in pca_results:
                pca_results["components"] = [
                    [float(x) for x in row] for row in pca_results["components"]
                ]
            if "explained_variance" in pca_results:
                pca_results["explained_variance"] = [
                    float(x) for x in pca_results["explained_variance"]
                ]
        
        response = {
            "feature_importance": importance,
            "correlations": correlations_dict,
            "pca_results": pca_results,
            "total_customers": len(customers),
            "features_analyzed": list(X.columns)
        }
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in feature analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/train")
async def train_models(db: Session = Depends(get_db)):
    """
    Train ML models on customer data
    """
    try:
        # Get all customers
        customers = db.query(Customer).all()
        
 # Convert to DataFrame
        df = pd.DataFrame([{
            'age': c.age,
            'gender': c.gender,
            'tenure': c.tenure,
            'monthly_charges': float(c.monthly_charges),
            'total_charges': float(c.total_charges),
            'contract_type': c.contract_type,
            'payment_method': c.payment_method,
            'churn': bool(c.churn)
        } for c in customers])        
        # Extract features
        features_df = feature_engineering.extract_features(df)
        
        # Prepare data
        X_train, X_test, y_train, y_test = model_training.prepare_data(
            features_df, 'churn'
        )
        
        # Train models
        cv_results = model_training.train_models(X_train, y_train)
        
        # Evaluate best model
        metrics = model_training.evaluate_model(
            model_training.best_model, X_test, y_test
        )
        
        # Save model
        model_path = model_training.save_model(
            model_training.best_model,
            model_training.best_model_name,
            metrics
        )
        
        return {
            "cv_results": cv_results,
            "best_model": model_training.best_model_name,
            "metrics": metrics,
            "model_path": model_path
        }
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_customer_data(customer_id: str, db: Session) -> Dict:
    """Helper function to get customer data by ID"""
    # Try to find by customer_id first
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    
    # If not found, try to find by numeric ID
    if not customer:
        try:
            customer_id_int = int(customer_id)
            customer = db.query(Customer).filter(Customer.id == customer_id_int).first()
        except ValueError:
            pass
    
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer not found with ID: {customer_id}")
    
    return {
        'customer_id': customer.customer_id,
        'age': customer.age,
        'gender': customer.gender,
        'tenure': customer.tenure,
        'monthly_charges': customer.monthly_charges,
        'total_charges': customer.total_charges,
        'contract_type': customer.contract_type,
        'payment_method': customer.payment_method,
        'churn': customer.churn
    }

@router.post("/llm/analyze-behavior/{customer_id}")
async def analyze_customer_behavior(customer_id: str, db: Session = Depends(get_db)):
    """
    Analyze customer behavior using LLM
    """
    try:
        customer_data = get_customer_data(customer_id, db)
        
        # Get LLM analysis
        analysis = llm_service.analyze_customer_behavior(customer_data)
        
        if "error" in analysis:
            raise HTTPException(status_code=503, detail=analysis["error"])
        
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in customer behavior analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/llm/generate-report/{customer_id}")
async def generate_customer_report(customer_id: str, db: Session = Depends(get_db)):
    """
    Generate detailed customer report using LLM
    """
    try:
        customer_data = get_customer_data(customer_id, db)
        
        # Generate report
        report = llm_service.generate_customer_report(customer_id, customer_data)
        
        if "error" in report:
            raise HTTPException(status_code=503, detail=report["error"])
        
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in report generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/llm/analyze-feedback")
async def analyze_feedback(feedback: str):
    """
    Analyze customer feedback using LLM
    """
    try:
        analysis = llm_service.analyze_feedback(feedback)
        
        if "error" in analysis:
            raise HTTPException(status_code=503, detail=analysis["error"])
        
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in feedback analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/llm/generate-recommendations/{customer_id}")
async def generate_recommendations(customer_id: str, db: Session = Depends(get_db)):
    """
    Generate personalized recommendations using LLM
    """
    try:
        customer_data = get_customer_data(customer_id, db)
        
        # Generate recommendations
        recommendations = llm_service.generate_recommendations(customer_data)
        
        if "error" in recommendations:
            raise HTTPException(status_code=503, detail=recommendations["error"])
        
        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in recommendation generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/llm/query")
async def natural_language_query(query: str, db: Session = Depends(get_db)):
    """
    Answer natural language queries about customer data
    """
    try:
        # Get all customers
        customers = db.query(Customer).all()
        
        # Convert to list of dicts
        customer_data = [{
            'customer_id': c.customer_id,
            'age': c.age,
            'gender': c.gender,
            'tenure': c.tenure,
            'monthly_charges': c.monthly_charges,
            'total_charges': c.total_charges,
            'contract_type': c.contract_type,
            'payment_method': c.payment_method,
            'churn': c.churn
        } for c in customers]
        
        # Get answer
        answer = llm_service.answer_natural_language_query(query, customer_data)
        
        if "error" in answer:
            raise HTTPException(status_code=503, detail=answer["error"])
        
        return answer
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 