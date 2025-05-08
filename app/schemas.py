from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class CustomerBase(BaseModel):
    customer_id: str
    age: int
    gender: str
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    payment_method: str
    churn: bool

class CustomerCreate(CustomerBase):
    pass

class Customer(CustomerBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class FeatureSetBase(BaseModel):
    name: str
    description: Optional[str] = None
    features: str  # JSON string of feature names

class FeatureSetCreate(FeatureSetBase):
    pass

class FeatureSet(FeatureSetBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ModelBase(BaseModel):
    name: str
    version: str
    feature_set_id: int
    model_type: str
    metrics: str  # JSON string of model metrics

class ModelCreate(ModelBase):
    pass

class Model(ModelBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PredictionRequest(BaseModel):
    customer_data: Dict[str, Any]

class PredictionResponse(BaseModel):
    churn_prediction: bool
    churn_probability: float

class TrainingResponse(BaseModel):
    message: str
    metrics: Dict[str, float] 