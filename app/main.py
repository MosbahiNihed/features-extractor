from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import customers, ml
from app.api.prediction_endpoints import router
from app.db.database import engine
from app.models import Base
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Feature Engineering API",
    description="API for customer feature engineering and ML model training",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(customers.router, prefix="/api/v1", tags=["customers"])
app.include_router(ml.router, prefix="/api/v1", tags=["insights"])
app.include_router(router, prefix="/api/v1", tags=["predict"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Feature Engineering API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 