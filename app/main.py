from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import customers
from app.db.database import engine, Base

app = FastAPI(
    title="Feature Engineering API",
    description="API for customer churn prediction with feature engineering",
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
app.include_router(customers.router, prefix="/api/v1/customers", tags=["customers"])

@app.on_event("startup")
async def startup_event():
    # Create database tables
    Base.metadata.create_all(bind=engine)

@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup resources if needed
    pass

@app.get("/")
async def root():
    return {"message": "Welcome to Feature Engineering API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 