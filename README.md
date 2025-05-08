# Feature Engineering Project

A comprehensive feature engineering and machine learning pipeline for customer churn prediction.

## Project Structure

```
.
├── app/                    # Main application code
│   ├── api/               # FastAPI endpoints
│   ├── core/              # Core application logic
│   ├── db/                # Database models and migrations
│   ├── ml/                # Machine learning models
│   └── services/          # Business logic services
├── data/                  # Data storage
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
├── notebooks/            # Jupyter notebooks for EDA
├── scripts/              # Utility scripts
├── tests/                # Test files
├── docker/               # Docker configuration
└── liquibase/           # Database migrations
```

## Features

- Data Collection & Storage
- Data Preprocessing & Feature Engineering
- Exploratory Data Analysis (EDA)
- Machine Learning Model Training
- Model Evaluation & Optimization
- FastAPI Backend
- Docker Containerization
- Liquibase Database Migrations

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
4. Run with Docker:
   ```bash
   docker-compose up --build
   ```

## Development

- FastAPI backend runs on port 8000
- API documentation available at `/docs`
- Database migrations managed by Liquibase
- ML models are versioned and tracked

## Technologies Used

- Python 3.9+
- FastAPI
- PostgreSQL
- Liquibase
- Docker
- Scikit-learn
- Pandas
- NumPy
- Jupyter
