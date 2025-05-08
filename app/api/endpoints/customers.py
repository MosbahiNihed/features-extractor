from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.db.database import get_db
from app.schemas import Customer, CustomerCreate
from app.services.customer_service import CustomerService

router = APIRouter()

@router.post("/")
def create_customer(customer: CustomerCreate, db: Session = Depends(get_db)):
    """
    Create a new customer
    """
    db_customer = CustomerService.get_customer_by_customer_id(db, customer.customer_id)
    if db_customer:
        raise HTTPException(status_code=400, detail="Customer ID already registered")
    return CustomerService.create_customer(db, customer)

@router.get("/")
def read_customers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve customers
    """
    customers = CustomerService.get_customers(db, skip=skip, limit=limit)
    return customers

@router.get("/{customer_id}")
def read_customer(customer_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a specific customer by ID
    """
    db_customer = CustomerService.get_customer(db, customer_id)
    if db_customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return db_customer

@router.put("/{customer_id}", response_model=Customer)
def update_customer(customer_id: int, customer: CustomerCreate, db: Session = Depends(get_db)):
    """
    Update a customer
    """
    db_customer = CustomerService.update_customer(db, customer_id, customer)
    if db_customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return db_customer

@router.delete("/{customer_id}")
def delete_customer(customer_id: int, db: Session = Depends(get_db)):
    """
    Delete a customer
    """
    success = CustomerService.delete_customer(db, customer_id)
    if not success:
        raise HTTPException(status_code=404, detail="Customer not found")
    return {"message": "Customer deleted successfully"} 