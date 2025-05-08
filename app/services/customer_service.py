from sqlalchemy.orm import Session
from app.db.models import Customer
from app.schemas import CustomerCreate
from typing import List, Optional

class CustomerService:
    @staticmethod
    def create_customer(db: Session, customer: CustomerCreate) -> Customer:
        db_customer = Customer(**customer.model_dump())
        db.add(db_customer)
        db.commit()
        db.refresh(db_customer)
        return db_customer

    @staticmethod
    def get_customer(db: Session, customer_id: int) -> Optional[Customer]:
        return db.query(Customer).filter(Customer.id == customer_id).first()

    @staticmethod
    def get_customer_by_customer_id(db: Session, customer_id: str) -> Optional[Customer]:
        return db.query(Customer).filter(Customer.customer_id == customer_id).first()

    @staticmethod
    def get_customers(db: Session, skip: int = 0, limit: int = 100) -> List[Customer]:
        return db.query(Customer).offset(skip).limit(limit).all()

    @staticmethod
    def update_customer(db: Session, customer_id: int, customer: CustomerCreate) -> Optional[Customer]:
        db_customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if db_customer:
            for key, value in customer.model_dump().items():
                setattr(db_customer, key, value)
            db.commit()
            db.refresh(db_customer)
        return db_customer

    @staticmethod
    def delete_customer(db: Session, customer_id: int) -> bool:
        db_customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if db_customer:
            db.delete(db_customer)
            db.commit()
            return True
        return False 