from shop.models import Customer
from shop.pricing import calculate_discount


def checkout_total(total: float, customer: Customer) -> float:
    return calculate_discount(total, customer.customer_type)
