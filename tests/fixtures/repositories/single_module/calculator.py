def calculate_discount(total: float, customer_type: str) -> float:
    if customer_type == "VIP":
        return total * 0.9
    return total
