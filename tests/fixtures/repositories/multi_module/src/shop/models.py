from dataclasses import dataclass


@dataclass(frozen=True)
class Customer:
    customer_type: str
