from shop.models import Customer
from shop.service import checkout_total


def test_checkout_total() -> None:
    assert checkout_total(100, Customer("VIP")) == 90
