from calculator import calculate_discount


def test_vip_discount() -> None:
    assert calculate_discount(100, "VIP") == 90
