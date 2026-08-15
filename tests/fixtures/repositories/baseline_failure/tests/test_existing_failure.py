from app import divide


def test_existing_failure() -> None:
    assert divide(4, 2) == 3
