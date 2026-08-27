import pytest
from discount import calculate_discount


def test_calculate_discount() -> None:
    assert calculate_discount(100, 0.2) == 80


def test_reject_negative_price() -> None:
    with pytest.raises(ValueError, match="price"):
        calculate_discount(-100, 0.2)
