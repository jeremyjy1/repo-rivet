def calculate_discount(price: float, rate: float) -> float:
    """Return the price after applying a fractional discount."""
    return price * (1 - rate)
