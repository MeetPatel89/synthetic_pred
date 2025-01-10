def add(a: float, b: float) -> float:
    try:
        return a + b
    except TypeError:
        raise ValueError("Both arguments must be numbers")


def subtract(a: float, b: float) -> float:
    try:
        return a - b
    except TypeError:
        raise ValueError("Both arguments must be numbers")


def multiply(a: float, b: float) -> float:
    try:
        return a * b
    except TypeError:
        raise ValueError("Both arguments must be numbers")


def divide(a: float, b: float) -> float:
    try:
        if b == 0:
            raise ValueError("The divisor must not be zero")
        return a / b
    except TypeError:
        raise ValueError("Both arguments must be numbers")
