
from synthetic_pred_module.calculator import (
	add, divide, multiply, myfunc, subtract
)
import pandas as pd
print(add(1, 2))
print(subtract(1, 2))
print(multiply(1, 2))
print(divide(1, 2))
print("Hello World!")
print(f"Pandas version is: {pd.__version__}")
myfunc()
