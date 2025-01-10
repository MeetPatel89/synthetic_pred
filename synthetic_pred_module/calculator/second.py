import os

import pandas as pd


def myfunc() -> None:
    print("Hello from myfunc in second.py")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Panadas version: {pd.__version__}")
