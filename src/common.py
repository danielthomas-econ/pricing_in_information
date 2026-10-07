"""Small helpers shared across the files"""

from pathlib import Path
from typing import cast

import numpy as np

# figures and other exported results go in src/outputs, wherever the notebook is run from
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# allows us to deal with the edge cases of p = 0 and p = 1 (where 1-p=0)
# avoding log2(0) undefined problems
def binary_entropy(p):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(p > 0, p * np.log2(p), 0.0)
        term2 = np.where(p < 1, (1 - p) * np.log2(1 - p), 0.0)
    return -(term1 + term2)


# an alternative to panda's insert(args) since it throws an error if the column already exists
# same logic but instead of df.insert, we use insert_or_replace(df, args)
def insert_or_replace(df, position, name, values):
    if name in df.columns:
        df.drop(columns=name, inplace=True)
    df.insert(position, name, values)


# gives the position to insert columns after a specified column
# for some reason pylance never lets me use .get_loc() + 1
def nextpos(df, col: str) -> int:
    pos = cast(int, df.columns.get_loc(col))
    return pos + 1


# if i ever need to swap cols
def swapcolumns(df, col1: str, col2: str):
    columns = list(df.columns)
    new = columns.index(col1)
    old = columns.index(col2)

    columns[new], columns[old] = columns[old], columns[new]

    df = df[columns]
    return df


# just to ensure our df isnt getting bloated anywhere
def check_memory(df):
    print(df.memory_usage(deep=True) / 1e6)  # MB per column
    print()
    print(f"Total memory used: {df.memory_usage(deep=True).sum()/1e6}")