import pandas as pd
import random

def augment(table: pd.DataFrame, op: str):
    if op == 'sample_table':
        table = table.copy()
        if len(table) > 5 and len(table.columns) > 2:
            sample_size = random.randint(1, len(table)-1)
            table = table.sample(n=sample_size, replace=True)
            
            num_to_drop = random.randint(1, len(table.columns) - 1)
            cols_to_drop = random.sample(list(table.columns), num_to_drop)
            table = table.drop(columns=cols_to_drop)
    else:
        raise ValueError(f"Unsupported operation: {op}")

    return table