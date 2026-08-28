import pandas as pd
import numpy as np

def load_data(path : str)-> pd.DataFrame :
    print(f"Loading Data From Path : {path}")
    df = pd.read_csv(path)
    return df

def null_pct(df : pd.DataFrame) -> pd.DataFrame :
    print(f"Calculating Null Percentage For DataFrame")
    df = df.copy()

    null_count = df.isna().sum()
    null_percent = (df.isna().sum() / len(df)) * 100

    null_df = pd.DataFrame({
        'null_count' : null_count,
        'null_percent' : null_percent
    })

    return null_df[null_df['null_count'] > 0]

def log_transform(df : pd.DataFrame , cols : list[str]) -> pd.DataFrame : 
    df = df.copy()
    for col in cols:
        if col not in df.columns:
                continue
        df[f'log_{col}'] = np.log1p(df[col])
    return df

def transform_target(df : pd.DataFrame , target_col : str) -> pd.DataFrame :
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target Column '{target_col}' Not Found In DataFrame")
    df[f'log_{target_col}'] = np.log1p(df[target_col])

    price_upper_limit = df[target_col].quantile(0.98)
    df = df[(df[target_col] <= price_upper_limit) & (df[target_col] >= 8000.0)]

    df = df.dropna(subset = ['price'])

    return df