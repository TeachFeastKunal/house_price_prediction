import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd


def plot_univariate_numerical(df : pd.DataFrame, col : str , ax : Axes = None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x=col, kde=True, ax=ax, **kwargs)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    return ax

def plot_univariate_categorical(df : pd.DataFrame, col : str, ax : Axes =None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data = df , x= col, ax=ax, **kwargs)
    ax.set_title(f"Value counts of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_xticklabels(rotation=45)
    return ax

def plot_bivariate_num_num(df : pd.DataFrame, x : str, y : str, ax : Axes =None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x=x, y=y, ax=ax, **kwargs)
    ax.set_title(f"{y} vs {x}")
    return ax

def plot_bivariate_cat_num(df : pd.DataFrame, cat : str, num : str, ax : Axes =None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df.sample(min(5000, len(df)) , random_state = 42), x=cat, y=num, ax=ax, **kwargs)
    ax.set_title(f"{num} by {cat}")
    ax.set_xticklabels(rotation=45)
    return ax

def plot_correlation_heatmap(df : pd.DataFrame, cols : list[str], ax : Axes =None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap="rocket", fmt = '.2f' , ax=ax, **kwargs)
    ax.set_title("Correlation Heatmap")
    return ax