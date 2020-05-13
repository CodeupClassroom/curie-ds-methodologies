import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('figure', figsize=(11, 9))
plt.rc('font', size=13)

import requests
import os
from datetime import timedelta, datetime as dt

from acquire_walkthrough import get_store_data

# Function for numeric distributions

def numeric_hists(df, bins=20):
    """
    Function to take in a DataFrame, bins default 20,
    select only numeric dtypes, and
    display histograms for each numeric column
    """
    num_df = df.select_dtypes(include=np.number)
    num_df.hist(bins=bins, color='thistle')
    plt.suptitle('Numeric Column Distributions')
    plt.show()

# Function to acquire and prep store df

def prepped_store_df():
    """
    Function to acquire and prepare
    store dataframe and show
    distributions for numeric columns
    """
    # Acquire the df
    df = df.get_store_data()
    
    # Create date part columns
    df['month'] = df.index.month
    df['weekday'] = df.index.day_name()
    
    # Create calculated columns
    df = df.assign(sales_total = df.sale_amount * df.item_price)
    df = df.assign(sales_diff = df.sales_total.diff(periods=1))
    
    # Change dtypes of numeric columns to object and category
    df = (df.astype({'sale_id': object, 'store_id': object, 
                     'store_zipcode': object, 'item_id': object, 
                     'item_upc12': object, 'item_upc14': object, 
                     'month': 'category', 'weekday': 'category'}))
    
    # Display distributions of numeric columns
    numeric_hists(df)
    
    return df