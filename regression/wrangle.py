import pandas as pd
import numpy as np

from env import host, user, password

# function to get the url

def get_db_url(db_name):
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"

# function that passes my query and my url to return df

def get_data_from_sql():
    query = """
    SELECT customer_id, monthly_charges, tenure, total_charges
    FROM customers
    WHERE contract_type_id = 3;
    """
    df = pd.read_sql(query, get_db_url('telco_churn'))
    return df

# function that rules them all by acquiring and prepping my df for exploration or modeling

def wrangle_telco():
    """
    Queries the telco_churn database
    Returns a clean df with four columns:
    customer_id(object), monthly_charges(float), tenure(int), total_charges(float)
    """
    df = get_data_from_sql()
    df.tenure.replace(0, 1, inplace=True)
    df.total_charges.replace(' ', df.monthly_charges, inplace=True)
    df.total_charges = df.total_charges.astype(float)
    return df