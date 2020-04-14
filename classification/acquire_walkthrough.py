import pandas as pd 
from env import host, user, password

def get_connection(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
    sql_query = 'SELECT * FROM passengers'
    return pd.read_sql(sql_query, get_connection('titanic_db'))

def get_iris_data():
    sql_query = """
    SELECT species_id,
    species_name,
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
    FROM measurements
    JOIN species
    USING(species_id)
    """
    return pd.read_sql(sql_query, get_connection('iris_db'))