import pandas as pd
import env
from os import path

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_store_data(cache=True):
    csv_file_path = './store_item_demand.csv'

    query = '''
    SELECT
        sales.*,
        items.item_brand,
        items.item_name,
        items.item_price,
        stores.store_address,
        stores.store_zipcode,
        stores.store_city,
        stores.store_state
    FROM sales
    JOIN items USING(item_id)
    JOIN stores USING(store_id)
    '''

    if cache and path.exists(csv_file_path):
        return pd.read_csv(csv_file_path)
    else:
        df = pd.read_sql(query, get_connection('tsa_item_demand'))
        df.to_csv(csv_file_path, index=False)
        return df
