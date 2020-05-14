import pandas as pd

def remove_columns(df, cols_to_remove):
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def prep_store_data_prophet(df: pd.DataFrame) -> pd.DataFrame:
    return (df.assign(ds=pd.to_datetime(df.sale_date)).sort_values('ds')
            .assign(y=df.sale_amount * df.item_price)\
            .groupby(['ds'])['y'].sum().reset_index().set_index('ds'))

def prep_store_data(df):
    # parse the date column and set it as the index
    # fmt = '%a, %d %b %Y %H:%M:%S %Z'
    # df.sale_date = pd.to_datetime(df.sale_date, format=fmt)
    df.sale_date = pd.to_datetime(df.sale_date)
    df = df.sort_values(by='sale_date').set_index('sale_date')

    # add some time components as features
    # df['month'] = df.index.strftime('%m-%b')
    # df['weekday'] = df.index.strftime('%w-%a')

    # derive the total sales
    df['sales_total'] = df.sale_amount * df.item_price
    
    return df

def get_sales_by_day(df):
    sales_by_day = df.resample('D')[['sales_total']].sum()
    sales_by_day['diff_with_last_day'] = sales_by_day.sales_total.diff()
    return sales_by_day
    
def split_store_data(df, train_prop=.66): 
    train_size = int(len(df) * train_prop)
    train, test = df[0:train_size].reset_index(), df[train_size:len(df)].reset_index()
    return train, test
