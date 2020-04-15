import pandas as pd
import numpy as np
import scipy as sp 

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

def label_encode(train, test):
    le = LabelEncoder()
    train['species'] = le.fit_transform(train.species)
    test['species'] = le.transform(test.species)
    return le, train, test


def prep_iris(df):
    df = df.drop(columns='species_id')
    df = df.rename(columns={'species_name': 'species'})
    train, test = train_test_split(df, train_size=.75, stratify=df.species, random_state=123)
    train, test, le = label_encode(train, test)
    return train, test, le


def inverse_encode(train, test, le):
    train['species'] = le.inverse_transform(train.species)
    test['species'] = le.inverse_transform(test.species)
    return train, test

def drop_columns(df):
    df.drop(columns=['deck'], inplace=True)
    return df

def impute_embark_town(train, test):
    train['embark_town'] = train['embark_town'].fillna('Southampton')
    test['embark_town'] = test['embark_town'].fillna('Southampton')
    return train, test

def impute_embarked(train, test):
    train['embarked'] = train['embarked'].fillna('S')
    test['embarked'] = test['embarked'].fillna('S')
    return train, test

def impute_age(train, test):
    avg_age = train.age.mean()
    train.age = train.age.fillna(avg_age)
    test.age = test.age.fillna(avg_age)
    return train, test

def scale_columns(train, test):
    scaler = MinMaxScaler()
    train[['age','fare']] = scaler.fit_transform(train[['age','fare']])
    test[['age','fare']] = scaler.transform(test[['age','fare']])
    return scaler, train, test

def ohe_columns(train, test):
    # create encoder
    ohe = OneHotEncoder(sparse=False, categories='auto')
    
    # fit scaler on train and transform train and test to dense matrices
    train_matrix = ohe.fit_transform(train[['embarked']])
    test_matrix = ohe.transform(test[['embarked']])
    
    # transform matrices to DataFrames
    train_ohe = pd.DataFrame(train_matrix, columns=ohe.categories_[0], index=train.index)
    test_ohe = pd.DataFrame(test_matrix, columns=ohe.categories_[0], index=test.index)
    
    # join encoded matrix with original train or test matrices
    train = train.join(train_ohe)
    test = test.join(test_ohe)
    
    return ohe, train, test

def prep_titanic(df):

    # drop the deck column bc most values Null
    drop_columns(df)
    
    train, test = train_test_split(df, train_size=.75, stratify=df.survived, random_state=123)
    
    # impute 2 NaNs in embark_town with most frequent value
    train, test = impute_embark_town(train, test)
    
    # impute 2 NaNs in embarked with most frequent value
    train, test = impute_embarked(train, test)
    
    # impute NaNs in age in train and test with the mean age in train
    train, test = impute_age(train, test)
    
    # use a minmax scaler on age and fare bc of differing measurement units
    scaler, train, test = scale_columns(train, test)
    
    # ohe embarked creating three new columns for C, Q, S representing embark towns
    ohe, train, test = ohe_columns(train, test)
    
    return scaler, ohe, train, test

