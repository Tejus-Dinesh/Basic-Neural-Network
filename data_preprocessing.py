import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import KNNImputer


# def preprocessing(df):
#     cat_cols = ['Delivery phase', 'Community', 'IFA', 'Education', 'Residence']
#     for column in cat_cols:
#         df[column].fillna(df[column].mode()[0], inplace=True)

#     numeric_integer_col = ['Age', 'Weight']
#     for col in numeric_integer_col:
#         df[col].fillna((round(df[col].mean())), inplace=True)

#     df['HB'].fillna(round(df['HB'].mean(), 1), inplace=True)
#     df['BP'].fillna(round(df['BP'].mean(), 5), inplace=True)
#     return df

if os.path.exists("LBW_processed.csv"):
    print("Pre-Processed csv already exists")
else:
    df = pd.read_csv('LBW_Dataset.csv')
    imputer = KNNImputer(n_neighbors=3, weights="uniform")
    df1 = pd.DataFrame(imputer.fit_transform(df))
    df1.columns = list(df.columns)
    df2 = pd.get_dummies(df1, columns=["Community"], prefix="Community")
    scaler = preprocessing.MinMaxScaler()
    df3 = pd.DataFrame(scaler.fit_transform(df2.values))
    df3.columns = list(df2.columns)
    df3.to_csv('LBW_processed.csv', index=False, header=True, encoding='utf-8')
