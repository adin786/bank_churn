import os
import pandas as pd
import pickle
import time
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

class OrdinalEncoder_custom(TransformerMixin, BaseEstimator):
    '''wraps OrdinalEncoder to enable .get_feature_names_out() method.'''
    def __init__(self, categories='auto'):
        self.categories = categories
        self.encoder = OrdinalEncoder(categories=categories)
    
    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        self.feature_names = X.columns.tolist()
        return self
    
    def transform(self, X, y=None):
        return self.encoder.transform(X)
        
    def fit_transform(self, X, y=None):
        self.feature_names = X.columns.tolist()
        return self.encoder.fit_transform(X, y)

    def get_feature_names_out(self):
            return self.feature_names

def fill_marital_status(df):
  '''Replace Unknown with Married.'''
  df.loc[:,'marital_status'] = df.marital_status.replace('Unknown','Married')
  return df

def fill_incomes(df):
  '''Replace Unknown with "less than 40k".'''
  df.loc[:,'income_category'] = df.income_category.replace('Unknown','Less than $40K')
  return df

def preprocess_raw(df):
    df.columns = df.columns.str.lower()
    
    df = df.drop(columns='unnamed: 21') if 'unnamed: 21' in df.columns else df
    
    df.loc[:,'churn'] = (df['attrition_flag'] == 'Attrited Customer').astype(int)
    df = df.drop(columns='attrition_flag') if 'attrition_flag' in df.columns else df
    df.churn.value_counts()
    
    df.loc[:,'gender'] = df.gender.astype('category')
    df.loc[:,'education_level'] = df.education_level.astype('category')
    df.loc[:,'marital_status'] = df.marital_status.astype('category')
    df.loc[:,'income_category'] = df.income_category.astype('category')
    df.loc[:,'card_category'] = df.card_category.astype('category')
    
    df = fill_marital_status(df)
    
    df = fill_incomes(df)

    df.loc[:,'tenure_per_age'] = df.months_on_book / (df.customer_age * 12)
    df.loc[:,'utilisation_per_age'] = df.avg_utilization_ratio / df.customer_age
    df.loc[:,'credit_lim_per_age'] = df.credit_limit / df.customer_age
    df.loc[:,'total_trans_amt_per_credit_lim'] = df.total_trans_amt / df.credit_limit
    df.loc[:,'total_trans_ct_per_credit_lim'] = df.total_trans_ct / df.credit_limit
    
    df = df.drop(columns='clientnum')
    
    return df

def predict(X):
    '''Predict a result for the given X values'''
    if len(X) > 1:  
        raise ValueError('Classify method only works with a single customer record')
    with open('model.pkl','rb') as f:
        clf = pickle.load(f) 

    print(f'Preprocessed data (X): shape {X.shape}')
    print(X.T)
    y = clf.predict(X)
    churn = True if y[0] > 1 else False
    return churn

def classify(CLIENTNUM=768805383,
            Attrition_Flag='Existing Customer',
            Customer_Age=45,
            Gender='M',
            Dependent_count=3,
            Education_Level='High School',
            Marital_Status='Married',
            Income_Category='$60k - 80k',
            Card_Category='Blue',
            Months_on_book=39,
            Total_Relationship_Count=5,
            Months_Inactive_12_mon=1,
            Contacts_Count_12_mon=3,
            Credit_Limit=12691,
            Total_Revolving_Bal=777,
            Avg_Open_To_Buy=11914,
            Total_Amt_Chng_Q4_Q1=1.335,
            Total_Trans_Amt=1144,
            Total_Trans_Ct=42,
            Total_Ct_Chng_Q4_Q1=1.625,
            Avg_Utilization_Ratio=0.061):
    '''Classifies a single customer record with a churn prediction True or False'''

    ser = pd.Series({
        'CLIENTNUM':CLIENTNUM,
        'Attrition_Flag':Attrition_Flag,
        'Customer_Age':Customer_Age,
        'Gender':Gender,
        'Dependent_count':Dependent_count,
        'Education_Level':Education_Level,
        'Marital_Status':Marital_Status,
        'Income_Category':Income_Category,
        'Card_Category':Card_Category,
        'Months_on_book':Months_on_book,
        'Total_Relationship_Count':Total_Relationship_Count,
        'Months_Inactive_12_mon':Months_Inactive_12_mon,
        'Contacts_Count_12_mon':Contacts_Count_12_mon,
        'Credit_Limit':Credit_Limit,
        'Total_Revolving_Bal':Total_Revolving_Bal,
        'Avg_Open_To_Buy':Avg_Open_To_Buy,
        'Total_Amt_Chng_Q4_Q1':Total_Amt_Chng_Q4_Q1,
        'Total_Trans_Amt':Total_Trans_Amt,
        'Total_Trans_Ct':Total_Trans_Ct,
        'Total_Ct_Chng_Q4_Q1':Total_Ct_Chng_Q4_Q1,
        'Avg_Utilization_Ratio':Avg_Utilization_Ratio
    })
    X_raw = pd.DataFrame(ser).T
    X = preprocess_raw(X_raw)

    churn = predict(X)
    print(f'\nPredicted churn status = {churn}')
    return churn

    

if __name__ == '__main__':
    print('main')
    churn = classify()

else:
    print('not main')
