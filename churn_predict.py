import os
import pandas as pd
import pickle
import time
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

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
    
    df = df.drop(columns='unnamed: 21')
    
    df.loc[:,'churn'] = (df['attrition_flag'] == 'Attrited Customer').astype(int)
    df = df.drop(columns='attrition_flag')
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

def classify(customer={}):
    path = os.path.join('data','BankChurners.csv')
    df = preprocess_raw(pd.read_csv(path))

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up sklearn pipeline to organise ml training steps
    num_features = X_train.select_dtypes(include=['int','float']).columns.tolist()
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())])
    # Ordinal features
    ord_features = ['education_level']
    ord_transformer = OrdinalEncoder_custom(categories=[['Unknown','Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate']])
    # Categorical preprocessor
    cat_features = ['gender', 'marital_status', 'income_category', 'card_category']
    cat_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
    # Combined preprocessor
    preprocessor = ColumnTransformer([('ords', ord_transformer, ord_features),
                                    ('cats', cat_transformer, cat_features),
                                    ('nums', num_transformer, num_features)],
                                    remainder='drop')
    clf = Pipeline([('preprocessor', preprocessor),
                        ('xgbclf', XGBClassifier(eval_metric='logloss', use_label_encoder=False))])

    start = time.time()
    clf.fit(X_train,y_train)
    end = time.time()
    print(f'Time to train model: {end-start:.2f} seconds')

    # X_pred = 
    y = clf.predict(X_test)
    return y

def classify_pretrained(df):
    print(df.iloc[0])
    df = preprocess_raw(df.iloc[0])
    print(df)
    X = df
    print(len(X))
    if len(X) > 1:
        raise ValueError('Classify method only works with a single customer record')
    
    with open('model.pkl','rb') as f:
        clf = pickle.load(f) 
    y = clf.predict(X)

    churn = True if y[0] > 1 else False
    return churn


if __name__ == '__main__':
    # for testing, load a preprocessed dataset
    path = os.path.join('data','BankChurners.csv')
    df = preprocess_raw(pd.read_csv(path))
    # print(df.head())

    churn = classify_pretrained(df)
    print(churn)