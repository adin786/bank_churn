import streamlit as st
import pickle
import pandas as pd
from clf_funcs import classify, OrdinalEncoder_custom
import numpy as np

st.write('''
# Bank Customers - Churn modelling
## Background
Classifier model trained on 10,000 customer records.  Predicts whether the customer is likely to churn or not.

## Run a prediction
''')

col1, col2 = st.columns(2)

with col1:
    clientnum = st.number_input(label='CLIENTNUM:', value=768805383, format='%d', min_value=0)

    customer_age = st.slider(label='Customer_Age:', value=45, min_value=1, max_value=100)

    gender = st.radio(label='Gender:', options=['M (Male)','F (Female)'])

    dependent_count = st.slider(label='Dependent_count:', value=3, min_value=0, max_value=7)

    education_level = st.selectbox(label='Education_Level:', index=0, 
        options=['High School', 'Graduate', 'Uneducated', 'College', 'Post-Graduate', 'Doctorate', 'Unknown']
    )

    marital_status = st.selectbox(label='Marital_Status:', index=0, options=['Married', 'Single', 'Divorced', 'Unknown'])

    income_category = st.select_slider(label='Income_Category', value='$60K - $80K', options=['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', 'Unknown'])

    card_category = st.selectbox(label='Card_Category', index=0, options=['Blue', 'Gold', 'Silver', 'Platinum'])

    months_on_book = st.number_input(label='Months_on_book', value=39, min_value=0, max_value=100)

    total_relationship_count = st.number_input('Total_Relationship_Count', value=5, min_value=0, max_value=10)

with col2:
    months_inactive_12_mon = st.number_input('Months_Inactive_12_mon', value=1, min_value=0, max_value=12)

    contacts_count_12_mon = st.number_input('Contacts_Count_12_mon', value=3, min_value=0, max_value=10)

    credit_limit = st.number_input('Credit_Limit', value=12691, min_value=0)

    total_revolving_bal = st.number_input('Total_Revolving_Bal', value=777, min_value=0)

    avg_open_to_buy = st.number_input('Avg_Open_To_Buy', value=11914, min_value=0)

    total_amt_change_q1_q4 = st.number_input('Total_Amt_Chng_Q4_Q1', value=1.335, min_value=0.0)

    total_trans_amt = st.number_input('Total_Trans_Amt', value=1144, min_value=0)

    total_trans_ct = st.number_input('Total_Trans_Ct', value=42, min_value=0)

    total_ct_chng_q4_q1 = st.number_input('Total_Ct_Chng_Q4_Q1', value=1.625, min_value=0.0)

    avg_utilisation_ratio = st.number_input('Avg_Utilization_Ratio', value=0.061, min_value=0.0)


if st.button('Compute a prediction'):

    with st.spinner('In progress'):
        churn = classify(
            clientnum, 
            None, 
            customer_age,
            gender,
            dependent_count,
            education_level,
            marital_status,
            income_category,
            card_category,
            months_on_book,
            total_relationship_count,
            months_inactive_12_mon,
            contacts_count_12_mon,
            credit_limit,
            total_revolving_bal,
            avg_open_to_buy,
            total_amt_change_q1_q4,
            total_trans_amt,
            total_trans_ct,
            total_ct_chng_q4_q1,
            avg_utilisation_ratio,
        )
        st.success('Done')

    if churn:
        st.write('This customer is **PREDICTED TO CHURN**')
    else:
        st.write('This customer is **PREDICTED NOT TO CHURN**')