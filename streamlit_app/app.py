# General Libraries
import pickle
import pandas as pd

# Model deployment
import streamlit as st

model = pickle.load(open('final_model.pkl', 'rb'))
X_holdout = pd.read_csv('holdout.csv', index_col=0).reset_index(drop=True)
holdout_indices = X_holdout.index.to_list()
st.title("Food Insecurity Identification")
html_temp = """
<div style="background:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;"> Food Insecurity Identification ML App </h2>
</div>
"""
st.dataframe(X_holdout)
st.markdown(html_temp, unsafe_allow_html = True)

#adding a selectbox
choice = st.selectbox(
    "Select Index Number:",
    options = holdout_indices)

def predict_if_fi(index):    
    respondent = X_holdout.loc[[index]]
    st.dataframe(respondent)
    prediction_num = model.predict(respondent)[0]
    pred_map = {1: 'Food Insecure', 0: 'Food Secure'}
    prediction = pred_map[prediction_num]
    return prediction

if st.button("Predict"):
    output = predict_if_fi(choice)
    if output == 'Food Insecure':
        st.error('This respondent may be Food Insecure', icon="🚨")
    elif output == 'Food Secure':
        st.success('This respondent is Food Secure', icon="✅")