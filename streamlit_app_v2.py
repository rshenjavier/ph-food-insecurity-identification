# General libraries
!pip install matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
import pickle

# Model deployment
from flask import Flask
import streamlit as st

def load_data():
    # Load the data
    data = pd.read_csv(
        "phl_hfs_r1_household_survey_data.csv",
        encoding='ISO-8859-1'
    )
    return data


def introduction():
    # Load photo
    st.image("slides/title.png")

    st.markdown('''Food insecurity is a condition where individuals, households, or communities do not have reliable access to sufficient, safe, and nutritious food. 
    It can be caused by a range of factors, including poverty, lack of access to food markets, and natural disasters that disrupt food systems.''')

    st.markdown("In fact, according to the Food and Agriculture Organization (FAO), 2 out of 10 Pinoys  or roughly, 16.6 million Pinoys, were undernourished from 2018 to 2020.")

    st.image("slides/stat1.png")

    st.markdown('''Using data from the World Bank's household surveys, we found that one in five Pinoys is not able to buy rice, protein or vegetables and fruits.
    ''')

    st.image("slides/stat2.png")

    st.markdown('''According to the Philippine Institute for Development Studies or PIDS, the cost of a considered healthy diet per person per day in the Philippines
    have been consistently rising from 230-250 Pesos in the last four years. That is almost half of the minimum daily wage in Manila. In a household of four members, that is 1000 pesos
    per family. Imagine if there is only one breadwinner.''')

    st.image("slides/stat3.png")

    st.markdown('''Ultimately, this study poses its value in touching not one but many of the Sustainable Development Goals of the United Nations such as
    SDG 1: No Poverty, SDG2: Zero Hunger, SDG 3: Good Health and Well-Being, and SDG 10: Reduced Inequalities.''')

    st.markdown('''The objectives of this sprint are:
    1. Identify which features drive food insecurity 
    2. Recommend targeted actionable plans''')
    


def dataset():
    st.markdown('''We obtained our dataset from the World Bank Microdata Library, which has a large collection of data on various topics, including food insecurity.
    Our dataset consists of responses to the “COVID-19 Households Survey 2020-2021” conducted in the Philippines.
    The raw dataset has 9,448 rows corresponding to the number of respondents and 217 columns corresponding to the questions asked or variables recorded about the respondents.
    ''')
    
    # Load data
    data = load_data()

    # Display data
    st.markdown("**The Data**")
    st.dataframe(data)
    st.markdown("Source: COVID-19 Households Survey 2020-2021, Philippines from World Bank. Descriptions of the variables are available [here](https://microdata.worldbank.org/index.php/catalog/4480/data-dictionary/F1?file_name=phl_hfs_r1_household_survey_data).")


def methodology():
    # Write the title
    st.title(
        "What we did"
    )

    st.image("slides/methodology.png")

    st.markdown("After obtaining the dataset, we moved on to data preprocessing. First, we performed data cleaning by removing columns that had more than 20% missing values and then removed the remaining rows that still had missing values. This reduced the number of rows from 9,448 to 7,589.")

    st.markdown("We kept as many features as possible, removing the ones that cannot be replicated in future data collections (i.e. pandemic-specific features). These can be categorized as demographics, household composition, assets and housing conditions, travel and transportation, financial situation and food access.")
    
    st.markdown('''For our target variable, we combined the responses to the following 5 food insecurity questions into a Food Insecurity Experience Scale (FIES). We considered individuals with 4-5 affirmative responses to be food insecure and those with fewer than that (0-3) as food secure. The total number of Food Insecure is only 22 percent and 78 percent are Food Secure so our dataset is imbalanced.
    During the last 12 months, was there a time when, because of lack of money or other resources:
    1. worried about food due to lack of resources?
    2. ate less due to lack of resources?
    3. no food due to lack of resources?
    4. hungry but no food due to lack of resources?
    5. no food for a day due to lack of resources?''')
 
    st.markdown("After acquiring and preprocessing our dataset, we split it into a training set (train-validation) and a hold-out set, using a 75 to 25 ratio.")

    st.markdown("Then, using the training set, we tried different combinations of categorical variable encoders (e.g. one-hot encoder and Ordinal encoder), scaling methods (e.g. StandardScaler/MinMaxScaler), and classifiers (e.g. KNN, Logistic Regression, Decision Trees, RandomForest, GradientBoosting, and XGBoost). To account for the imbalance in our dataset, we also tried oversampling techniques like SMOTE and ADASYN. We evaluated these combinations using Stratified K-fold cross-validation with 5 folds and optimized for F1 score while also preferring a high recall score.")
    
    st.markdown("After retraining our best model on the entire training set, we evaluated its performance on the hold-out set. We then used SHAP in interpreting the results.")
    

def final_model():
    # Write the title and the subheader
    st.title(
        "Predicting Food Insecurity in the Philippines"
    )

    st.markdown('''The final model we arrived at is a logistic regression model with a recall score of 71% which means 71% of those food insecure are predicted as such. The model has an F1 score of 57%.
    These are good scores compared to the proportion chance criterion of 27%. We focused on the recall because if someone is predicted as not food insecure when they are actually food insecure, these might bring worse consequences.
    For example, they might not receive government assistance because they were tagged food secure when they actually need help to have enough food.
    Thus, we wanted to maximize how many of the actual food insecure persons are predicted as food insecure.
    ''')

    st.subheader("Being able to buy food, financial worry and assets owned are the top features that contribute to food insecurity.")
    
    st.image("slides/feature_importance.png")

    st.markdown('''Let us look at two persons. Using Shapley values for local explainability, these are the features that contribute to person A and person B being food insecure or not.
    Person A has no dvd players, no bedrooms, is not too worried with finances and lives alone.
    Person B has no dvd players, is very worried with finances and is unable to buy rice in the past 7 days.''')
    
    st.markdown("Person A is food secure. While they have none of these mentioned assets which can push them to be classified as food insecure, this is balanced out by them not being too worried financially and living alone.")
    
    st.markdown("Person B on the other hand is food insecure. Having no assets, being very worried financially and being unable to buy rice define this.")
    
    st.markdown("You can check other participants' food security status using the machine learning app below.")

    # ML app
    model = pickle.load(open('streamlit_app/final_model.pkl', 'rb'))
    X_holdout = pd.read_csv('streamlit_app/holdout.csv', index_col=0).reset_index(drop=True)
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


def predict_if_fi(id):    
    observation = X_holdout.loc[id].values.reshape(1, -1)
    prediction_num = model.predict(observation)[0]
    pred_map = {1: 'Food Insecure', 0: 'Food Secure'}
    prediction = pred_map[prediction_num]
    return prediction

# Create a boolean state variable to control the visibility of the button
show_button = False

# Use st.checkbox() to create a checkbox to toggle the visibility of the button
show_button = st.checkbox("Go to Predicting Food Insecurity for the Prediction Button")

# Use an if statement to display the button based on the state of the checkbox
if show_button:
    if st.button("Predict", key="predict button"):
        output = predict_if_fi(choice)
    if output == 'Food Insecure':
        st.error('This person is Food Insecure', icon="🚨")
    elif output == 'Food Secure':
        st.success('This person is Food Secure!', icon="✅")

# TO ADD - force_plot, inputting own numbers
    

def recommendations():
    # Write the title
    st.title(
        "What We Can Do"
    )

    tab_reco1, tab_reco2, tab_reco3 = st.tabs(["Food Subsidies", "Cash Transfers", "Increasing the Minimum Daily Wage"])
    
    with tab_reco1:
            
            st.markdown("""Food subsidies are for immediately providing sufficient and nutritious food for vulnerable households.
            Usually, the government provides rice subsidies to specific populations such as government employees, farmers and recipients of the 4Ps.
            However, these might exclude people who are food insecure but not part of those groups. A model like ours might be more helpful in reaching those who are most in need.
            """)

            st.image("slides/food_subsidy.jpg")

    with tab_reco2:
            
            st.markdown("""If you are not eating well, it affects your ability to work and earn. Thus, it affects how you can provide things such as shelter, assets and of course, food for yourself and your family.
            Cash transfers can provide that initial lift to provide for the family, lessen their financial worries and empower people to make financial decisions.
            We have the Pantawid Pamilyang Pilipino Program. However, the qualifications to receive cash transfers only focuses on income. This is the same case for other ayuda programs.
            Using a model like ours that studies people more holistically can help identify the people who should be prioritized to receive help.
            """)     

            st.image("slides/4ps.png")

    with tab_reco3:
            
            st.markdown("""Currently, the minimum wage in the Philippines is barely enough for one person to afford three healthy meals every day. Increasing the minimum daily wage, even just to the requested P750, can help them afford nutritious meals more.
            """)

            st.image("slides/min_wage.jpg")


list_of_pages = [
    "Food Insecurity in the Philippines",
    "The COVID-19 Households Survey Dataset",
    "Methodology",
    "Predicting Food Insecurity",
    "What We Can Do"
]

st.sidebar.title(':scroll: Main Pages')
selection = st.sidebar.radio("Go to: ", list_of_pages)

if selection == "Food Insecurity in the Philippines":
    introduction()

elif selection == "The COVID-19 Households Survey Dataset":
    dataset()

elif selection == "Methodology":
    methodology()

elif selection == "Predicting Food Insecurity":
    final_model()

elif selection == "What We Can Do":
    recommendations()
