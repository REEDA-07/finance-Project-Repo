import streamlit as st
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings('ignore')

st.markdown("<h1 style = 'color: #FF9130; text-align: center; font-family:Copperplate Gothic '>FINANCIAL INCLUSION PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FF5B22; text-align: center; font-family: Copperplate Gothic'> By REEDA: Daintree Cohort </h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com (8).png', width = 700)

st.header('Project Background Information', divider = True)
st.write("The objectives of the predictive model include identifying underserved populations, predicting adoption of financial services, accurately assessing creditworthiness, implementing targeted interventions, optimizing resource allocation, and reducing overall financial exclusion rates. Additionally, it aims to promote inclusive growth, monitor progress and impact, ensure equitable access to financial opportunities, and foster sustainable socio-economic development.")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

data = pd.read_csv('Financial_inclusion_dataset.csv')
st.dataframe(data.drop('uniqueid', axis = 1))

st.sidebar.image('pngwing.com (1).png', caption = 'Welcome User')
st.sidebar.divider()
st.sidebar.markdown("<br>", unsafe_allow_html= True)

st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)

# Declare user Input variables 
st.sidebar.subheader('Input Variables', divider= True)
age_of_respo = st.sidebar.number_input('age_of_respondent', data['age_of_respondent'].min(), data['age_of_respondent'].max())
house_hold = st.sidebar.number_input('household_size', data['household_size'].min(), data['household_size'].max())
job = st.sidebar.selectbox('job_type', data['job_type'].unique())
edu = st.sidebar.selectbox('education_level', data['education_level'].unique())
mar_status = st.sidebar.selectbox('marital_status', data['marital_status'].unique())
count_ry = st.sidebar.selectbox('country', data['country'].unique())
location = st.sidebar.selectbox('location_type', data['location_type'].unique())
rship = st.sidebar.selectbox('relationship_with_head', data['relationship_with_head'].unique())


# display the users input
input_var = pd.DataFrame()
input_var['age_of_respondent'] = [age_of_respo]
input_var['household_size'] = [house_hold]
input_var['job_type'] = [job]
input_var['education_level'] = [edu]
input_var['marital_status'] = [mar_status]
input_var['country'] = [count_ry]
input_var['location_type'] = [location]
input_var['relationship_with_head'] = [rship]

st.markdown("<br>", unsafe_allow_html= True)
# display the users input variable 
st.subheader('Users Input Variables', divider= True)
st.dataframe(input_var)


job = joblib.load('job_type_encoder.pkl')
edu = joblib.load('education_level_encoder.pkl')
mar_status = joblib.load('marital_status_encoder.pkl')
count_ry = joblib.load('country_encoder.pkl')
location = joblib.load('location_type_encoder.pkl')
rship = joblib.load('relationship_with_head_encoder.pkl')



# transform the users input with the imported scalers 
input_var['job_type'] = job.transform(input_var[['job_type']])
input_var['education_level'] =  edu.transform(input_var[['education_level']])
input_var['marital_status'] = mar_status.transform(input_var[['marital_status']])
input_var['country'] = count_ry.transform(input_var[['country']])
input_var['location_type'] = location.transform(input_var[['location_type']])
input_var['relationship_with_head'] = rship.transform(input_var[['relationship_with_head']])

model = joblib.load('FinanceInclusionModel.pkl')
predicted = model.predict(input_var)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

if st.button('Predict Bank Account User'):
    if predicted == 0:
        st.error(' Individual does not have a bank account')
    else:
        st.success('Individual has a bank account')

