import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Customizing app appearance with Streamlit themes
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the Encoders & Scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.write("### Hello, Rakesh! Let's predict if a customer is likely to churn.")

# User input section with styled headers
st.markdown("#### Customer Information")
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, help="Select the customer's age.")
balance = st.number_input('Balance', min_value=0.0, help="Enter the current account balance.")
credit_score = st.number_input('Credit Score', min_value=0.0, help="Enter the credit score.")
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, help="Enter the estimated salary.")
tenure = st.slider('Tenure', 0, 10, help="Enter the number of years the customer has been with the bank.")
num_of_products = st.slider('Number of Products', 1, 4, help="Enter the number of products the customer has.")
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# Onehot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine both DataFrames
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict the output
prediction = model.predict(input_data_scaled)
predict_proba = prediction[0][0]

# Display prediction results with stylized feedback
st.markdown("#### Prediction Result")
if predict_proba > 0.5:
    st.markdown("<h3 style='color: red;'>The customer is likely to churn.</h3>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='color: green;'>The customer is not likely to churn.</h3>", unsafe_allow_html=True)

st.write(f"**Prediction Probability:** {predict_proba:.2%}")

# Custom footer
st.markdown("---")
st.markdown("<footer style='text-align: center; color: gray;'>Built by Rakesh | Powered by Streamlit & TensorFlow</footer>", unsafe_allow_html=True)

'''import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

#Load the trained model
model = tf.keras.models.load_model('model.h5')

#Load the Encoders & Scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

##Streamlit app
st.title('Customer Churn Prediction')

#User input 
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is active member',[0,1])

#Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]], 
    'Age': [age],
    'Tenure': [tenure], 
    'Balance': [balance], 
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card], 
    'IsActiveMember': [is_active_member], 
    'EstimatedSalary': [estimated_salary],
})

#Onehot encoded 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine both the DataFrames
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

#Predict the output
prediction = model.predict(input_data_scaled)
predict_proba= prediction[0][0]

if predict_proba>0.5:
    st.write('The customer is likely to Churn')
else:
    st.write('The customer is not likely to Churn')

st.title("Hello Rakesh!!")'''