# import streamlit as st
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
# import numpy as np
# import pandas as pd
# import pickle
# from tensorflow.keras.models import load_model

# # Load the trained model
# model = load_model(r"C:\Users\ronga\Desktop\ANN CLASSIFICATION(BANK)\model.h5")
# st.write("✅ Model loaded successfully!")

# # Load the pickle files
# with open(r"C:\Users\ronga\Desktop\ANN CLASSIFICATION(BANK)\label_encode_gen.pkl", "rb") as file:
#     label_encode_gen = pickle.load(file)

# with open(r"C:\Users\ronga\Desktop\ANN CLASSIFICATION(BANK)\one_hot_geo.pkl", "rb") as file:
#     one_hot_geo = pickle.load(file)

# with open(r"C:\Users\ronga\Desktop\ANN CLASSIFICATION(BANK)\scaler.pkl", "rb") as file:
#     scaler = pickle.load(file)

# # Streamlit App Title
# st.title("Customer Churn Prediction")

# # User Input Fields
# geography = st.selectbox("Geography", one_hot_geo.categories_[0])
# gender = st.selectbox("Gender", label_encode_gen.classes_)
# age = st.slider("Age", 18, 92)
# balance = st.number_input("Balance", min_value=0.0, value=0.0)
# credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
# estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=0.0)
# tenure = st.slider("Tenure", 0, 10, value=5)
# num_of_products = st.slider("Num of Products", 1, 4, value=2)
# has_cr_card = st.selectbox("Has Credit Card", [0, 1])
# is_active_member = st.selectbox("Is Active Member", [0, 1])

# # Create a dictionary for input
# user_input = {
#     "CreditScore": credit_score,
#     "Gender": gender,
#     "Age": age,
#     "Tenure": tenure,
#     "Balance": balance,
#     "NumOfProducts": num_of_products,
#     "HasCrCard": has_cr_card,
#     "IsActiveMember": is_active_member,
#     "EstimatedSalary": estimated_salary,
#     "Geography": geography
# }

# # Convert dictionary to DataFrame
# input_df = pd.DataFrame([user_input])

# # Encode categorical features
# geo_encoded = one_hot_geo.transform(input_df[['Geography']]).toarray()
# geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_geo.get_feature_names_out(['Geography']))

# # Correct Gender Encoding
# input_df['Gender'] = label_encode_gen.transform(input_df['Gender'])  # ✅ Fixed

# # Drop original categorical column and add encoded features
# input_df = input_df.drop(columns=['Geography'])
# input_df = pd.concat([input_df, geo_encoded_df], axis=1)

# # Scale numerical features
# input_scaled = scaler.transform(input_df)

# # Convert to NumPy array before prediction
# input_array = np.array(input_scaled)

# # Predict using model
# prediction = model.predict(input_array)

# # Show result
# st.write(f"Prediction Probability: **{prediction[0][0]:.4f}**")

# if prediction[0][0] > 0.5:  # ✅ Fixed condition
#     st.write("🔴 The customer is **likely to churn (exit)**.")
# else:
#     st.write("🟢 The customer is **not likely to churn (stay)**.")  



##WITH VOICE FOR THE MESSAEGE
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd
import pickle
import pyttsx3  
import threading  
from tensorflow.keras.models import load_model

# Function to handle text-to-speech
def speak(text):
    """Creates a new pyttsx3 engine instance for every call to ensure speech works every time."""
    def run():
        engine = pyttsx3.init()  # ✅ Create a new engine instance
        engine.say(text)
        engine.runAndWait()
    
    threading.Thread(target=run, daemon=True).start()  # ✅ Run in a separate thread

# Load the trained model
model = load_model(r"C:\Users\ronga\Desktop\ANN CLASSIFICATION(BANK)\model.h5")
st.write("✅ Model loaded successfully!")

# Load the pickle files
with open(r"C:\Users\ronga\Desktop\ANN CLASSIFICATION(BANK)\label_encode_gen.pkl", "rb") as file:
    label_encode_gen = pickle.load(file)

with open(r"C:\Users\ronga\Desktop\ANN CLASSIFICATION(BANK)\one_hot_geo.pkl", "rb") as file:
    one_hot_geo = pickle.load(file)

with open(r"C:\Users\ronga\Desktop\ANN CLASSIFICATION(BANK)\scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit App Title
st.title("Customer Churn Prediction")

# User Input Fields
geography = st.selectbox("Geography", one_hot_geo.categories_[0])
gender = st.selectbox("Gender", label_encode_gen.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0, value=0.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=0.0)
tenure = st.slider("Tenure", 0, 10, value=5)
num_of_products = st.slider("Num of Products", 1, 4, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Create a dictionary for input
user_input = {
    "CreditScore": credit_score,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
    "Geography": geography
}

# Convert dictionary to DataFrame
input_df = pd.DataFrame([user_input])

# Encode categorical features
geo_encoded = one_hot_geo.transform(input_df[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_geo.get_feature_names_out(['Geography']))

# Correct Gender Encoding
input_df['Gender'] = label_encode_gen.transform(input_df['Gender'])  # ✅ Fixed

# Drop original categorical column and add encoded features
input_df = input_df.drop(columns=['Geography'])
input_df = pd.concat([input_df, geo_encoded_df], axis=1)

# Scale numerical features
input_scaled = scaler.transform(input_df)

# Convert to NumPy array before prediction
input_array = np.array(input_scaled)

# Predict using model
prediction = model.predict(input_array)

# Show result
st.write(f"Prediction Probability: **{prediction[0][0]:.4f}**")

# Display prediction result with voice
if prediction[0][0] > 0.5:  
    message = "🔴 The customer is likely to exit from the bank."
    st.write(message)
else:
    message = "🟢 The customer is likely to continue in the bank."
    st.write(message)

# Speak the result (Runs in a separate thread)
speak(message)

