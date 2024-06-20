import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load or create a model (For demonstration, we use a dummy model)
# In practice, you should load a pre-trained model
def load_model():
    # Example: A dummy random forest classifier
    model = RandomForestClassifier()
    return model

# Load model
model = load_model()

# Function to preprocess user inputs
def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Standardize numeric features (if any)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df_filled[numeric_cols] = scaler.fit_transform(df_filled[numeric_cols])
    
    return df_filled

# Streamlit app
st.title("Autism Spectrum Disorder Prediction")

st.sidebar.header("User Input Features")
st.sidebar.write("Please provide the following details:")

age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
ethnicity = st.sidebar.selectbox("Ethnicity", ["White", "Black", "Asian", "Hispanic", "Other"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
jaundice = st.sidebar.selectbox("Jaundice", ["Yes", "No"])
used_app_before = st.sidebar.selectbox("Used App/Website Before", ["Yes", "No"])
relation = st.sidebar.selectbox("Relation", ["Parent", "Self", "Caretaker", "Other"])

questions = {
    "Q1": "I often notice small sounds when others do not:",
    "Q2": "I find it easy to do more than one thing at once:",
    "Q3": "I find it easy to read between the lines when someone is talking to me:",
    "Q4": "When I’m reading a story, I find it difficult to work out the character’s intentions:",
    "Q5": "I find it easy to work out what someone is thinking or feeling just by looking at their face:",
    "Q6": "I usually concentrate more on the whole picture, rather than the small details:",
    "Q7": "If there is an interruption, I can switch back to what I was doing very quickly:",
    "Q8": "I know how to tell if someone listening to me is getting bored:",
    "Q9": "I like to collect information about categories of things:",
    "Q10": "I find it difficult to work out people's intentions:"
}

responses = {}
for q_key, q_text in questions.items():
    responses[q_key] = st.sidebar.selectbox(q_text, ["Yes", "No"])

# Prepare input data
input_data = {
    'age': age,
    'ethnicity': ethnicity,
    'gender': gender,
    'jaundice': jaundice,
    'used_app_before': used_app_before,
    'relation': relation,
    **responses
}

if st.button('Predict'):
    # Preprocess input data
    preprocessed_data = preprocess_input(input_data)
    
    # Make prediction (For demonstration, using a dummy prediction)
    # In practice, you should use `model.predict(preprocessed_data)`
    prediction = model.predict(preprocessed_data)
    
    # Display prediction result
    if prediction[0] == 1:
        st.write("The model predicts: High likelihood of Autism Spectrum Disorder.")
    else:
        st.write("The model predicts: Low likelihood of Autism Spectrum Disorder.")
