import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import NotFittedError

# Define credentials
CREDENTIALS = {
    "user1": "password1",
    "user2": "password2"
}

# Function to load data
@st.cache_data
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

# Function to preprocess data
def preprocess_data(train_data):
    X = train_data.drop('Class/ASD', axis=1)
    y = train_data['Class/ASD']
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, y, preprocessor

# Function to split data
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, matrix

# Login function with debug
def login():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        st.write(f"Entered Username: {username}")  # Debug
        st.write(f"Entered Password: {password}")  # Debug
        if username in CREDENTIALS and CREDENTIALS[username] == password:
            st.success("Logged in successfully!")
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password")

# Streamlit UI
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
else:
    st.title('Autism Spectrum Disorder Detection')

    # The rest of your code...

    uploaded_train_file = st.file_uploader("Choose a CSV file for training", type="csv")
    uploaded_test_file = st.file_uploader("Choose a CSV file for testing", type="csv")

    if uploaded_train_file and uploaded_test_file:
        train_data, test_data = load_data(uploaded_train_file, uploaded_test_file)
        
        st.write("Training Data Sample:")
        st.write(train_data.head())
        
        st.write("Test Data Sample:")
        st.write(test_data.head())
        
        X, y, preprocessor = preprocess_data(train_data)
        
        # Debug: Check the type and shape of X and y
        st.write(f'X type: {type(X)}, shape: {X.shape}')
        st.write(f'y type: {type(y)}, shape: {y.shape}, unique values: {np.unique(y)}')
        
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Debug: Check the type and shape of X_train and y_train
        st.write(f'X_train type: {type(X_train)}, shape: {X_train.shape}')
        st.write(f'y_train type: {type(y_train)}, shape: {y_train.shape}')
        
        try:
            model = train_model(X_train, y_train)
            accuracy, report, matrix = evaluate_model(model, X_test, y_test)
            
            st.write(f'Model Accuracy: {accuracy}')
            st.write("Classification Report:")
            st.text(report)
            st.write("Confusion Matrix:")
            st.write(matrix)
            
            # Save the trained model and preprocessor
            st.session_state.model = model
            st.session_state.preprocessor = preprocessor
            
        except Exception as e:
            st.write(f'Error during training: {e}')
    
    # Prediction on new data
    if st.button("Make Prediction"):
        if 'model' in st.session_state and 'preprocessor' in st.session_state:
            try:
                # Example new data for prediction
                new_data = {
                    "Feature1": [value1],
                    "Feature2": [value2],
                    # add more features as per your dataset
                }
                new_df = pd.DataFrame(new_data)
                preprocessed_data = st.session_state.preprocessor.transform(new_df)
                prediction = st.session_state.model.predict(preprocessed_data)
                st.write(f'Prediction: {prediction}')
            except NotFittedError as e:
                st.write(f'Error: Model not fitted properly. {e}')
            except Exception as e:
                st.write(f'Error: {e}')
        else:
            st.write("Model not trained yet. Please upload training data and train the model.")
