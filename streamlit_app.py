import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to load data
@st.cache
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

# Streamlit UI
st.title('Autism Spectrum Disorder Detection')

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
    except Exception as e:
        st.write(f'Error: {e}')
else:
    st.write("Please upload training and test data files.")
