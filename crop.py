import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.image('images/banner.png', use_column_width=True)

# Function to load data
@st.cache_data
def load_data():
    crop_dataset = pd.read_csv('Crop_recommendation.csv')
    return crop_dataset

# Function to preprocess data
def preprocess_data(data):
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    X = data.drop(columns='label', axis=1)
    Y = data['label']
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    return standardized_data, Y, scaler, le.classes_

# Function to train model
def train_model(X_train, Y_train):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, Y_train)
    return rf_model

# Function to predict
def predict_crop(input_data, scaler, rf_model, class_labels):
    std_data = scaler.transform([input_data]) 
    prediction = rf_model.predict(std_data)
    return class_labels[prediction][0]

def main(data):
    # st.title("Crop Recommendation App")

    # Navbar
    st.markdown("""
        <style>
            .navbar {
                display: flex;
                justify-content: center;
                padding: 1rem;
                background-color: #4CAF50;
                color: white;
                margin-bottom: 1rem;
            }
            .navbar-title {
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="navbar"><div class="navbar-title">🌱AGRICONNECT-CROP RECOMMENDATION APP🌿</div></div>', unsafe_allow_html=True)

    # Preprocess data
    X, Y, scaler, class_labels = preprocess_data(data)

    # Train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Train model
    rf_model = train_model(X_train, Y_train)

    # Display accuracy in sidebar table
    st.sidebar.title("Accuracy Scores:")
    st.sidebar.table({
        'Accuracy Score': ['Training', 'Testing'],
        'Score': [accuracy_score(rf_model.predict(X_train), Y_train), accuracy_score(rf_model.predict(X_test), Y_test)]
    })
    st.sidebar.title("Algorithm used:")
    st.sidebar.success("RandomForest")

    st.sidebar.subheader("Distribution of Crops")
    crop_counts = data['label'].value_counts()
    st.sidebar.bar_chart(crop_counts)

    


    # User input
    st.markdown("<h2 style='text-align: center; padding-bottom: 20px;'>🌤️Enter Environmental Parameters❄️</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("N (Nitrogen)", min_value=0.0, max_value=120.0, value=50.0, step=0.1, format="%.1f", help="Enter value between 0 to 100")
        P = st.number_input("P (Phosphorus)", min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f", help="Enter value between 0 to 100")
        K = st.number_input("K (Potassium)", min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f", help="Enter value between 0 to 100")
        rainfall = st.number_input("Rainfall", min_value=0.0, max_value=500.0, value=100.0, step=0.1, format="%.1f", help="Enter value between 0 to 500")
    with col2:
        temperature = st.number_input("Temperature", min_value=0.0, max_value=100.0, value=25.0, step=0.1, format="%.1f", help="Enter value between 0 to 100")
        humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f", help="Enter value between 0 to 100")
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1, format="%.1f", help="Enter value between 0 to 14")

    # Predict button
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        if st.button("Predict", key="predict_button"):
            input_data = (N, P, K, temperature, humidity, ph, rainfall)
            recommended_crop = predict_crop(input_data, scaler, rf_model, class_labels)
            st.write("### Recommendation")
            st.write(f"## Recommended Crop: {recommended_crop}")
            # Display image of the crop
            image_path = f"images/{recommended_crop}.jpg"  # Assuming the images are stored in an "images" folder
            st.image(image_path, width=200)

        # Apply CSS to set the width of the button
        st.markdown(
            """
            <style>
                div[data-testid="stButton"] > button {
                width: 200px;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

    # Footer
    st.markdown('<hr>', unsafe_allow_html=True)
    # Footer with your name and GitHub profile link
    st.write('<center>App Developed & Crafted by <a href="https://github.com/Dinesh-kumar-M-2002">Dinesh Kumar M</a>👀</center>', unsafe_allow_html=True)

if __name__ == "__main__":
    # Load data
    data = load_data()
    main(data)
