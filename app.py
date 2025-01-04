import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from utils.utils import save_feedback

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the model and vectorizer
def load_model_and_vectorizer():
    with open('./models/fake_news_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('./models/tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

# Function to clean text
def clean_text(text):
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Function to make predictions
def predict_news(text, model, vectorizer):
    if not text or text.isspace():
        return "Invalid Input", [0, 0]

    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)
    probability = model.predict_proba(text_vector)
    return prediction[0], probability[0]

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

st.title('Fake News Classification')

# User input
user_input = st.text_area("Enter the news article:")

if st.button('Classify'):
    if user_input:
        prediction, probability = predict_news(user_input, model, vectorizer)

        st.subheader('Classification Result:')
        if prediction == 1:
            st.success(f"This article is classified as: Real News")
        else:
            st.error(f"This article is classified as: Fake News")

        st.write(f"Confidence: {max(probability) * 100:.2f}%")
    # else:
    #     st.warning('Please enter a news article to classify.')
    
            # Feedback mechanism
feedback = st.radio("Is this classification correct?", ("Yes", "No"))
if feedback == "No":
    correct_class = st.radio("What is the correct classification?", ("Real News", "Fake News"))
    reason = st.text_area("Please provide a brief explanation for the correction:")
    if st.button("Submit Feedback"):
        # Here you would typically save this feedback for later model improvement
        st.success("Thank you for your feedback! This will help improve our model.")
        
        # For demonstration purposes, let's print the feedback
        st.write("Feedback recorded:")
        st.write(f"User input: {user_input}")
        st.write(f"Model prediction: {'Real' if prediction == 1 else 'Fake'}")
        st.write(f"User correction: {correct_class}")
        st.write(f"Reason: {reason}")
        
        # In a real scenario, you would save this data for retraining
        # save_feedback(user_input, prediction, correct_class, reason)


# Display visualizations
st.subheader('Project Visualizations')

# Model Performance Comparison
st.write('Model Performance Comparison')
try:
    performance_img = Image.open('./visualizations/model_performance.png')
    st.image(performance_img, use_container_width=True)
except FileNotFoundError:
    st.write("Model performance image not found.")

# Dataset Distribution
st.write('Dataset Distribution')
try:
    distribution_img = Image.open('./visualizations/dataset_distribution.png')
    st.image(distribution_img, use_container_width=True)
except FileNotFoundError:
    st.write("Dataset distribution image not found.")

# Text Length Distribution
st.write('Text Length Distribution')
try:
    length_dist_img = Image.open('./visualizations/text_length_distribution.png')
    st.image(length_dist_img, use_container_width=True)
except FileNotFoundError:
    st.write("Text length distribution image not found.")

# Learning Curves
st.write('Learning Curves')
models = ['Naive Bayes', 'Logistic Regression', 'Random Forest']
for model_name in models:
    try:
        # learning_curve_img = Image.open(f'./visualizations/learning_curve_{model_name}.png')
        learning_curve_img = Image.open(f'visualizations/learning_curve_{model_name}.png')

        st.image(learning_curve_img, caption=f'Learning Curve - {model_name}', use_container_width=True)
    except FileNotFoundError:
        st.write(f"Learning curve image for {model_name} not found.")

# Feature Importance
st.write('Feature Importance')
try:
    feature_importance_img = Image.open('./visualizations/feature_importance.png')
    st.image(feature_importance_img, use_container_width=True)
except FileNotFoundError:
    st.write("Feature importance image not found.")
