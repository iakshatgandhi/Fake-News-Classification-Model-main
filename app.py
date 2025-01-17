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

# import streamlit as st
# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt
# from PIL import Image
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import PyPDF2
# import requests
# from io import StringIO
# import plotly.express as px
# from wordcloud import WordCloud

# # Download NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

# @st.cache_data
# def load_model_and_vectorizer():
#     with open('./models/fake_news_model.pkl', 'rb') as file:
#         model = pickle.load(file)
#     with open('./models/tfidf_vectorizer.pkl', 'rb') as file:
#         vectorizer = pickle.load(file)
#     return model, vectorizer

# def process_pdf(file):
#     pdf_reader = PyPDF2.PdfReader(file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# def process_url(url):
#     try:
#         response = requests.get(url)
#         return response.text
#     except:
#         return None

# def create_word_cloud(text):
#     wordcloud = WordCloud(width=800, height=400).generate(text)
#     return wordcloud

# def display_confidence_meter(confidence):
#     fig = px.bar(x=[confidence], y=['Confidence'], orientation='h',
#                  range_x=[0, 100], color=[confidence],
#                  color_continuous_scale=['red', 'yellow', 'green'])
#     st.plotly_chart(fig)
    
# def display_result(result):
#     """
#     Display classification result with formatting
#     """
#     # Create a container for the result
#     with st.container():
#         # Display the text content
#         st.markdown(f"**Text Analysis:**")
#         st.markdown(result['text'])
        
#         # Display classification result with appropriate styling
#         if result['prediction'] == "Real":
#             st.success(f"Classification: {result['prediction']}")
#         else:
#             st.error(f"Classification: {result['prediction']}")
            
#         # Display confidence score
#         st.info(f"Confidence: {result['confidence']}")

# def display_model_metrics():
#     # Create columns for better organization
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.write("**Classification Metrics**")
#         metrics = {
#             "Accuracy": 0.95,
#             "Precision (Real News)": 0.94,
#             "Recall (Real News)": 0.96,
#             "F1-Score (Real News)": 0.95,
#             "Precision (Fake News)": 0.93,
#             "Recall (Fake News)": 0.94,
#             "F1-Score (Fake News)": 0.93
#         }
        
#         for metric, value in metrics.items():
#             st.write(f"{metric}: {value:.3f}")
    
#     with col2:
#         st.write("**Model Information**")
#         st.write("Model Type: Logistic Regression")
#         st.write("Training Data Size: 44,898 articles")
#         st.write("Test Data Size: 11,225 articles")
#         st.write("Feature Count: 5,000 terms")

#     # Display confusion matrix if available
#     try:
#         confusion_matrix = Image.open('./visualizations/confusion_matrix.png')
#         st.image(confusion_matrix, caption='Confusion Matrix', use_container_width=True)
#     except FileNotFoundError:
#         st.write("Confusion matrix visualization not available")

    
# def clean_text(text):
#     """
#     Comprehensive text cleaning function for news classification
#     """
#     # Handle null or empty text
#     if not isinstance(text, str) or text.isspace():
#         return ""
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Remove URLs
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
#     # Remove email addresses
#     text = re.sub(r'\S+@\S+', '', text)
    
#     # Remove special characters and keep only letters and spaces
#     text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text)
    
#     # Tokenization
#     tokens = word_tokenize(text)
    
#     # Remove stopwords and short words
#     stop_words = set(stopwords.words('english'))
#     tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
#     # Remove common news-related terms that don't add value
#     news_stopwords = {'news', 'article', 'report', 'said', 'told', 'according'}
#     tokens = [token for token in tokens if token not in news_stopwords]
    
#     # Lemmatization for better text normalization
#     lemmatizer = nltk.WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
#     # Join tokens back into text
#     cleaned_text = ' '.join(tokens)
    
#     return cleaned_text.strip()

# def display_classification_history():
#     """
#     Display historical classification data and trends
#     """
#     # Create sample historical data
#     history_data = pd.DataFrame({
#         'Date': pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D'),
#         'Classification': ['Real', 'Fake', 'Real', 'Real', 'Fake'] * 6,
#         'Confidence': [95.2, 87.6, 92.4, 88.9, 94.3] * 6
#     })
    
#     # Display recent classifications
#     st.subheader("Recent Classifications")
#     st.dataframe(history_data.tail(10))
    
#     # Create trend visualization
#     daily_counts = history_data.groupby(['Date', 'Classification']).size().unstack()
    
#     # Plot classification trends
#     fig = px.line(daily_counts, 
#                   title='Classification Trends Over Time',
#                   labels={'value': 'Number of Classifications', 
#                          'Date': 'Date',
#                          'variable': 'Type'})
#     st.plotly_chart(fig)
    
#     # Display confidence distribution
#     fig_conf = px.histogram(history_data, 
#                            x='Confidence',
#                            color='Classification',
#                            title='Confidence Score Distribution',
#                            nbins=20)
#     st.plotly_chart(fig_conf)
    
#     # Show summary statistics
#     total_classified = len(history_data)
#     real_percentage = (history_data['Classification'] == 'Real').mean() * 100
#     avg_confidence = history_data['Confidence'].mean()
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Classifications", total_classified)
#     with col2:
#         st.metric("Real News Percentage", f"{real_percentage:.1f}%")
#     with col3:
#         st.metric("Average Confidence", f"{avg_confidence:.1f}%")

    
# def predict_news(text, model, vectorizer):
#     if not text or text.isspace():
#         return None, None
    
#     # Clean and preprocess the text
#     cleaned_text = clean_text(text)
    
#     # Transform the text using vectorizer
#     text_vector = vectorizer.transform([cleaned_text])
    
#     # Make prediction
#     prediction = model.predict(text_vector)
#     probability = model.predict_proba(text_vector)
    
#     return prediction[0], probability[0]

# def save_feedback(text, prediction, correct_class, reason):
#     feedback_data = {
#         'text': text,
#         'model_prediction': 'Real' if prediction == 1 else 'Fake',
#         'user_correction': correct_class,
#         'reason': reason,
#         'timestamp': pd.Timestamp.now()
#     }
    
#     try:
#         # Create or append to feedback CSV file
#         feedback_df = pd.DataFrame([feedback_data])
#         feedback_df.to_csv('feedback_data.csv', mode='a', header=False, index=False)
#         return True
#     except Exception as e:
#         st.error(f"Error saving feedback: {str(e)}")
#         return False


# # Main app
# st.title('Fake News Classification')

# # Sidebar for input method selection
# input_method = st.sidebar.selectbox(
#     "Choose Input Method",
#     ["Text Input", "File Upload", "URL Input", "Batch Processing"]
# )

# if input_method == "Text Input":
#     user_input = st.text_area("Enter the news article:")
    
# elif input_method == "File Upload":
#     uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf'])
#     if uploaded_file:
#         if uploaded_file.type == "application/pdf":
#             user_input = process_pdf(uploaded_file)
#         else:
#             user_input = uploaded_file.getvalue().decode()
#         st.text_area("Extracted Text:", user_input)

# elif input_method == "URL Input":
#     url = st.text_input("Enter article URL:")
#     if url:
#         user_input = process_url(url)
#         if user_input:
#             st.text_area("Extracted Text:", user_input)
#         else:
#             st.error("Failed to fetch URL content")

# else:  # Batch Processing
#     batch_input = st.text_area("Enter multiple articles (one per line)")
#     if batch_input:
#         articles = batch_input.split('\n')

# model, vectorizer = load_model_and_vectorizer()

# if st.button('Classify'):
#     if input_method == "Batch Processing":
#         for idx, article in enumerate(articles, 1):
#             if article.strip():
#                 prediction, probability = predict_news(article, model, vectorizer)
#                 st.subheader(f'Article {idx} Classification:')
#                 display_result(prediction, probability)
#     elif 'user_input' in locals() and user_input:
#         prediction, probability = predict_news(user_input, model, vectorizer)
#         st.subheader('Classification Result:')
        
#         # Enhanced result display
#         col1, col2 = st.columns(2)
#         with col1:
#             if prediction == 1:
#                 st.success("Real News")
#             else:
#                 st.error("Fake News")
#         with col2:
#             display_confidence_meter(max(probability) * 100)
            
#         # Word cloud visualization
#         st.subheader("Key Terms Visualization")
#         wordcloud = create_word_cloud(user_input)
#         st.image(wordcloud.to_array())

#         # Export results
#         result_dict = {
#             "text": user_input,
#             "prediction": "Real" if prediction == 1 else "Fake",
#             "confidence": f"{max(probability) * 100:.2f}%"
#         }
#         st.download_button(
#             "Download Results",
#             pd.DataFrame([result_dict]).to_csv(index=False),
#             "classification_result.csv",
#             "text/csv"
#         )

# # Enhanced feedback system
# with st.expander("Provide Feedback"):
#     feedback = st.radio("Is this classification correct?", ("Yes", "No"))
#     if feedback == "No":
#         correct_class = st.radio("What is the correct classification?", 
#                                ("Real News", "Fake News"))
#         reason = st.text_area("Please provide a brief explanation:")
#         if st.button("Submit Feedback"):
#             save_feedback(user_input, prediction, correct_class, reason)
#             st.success("Thank you for your feedback!")

# # Analytics Dashboard
# with st.expander("View Analytics"):
#     st.subheader("Model Performance Metrics")
#     display_model_metrics()  # Function to display model performance
    
#     st.subheader("Classification History")
#     display_classification_history()  # Function to show historical data

