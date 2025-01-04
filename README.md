Fake News Classification Project
This project aims to classify news articles as either real or fake using machine learning techniques, with a Streamlit-based web interface for easy interaction.
Project Overview
The fake news classification system uses natural language processing and machine learning algorithms to analyze and categorize news articles. It includes data preprocessing, feature extraction, model training, evaluation components, and a user-friendly web interface.
Features
Data cleaning and preprocessing
Text vectorization using TF-IDF
Multiple classification models (Naive Bayes, Logistic Regression, Random Forest)
Model evaluation and comparison
Visualization of results
Streamlit-based web interface for real-time classification
Installation
Clone this repository
Install the required packages:
text
pip install -r requirements.txt
Download NLTK data:
python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Usage
Prepare your dataset:
Place your 'True.csv' and 'Fake.csv' files in the project directory
Run the data preprocessing and model training script:
text
python data_cleaning.py
Launch the Streamlit app:
text
streamlit run app.py
Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501)
File Descriptions
data_cleaning.py: Script for data preprocessing, model training, and evaluation
app.py: Streamlit application for the web interface
fake_news_model.pkl: Saved best-performing model
tfidf_vectorizer.pkl: Saved TF-IDF vectorizer
Model Performance
The script evaluates multiple models:
Naive Bayes
Logistic Regression
Random Forest
Performance metrics include accuracy, precision, recall, and F1-score.
Visualizations
The project generates several visualizations:
Model performance comparison
Dataset distribution
Text length distribution
Learning curves
Feature importance (for Random Forest)
These visualizations can be viewed in the Streamlit app.
Web Interface
The Streamlit-based web interface allows users to:
Input news articles for real-time classification
View the classification result and confidence score
Explore model performance metrics and visualizations
Future Improvements
Implement cross-validation for more robust model evaluation
Experiment with deep learning models (e.g., LSTM, BERT)
Add more features (e.g., sentiment analysis, named entity recognition)
Enhance the web interface with more interactive elements
Implement user feedback collection for continuous model improvement
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is open source and available under the MIT License.